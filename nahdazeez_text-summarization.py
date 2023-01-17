
import numpy as np
import pandas as pd
import os
import json
import glob
import sys
sys.path.insert(0, "../")
import re
import os

pd.set_option("display.max_colwidth", 100000) # Extend the display width to prevent split functions to not cover full text


# NLP libraries
import nltk
nltk.download('wordnet')
nltk.download('punkt')     
nltk.download('stopwords')    
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer 
from collections import Counter
import matplotlib.pyplot as plt

import re,string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
biorxiv_clean = pd.read_csv("../input/clean-input-file/biorxiv_clean.csv")
clean_pmc = pd.read_csv("../input/clean-input-file/clean_pmc.csv")
clean_pmc.head(1)
#clean_pmc.shape
# function to plot the frequency of words in a list
def plot_word_freq(list_words):
    remove_stop_words = [word for word in list_words if word not in STOPWORDS] #Remove stop words
    remove_numbers = [''.join(x for x in i if x.isalpha()) for i in remove_stop_words] # Remove numbers
    remove_empty_string = ' '.join(remove_numbers).split() # Remove empty strings
    counts = dict(Counter(remove_empty_string)) #Count the occurence of each string
    popular_words = sorted(counts, key = counts.get, reverse = True) #sort by number of occurence
    plt.barh(range(20), [counts[w] for w in reversed(popular_words[0:20])]) #plot horizontal bar graph
    plt.yticks([x  for x in range(20)], reversed(popular_words[0:20]))
    plt.show()
#Function to clean the text 
def clean_text(text):
#    text_clean = text.lower()
    text_clean = text.replace("'", '')                  #remove single quotes  
    text_clean = re.sub('[!#?%*&$)@^(.:";]', '', text_clean)       #remove punctuation
    text_clean = text_clean.replace('\\n', ' ') 
#     text_clean = re.sub(r'\d+', '', text_clean)
    text_clean = text_clean.replace('\n', ' ')        #remove \\n
    text_clean = re.sub(r"\b[a-zA-Z]\b",'',text_clean)        #remove single letters
    text_clean = re.sub("(^|\W)\d+($|\W)", " ", text_clean)   #remove whitespace and numbers
    text_clean = text_clean.replace('introduction', '')       #remove 'introduction'
    text_clean = text_clean.replace('background', '')         #remove 'background'
    text_clean = text_clean.replace('abstract', '')           #remove 'abstract'
    text_clean = text_clean.strip()                           #remove whitespace
    return text_clean


def replace_tags(txt):
    tags = ['[', ']', '{', '}', 'author/funder', 'permission', 'No reuse allowed withou', 'All rights reserved',
           'medRxiv','license','preprint','doi:','copyright holder','perpetuity','. CC-BY-NC-ND 4.0 International','license',
           'It is made available under a author/funder','It is made available under a',
           'CC-BY-NC-ND 4.0 International license is made available under a The copyright holder for this preprint  is the author/funder.']

    for tag in tags:
        txt = txt.replace(tag, '')
        return txt
    
def regex_exp(txt):
        pattern =re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        regex_bracket_1 = "\(.*?\)|\s-\s.*"
        regex_bracket_2 = "\[.*?\]|\s-\s.*"
        txt =  pattern.sub('', txt)
        txt = re.sub(regex_bracket_1,'', txt)
        txt = re.sub(regex_bracket_2,'', txt)
        txt = re.compile('doi:').sub('', txt)
        txt = re.compile('medRxiv:').sub('', txt)
        txt = re.compile('.   medRxiv preprint').sub('', txt)
        return txt
## remove line from text if it contains certain stopwords:
def should_remove_line(line, stop_words):
    return any([word in line for word in stop_words])

# Function for tokenization, lemmatization
def tokenSentence(sentence):
    sent_str = str(sentence)
    stopwords = set(STOPWORDS)
    
    additional_words = set(['author/funder', 'permission', 'No reuse allowed without', 'All rights reserved','bioRxiv',
           'medRxiv','license','preprint','doi:','copyright holder','perpetuity','. CC-BY-NC-ND 4.0 International',
            'https://doi.org','medRxiv preprint','It is made available under a author/funder'])

    sentence = replace_tags(sent_str)
    sentence = regex_exp(sentence)
    clean_sentence = clean_text(sentence)
    
    x = nltk.sent_tokenize(clean_sentence)

    clean_line = []

    for line in x:
        if not(should_remove_line(line, additional_words)):
            clean_line.append(line)
            
    token_words=word_tokenize(str(clean_line)) #Tokenize
    
#     #adding additional stopwords -- this is done after an initial look at the wordcloud
    additional_words_2 = {'available','covid','corona','nCov','use','show','case'}
    STOPWORDS.update(additional_words_2)
    remove_stop_words = [word for word in token_words if word not in STOPWORDS] #Remove stop words
    token_words = []
    
    for word in remove_stop_words:
        token_words.append(WordNetLemmatizer().lemmatize(word, pos='v')) #lemmatization
    return token_words
# Function for Stemming
def stemSentence(sentence):
   
    porter=PorterStemmer()
    wthout_stem = []
    stem_sentence=[]
    clean_sentence = clean_text(sentence)
    token_words=word_tokenize(clean_sentence) #Tokenize 
    remove_stop_words = [word for word in token_words if word not in stopwords.words("english")] #Remove stop words
    for word in remove_stop_words:
        stem_sentence.append(porter.stem(WordNetLemmatizer().lemmatize(word, pos='v'))) #Stemming after Lemmatization
        #stem_sentence.append(" ")
    #return "".join(stem_sentence)
    return stem_sentence  
## Function to find paper id's of text containing words in a list
def filter_papers_word_list(word_list,df):
    papers_id_list = []
    for idx, paper in df.iterrows():
        if any(x in paper.text.lower() for x in word_list):
            papers_id_list.append(paper.paper_id)
    print("Total no of papers in ",len(df.index))
    print("Selected no of papers from",len(papers_id_list))
    return list(set(papers_id_list))

def filter_papers_topic_list(word_list,df,fldname):
    papers_id_list = []
    for idx, paper in df.iterrows():
        if any(word in paper[fldname].lower() for word in word_list):
            papers_id_list.append(paper.paper_id)
    print("Total no of papers in ",len(df.index))
    print("Selected no of papers from",len(papers_id_list))
    return list(set(papers_id_list))
## BOW , Dictionary, LDA Model
def extract_topic(text):
    
    stop_words = set(['author/funder', 'permission', 'No reuse allowed without', 'All rights reserved','bioRxiv',
       'medRxiv','license','preprint','doi:','copyright holder','perpetuity','. CC-BY-NC-ND 4.0 International',
        'https://doi.org','medRxiv preprint','It is made available under a author/funder'])

    x = nltk.sent_tokenize(text)

    clean_line = []

    for line in x:
        if not(should_remove_line(line, stop_words)):
            clean_line.append(line)

    ## split into paragraphs
    text = ''.join(clean_line).split('\\n\\n')
    
    tokenwords=tokenSentence(str(text))
    #Create a dictionary of words and its frequency
    counts = dict(Counter(tokenwords))
    vectorizer = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
    bow =  vectorizer.fit_transform(tokenwords).todense()
    
    number_topics = 1
    number_words = 10
    # Create and fit the LDA model
    lda = LDA(n_components=number_topics, n_jobs=-1)
    lda.fit(bow)
    return find_topics(lda, vectorizer, number_words)

# Find Topics by running the model

def find_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        topic_result  = " ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    return topic_result


# Function to add "Topics" as column to df

def add_topic(df):
    print("Shape before",df.shape)
    for index,row in df.iterrows():
        df.loc[index,'Text_Topics'] = extract_topic(df.loc[index,'text'])
        print(index)
    print("Shape after",df.shape)

from wordcloud import WordCloud, STOPWORDS

def create_wordcloud(topics_list):
    
    stopwords = set(STOPWORDS)
    
    #adding additional stopwords -- this is done after an initial look at the wordcloud
    additional_stop_words = ["medrxiv", "preprint", "fig",'however','although','abstract ','copyright',
                    'funder',' holder',' grant','grant','license','https','peer','reviewed','available',
         'peer-reviewed', 'author/funder', 'https//doiorg','doi','et','al']
    stopwords = additional_stop_words + list(STOPWORDS)
    
    # make all topics into one single string
    
    #topics_all = ''.join(list(map(str, topics_list)))
#     topics_all = tokenSentence(''.join(set(map(str, topics_list))))
    topics_all = tokenSentence(topics_list)
    
    # Create and generate a word cloud image:
#     wc = WordCloud(stopwords=STOPWORDS,
#             background_color='white',
#             max_words=200,
#             max_font_size=40, 
#             scale=5,
#             random_state=1,width=800, height=400
#         ).generate(str(topics_all))
    
    wc = WordCloud(width=1600, height=800).generate(str(topics_all))
    # generate word cloud
    #wc.generate(topics_all)

    #Plot the wordcloud image
    plt.figure( figsize=(20,10), facecolor='k')
    plt.imshow(wc)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    return 
#### Word frequencies
def word_frequency(text):
    stopwords = nltk.corpus.stopwords.words('english')
    word_frequencies = {}
    ## Tokenize the text, skip if number of tokens less than 5
    tokens = nltk.word_tokenize(text)
    if len(tokens) < 5: 
        return word_frequencies
    else :
        for word in tokens:
            if word not in stopwords:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1
        #print(word_frequencies)
        ## Calculate weighted frequency 
        maximum_frequncy = max(word_frequencies.values())

        for word in word_frequencies.keys():
            word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
        return word_frequencies

## Calculate sentence scores
def sentence_Score(sentence_list,word_frequencies):
    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]
        
    #return sentence_scores
    #calculate average sentence score to use as threshold
    sum_score_values = sum(sentence_scores.values())
    #print(len(sentence_scores.values()))
    if len(sentence_scores.values()) > 3:
        avg_score_values = sum_score_values/len(sentence_scores.values())
        summary = ''
        for sent in sentence_scores.keys():
            if sentence_scores[sent] >= (avg_score_values*0.50):
                t = sent
                summary = summary+sent

        print(summary)
    
def print_summary(txt1):
    for i in range(len(txt1)):  
        txt = replace_tags(txt1[i])
        txt = regex_exp(txt)  
        wordfreq=word_frequency(txt)
        sent = ''.join(txt)
        #sentence_list = nltk.sent_tokenize(sent)
        sentence_list = sent.split('.')
        #print(sentence_list)
        
        #if wordfreq is empty, summary = "" else summmarize the text
        if len(wordfreq) > 0:
            sentence_Score(sentence_list,wordfreq)

#extract the first word from each para
first_words = [p.split()[0] for p in clean_pmc['text'].tolist()]

#Convert it to lower case
first_words = [x.lower() for x in first_words]

plt.title("Occurence of First word in paragraph")
plot_word_freq(first_words)
biorxiv_clean.shape
#List of key words 
word_list = ['corona','covid','nCov','sars-cov-2','covid-19']
#list of paper id's containing the key words
selected_papers_1 = filter_papers_topic_list(word_list,biorxiv_clean,'text')
selected_papers_2 = filter_papers_topic_list(word_list,clean_pmc,'text')
#create a subset of original dataframe using the list of selected paper id's
subset_biorxiv = biorxiv_clean[biorxiv_clean['paper_id'].isin(selected_papers_1)]
subset_pmc = clean_pmc[clean_pmc['paper_id'].isin(selected_papers_2)]
#subset_df1.append(subset_df2)
#create a wordcloud
#create_wordcloud(subset_df1)

#find the topics in each paper
#add_topic(subset_biorxiv)
add_topic(subset_biorxiv)
#create wordcloud of all topics
create_wordcloud(subset_biorxiv['Text_Topics'].tolist())

plot_word_freq(str(subset_biorxiv['Text_Topics']).split(' '))
#subset_biorxiv['Text_Topics']
#stopwords.words("english")
word_list = ['vaccine','therapeutics','treatment','trial','tests','ADE','antibody']
list_topic = filter_papers_topic_list(word_list,subset_biorxiv,'text')

subset_df2 = subset_biorxiv[subset_biorxiv['paper_id'].isin(list_topic)]
#subset_df2
create_wordcloud(subset_df2['Text_Topics'].tolist())
plot_word_freq(str(subset_df2['Text_Topics']).split(' '))
for i in range(len(list_topic)):
    txt = subset_df2['text'][subset_df2['paper_id']==list_topic[i]].to_string()

    #Sentences containing these phrases should be removed

    stop_words = set(['author/funder', 'permission', 'No reuse allowed without', 'All rights reserved','bioRxiv',
           'medRxiv','license','preprint','doi:','copyright holder','perpetuity','. CC-BY-NC-ND 4.0 International',
            'https://doi.org','medRxiv preprint','It is made available under a author/funder'])

    x = nltk.sent_tokenize(txt)

    clean_line = []

    for line in x:
        if not(should_remove_line(line, stop_words)):
            clean_line.append(line)

    ## split into paragraphs
    text = ''.join(clean_line).split('\\n\\n')
    print("Paper ID : ", list_topic[i])
    print_summary(text)
    print("\n")
