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

#nltk.download('wordnet')

#nltk.download('punkt')     

#nltk.download('stopwords')    

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer,WordNetLemmatizer 

from collections import Counter

import matplotlib.pyplot as plt



import string

from nltk.tokenize import sent_tokenize, word_tokenize



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation as LDA
biorxiv_clean = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/biorxiv_clean.csv")

clean_comm_use = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_comm_use.csv")

clean_noncomm_use = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_noncomm_use.csv")

clean_pmc = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_pmc.csv")



all_data = pd.concat([biorxiv_clean, clean_comm_use, clean_noncomm_use, clean_pmc]).reset_index(drop=True)



all_data.head(1)

def clean_text(text):

    text_clean = text.lower()

    text_clean = text_clean.replace('\\n', '')                #remove \\n

    text_clean = re.sub('[!#?,%*&$)@^(.:";]', '', text_clean)       #remove punctuation

#     text_clean = re.sub(r’\d+’, ‘’, text_clean)             #remove numbers

    text_clean = text_clean.replace('introduction', '')       #remove 'introduction'

    text_clean = text_clean.replace('background', '')         #remove 'background'

    text_clean = text_clean.replace('abstract', '')           #remove 'abstract'

    text_clean = text_clean.strip()                           #remove whitespace

    return text_clean
# Function for tokenization, lemmatization and stemming

def tokenSentence(sentence):



    clean_sentence = clean_text(sentence)

    token_words=word_tokenize(clean_sentence) #Tokenize 

    stop_words = set(stopwords.words('english'))

    #adding additional stopwords -- this is done after an initial look at the wordcloud

    stop_words = ["medrxiv", "preprint", "fig"] + list(stop_words)

    remove_stop_words = [word for word in token_words if word not in stop_words] #Remove stop words

    token_words = []

    for word in remove_stop_words:

        token_words.append(WordNetLemmatizer().lemmatize(word, pos='v')) #lemmatization

    return token_words



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
## Function to run the LDA model and return topics

def find_topics(model, count_vectorizer, n_top_words):

    words = count_vectorizer.get_feature_names()

    for topic_idx, topic in enumerate(model.components_):

        topic_result  = " ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]])

    return topic_result





## Function to fit the LDA model

def extract_topic(text):



    tokenwords=tokenSentence(text)

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



# Function to add "Topics" as column to df



def add_topic(df):

    print("Shape before",df.shape)

    for index,row in df.iterrows():

        if index > 940:

            df.loc[index,'text_topic'] = extract_topic(df.loc[index,'text'])

            print(index)

    print("Shape after",df.shape) ## this is to track the progress of the function



add_topic(biorxiv_clean)

add_topic(clean_comm_use)

add_topic(clean_noncomm_use)

add_topic(clean_pmc)
#export to csv

biorxiv_clean.to_csv ("biorxiv_topic.csv", index = False, header=True)
## Function to filter papers based on keyword list





      

def filter_papers_word_list(word_list,df):

    papers_id_list = []

    for idx, paper in df.iterrows():

        if all(x in paper.text.lower() for x in word_list):

            papers_id_list.append(paper.paper_id)

    print("Total no of papers in ",len(df.index))

    print("Selected no of papers from",len(papers_id_list))

    return list(set(papers_id_list))





word_list = ['covid','corona','cov']

selected_biorxiv = filter_papers_word_list(word_list,biorxiv_clean)

selected_clean_comm_use = filter_papers_word_list(word_list,clean_comm_use)

selected_clean_noncomm_use = filter_papers_word_list(word_list,clean_noncomm_use)

selected_clean_pmc = filter_papers_word_list(word_list,clean_pmc)
##create a wordcloud to understand the topics

from wordcloud import WordCloud, STOPWORDS

## select papers without "corona" and "covid" in the text, create a wordcloud##

        

subset_df = clean_comm_use[~clean_comm_use['paper_id'].isin(selected_clean_comm_use)]



# make all topics into one single string

topics_list = subset_df['text'].tolist()

topics_all = ''.join(list(map(str, topics_list)))



# Create and generate a word cloud image:

wc = WordCloud(

        background_color='white',

        max_words=200,

        max_font_size=40, 

        scale=5,

        random_state=1

    ).generate(str(topics_all))



# generate word cloud

wc.generate(topics_all)



#Plot the wordcloud image

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")

fig = plt.figure(1, figsize=(15,15))



plt.show()
# function to plot the frequency of words in a list

def plot_word_freq(list_words):

    remove_stop_words = [word for word in list_words if word not in stopwords.words("english")] #Remove stop words

    remove_numbers = [''.join(x for x in i if x.isalpha()) for i in remove_stop_words] # Remove numbers

    remove_empty_string = ' '.join(remove_numbers).split() # Remove empty strings

    counts = dict(Counter(remove_empty_string)) #Count the occurence of each string

    popular_words = sorted(counts, key = counts.get, reverse = True) #sort by number of occurence

    plt.barh(range(20), [counts[w] for w in reversed(popular_words[0:20])]) #plot horizontal bar graph

    plt.yticks([x  for x in range(20)], reversed(popular_words[0:20]))

    plt.show()

    

#plot_word_freq(topics_all)