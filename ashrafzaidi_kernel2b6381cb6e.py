# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import matplotlib.pyplot as plt

import collections 

from collections import Counter

from wordcloud import WordCloud, STOPWORDS



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dir_job_bulletins= '../input/cityofla/CityofLA/Job Bulletins'
file_job_titles ='../input/cityofla/CityofLA/Additional data/job_titles.csv'

job_titles = pd.read_csv(file_job_titles, header=None, names=['job_title'])
data_list = []

for filename in os.listdir(dir_job_bulletins):

    with open(os.path.join(dir_job_bulletins, filename), 'r', errors='ignore') as f:

        data_list.append([filename, ''.join(f.readlines())])

jobs = pd.DataFrame(data_list, columns=['file', 'job_description'])



# Drop row with id 263

jobs.drop([263], inplace=True)
def merge_jobs_data(jobs, extracted_data):

    jobs['temp'] = extracted_data

    for index, row in jobs.iterrows():

        extracted_data = row['temp']

        if isinstance(extracted_data, pd.DataFrame):

            for c in extracted_data.columns:

                jobs.loc[index, c] = extracted_data[c][0]

    jobs = jobs.drop('temp', axis=1) 

    return jobs
def extract_text_by_regex_index(text, regex_dictionary):

    regex_dictionary = pd.DataFrame(regex_dictionary, columns=['name', 'regexpr'])



    result = regex_dictionary.copy()

    result['text'] = ''

    for index,row in regex_dictionary.iterrows():

        find_text = re.search(row['regexpr'], text)

        find_text = find_text.span(0)[0] if find_text else -1

        result.loc[index, 'start'] = find_text

    result = result[result['start'] >= 0]

    result['end'] = result['start'].apply(lambda x: np.min(result[result['start'] > x]['start'])).fillna(len(text))



    for index,row in result.iterrows():

        extracted_text = text[int(row['start']):int(row['end'])]

        find_reg = re.findall(row['regexpr']+'(.*)', extracted_text, re.DOTALL|re.IGNORECASE)

        extracted_text = find_reg[0] if find_reg else ''

        extracted_text = extracted_text.strip()

        result.loc[index, 'text'] = extracted_text

    return result.set_index('name')[['text']].T
regex_dictionary = [('metadata', r''), 

                      ('salary', r'(?:ANNUAL SALARY|ANNUALSALARY)'),

                      ('duties', r'(?:DUTIES)'),

                      ('requirements', r'(?:REQUIREMENTS/MINIMUM QUALIFICATIONS|REQUIREMENT/MINIMUM QUALIFICATION|REQUIREMENT|REQUIREMENTS|REQUIREMENT/MIMINUMUM QUALIFICATION|REQUIREMENT/MIMINUMUM QUALIFICATIONS|REQUIREMENT$/MIMINUMUM QUALIFICATION$|REQUIREMENTS)'),

                      ('where_to_apply', r'(?:WHERE TO APPLY|HOW TO APPLY)'),

                      ('application_deadline', r'(?:APPLICATION DEADLINE|APPLICATION PROCESS)'),

                      ('selection_process', r'(?:SELECTION PROCESS|SELELCTION PROCESS)'),

                      ]

extracted_data = jobs['job_description'].dropna().apply(lambda x: extract_text_by_regex_index(x, regex_dictionary))

jobs = merge_jobs_data(jobs,extracted_data)
def extract_metadata(text):

    # ToDo: Extract additional information add the end of the text '(Exam open...)'

    job_title = text.split('\n')[0].strip()

    regex_class_code = r'(?:Class Code:|Class  Code:)\s*(\d\d\d\d)'

    class_code = re.findall(regex_class_code, text, re.DOTALL|re.IGNORECASE)

    class_code = class_code[0].strip() if class_code else np.NaN



    regex_open_date = r'(?:Open Date:|Open date:)\s*(\d\d-\d\d-\d\d)'

    open_date = re.findall(regex_open_date, text, re.DOTALL|re.IGNORECASE)

    open_date = open_date[0].strip() if open_date else np.NaN



    regex_revised = r'(?:Revised:|Revised|REVISED:)\s*(\d\d-\d\d-\d\d)'

    revised = re.findall(regex_revised, text, re.DOTALL|re.IGNORECASE)

    revised = revised[0].strip() if revised else np.NaN



    result = pd.DataFrame({'job_title':job_title,

                           'class_code':class_code,

                           'open_date':open_date,

                           'revised':revised}

                          , index=[0])

    result['open_date'] = pd.to_datetime(result['open_date'], infer_datetime_format=True)

    result['revised'] = pd.to_datetime(result['revised'], infer_datetime_format=True)

    return result
jobs = merge_jobs_data(jobs, jobs['metadata'].dropna().apply(extract_metadata))
def extract_salary(text):

    regex_salary_from = r'\$((?:\d{1,3})(?:\,\d{3})*(?:\.\d{2})*).*'

    salary_from = re.findall(regex_salary_from, text, re.DOTALL|re.IGNORECASE)

    salary_from = float(salary_from[0].replace(',', '')) if salary_from else np.NaN

    

    regex_salary_to = r'(?:and|to) \$((?:\d{1,3})(?:\,\d{3})*(?:\.\d{2})*).*'

    salary_to = re.findall(regex_salary_to, text, re.DOTALL|re.IGNORECASE)

    salary_to = float(salary_to[0].replace(',', '')) if salary_to else np.NaN    

    

    regex_salary_flatrated = r'(flat-rated|Flat-Rated)'

    salary_flatrated = re.findall(regex_salary_flatrated, text, re.DOTALL|re.IGNORECASE)

    salary_flatrated = True if salary_flatrated else np.NaN    

    

    regex_salary_additional = r'(?:\n)(.*)(?:NOTES)'

    salary_additional = re.findall(regex_salary_additional, text, re.DOTALL|re.IGNORECASE)

    salary_additional = salary_additional[0].strip() if salary_additional else np.NaN   

    

    regex_salary_notes = r'(?:NOTES:)(.*)'

    salary_notes = re.findall(regex_salary_notes, text, re.DOTALL|re.IGNORECASE)

    salary_notes = salary_notes[0].strip() if salary_notes else np.NaN    



    result = pd.DataFrame({'salary_from':salary_from,

                           'salary_to':salary_to,

                           'salary_flatrated':salary_flatrated,

                           'salary_additional':salary_additional,

                           'salary_notes':salary_notes}

                          , index=[0])

    return result
jobs = merge_jobs_data(jobs, jobs['salary'].dropna().apply(extract_salary))
regex_dictionary = [('duties_text', r''), 

                      ('duties_notes', r'(?:NOTE:|NOTES:)'),

                      ]

extracted_data = jobs['duties'].dropna().apply(lambda x: extract_text_by_regex_index(x, regex_dictionary))

jobs = merge_jobs_data(jobs, extracted_data)
regex_dictionary = [('where_to_apply_text', r''), 

                         ('where_to_apply_notes', r'(?:NOTE:)'),

                      ]

extracted_data = jobs['where_to_apply'].dropna().apply(lambda x: extract_text_by_regex_index(x, regex_dictionary))

jobs = merge_jobs_data(jobs, extracted_data)
regex_dictionary = [('application_deadline_text', r''), 

                         ('application_deadline_notes', r'(?:NOTE:)'),

                         ('application_deadline_review', r'(?:QUALIFICATIONS REVIEW|EXPERT REVIEW COMMITTEE)'),

                      ]

extracted_data = jobs['application_deadline'].dropna().apply(lambda x: extract_text_by_regex_index(x, regex_dictionary))

jobs = merge_jobs_data(jobs, extracted_data)
regex_dictionary = [('selection_process_text', r''), 

                         ('selection_process_notes', r'(?:NOTES:)'),

                         ('selection_process_notice', r'(?:NOTICE:|Notice:)'),

                      ]

extracted_data = jobs['selection_process'].dropna().apply(lambda x: extract_text_by_regex_index(x, regex_dictionary))

jobs = merge_jobs_data(jobs, extracted_data)
cols = ['job_description', 'metadata', 'salary', 'duties', 'where_to_apply', 'application_deadline', 'selection_process']

jobs = jobs.drop(cols, axis=1)
jobs.to_csv('abc.csv',index=False)
#creating a wordcloud

data=pd.read_csv('abc.csv')

data.head()
#data.shape

#data.dtypes

#data.isnull().sum()



data = data.dropna(subset=['selection_process_notes'])

from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='ivory',

        stopwords=stopwords,

        max_words=200,

        max_font_size=40, 

        scale=3,

        random_state=1 # chosen at random by flipping a coin; it was heads

).generate(str(data))



    fig = plt.figure(1, figsize=(15, 15))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()



show_wordcloud(data['selection_process_notes'])



def cleaning(s):

    s = str(s)

    s = s.lower()

    s = re.sub('\s\W',' ',s)

    s = re.sub('\W,\s',' ',s)

    s = re.sub(r'[^\w]', ' ', s)

    s = re.sub("\d+", "", s)

    s = re.sub('\s+',' ',s)

    s = re.sub('[!@#$_]', '', s)

    s = s.replace("co","")

    s = s.replace("https","")

    s = s.replace(",","")

    s = s.replace("[\w*"," ")

    s = s.replace("dtype","")

    return s

data['selection_process_notes'] = [cleaning(s) for s in data['selection_process_notes']]



# lets visualize the top word selection_process_notes in the form of a bar chart:

text = data.selection_process_notes[0]
import collections

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 

  

stopwords = set(stopwords.words('english')) 

stopwords.update(["the", "The", ".", ",","may"])

wordtokens = word_tokenize(text) 

filtered_words = [word for word in wordtokens if word not in stopwords]

counted_words = collections.Counter(filtered_words)



words = []

counts = []

for letter, count in counted_words.most_common(10):

    words.append(letter)

    counts.append(count)
import matplotlib.cm as cm

import matplotlib.pyplot as plt

from matplotlib import rcParams



colors = cm.rainbow(np.linspace(0, 1, 10))

rcParams['figure.figsize'] = 20, 10



plt.title('Top words in the Selection Process Notes vs their count')

plt.xlabel('Count')

plt.ylabel('Words')

plt.barh(words, counts, color=colors)



# we see similar words as in the word cloud
# Lets start with the sentiment analysis. 

# we use the NLTK library to arrive at polarity scores and then use Afinn as well. conclude it with visualizations of the sentiments
#using selection process notes from the abc.csv file that we created earlier. As this would be enabale us with insights 

#as what needs to be done in order to attract diversity

final_X =data['selection_process_notes']
import nltk

from nltk.corpus import stopwords 

stop = set(stopwords.words('english')) 
#we use lemmatization in order to retain the meaning of the words  

from nltk.stem import WordNetLemmatizer 

from nltk.corpus import wordnet 

final_X = data['selection_process_notes']

import re

temp =[]

snow = WordNetLemmatizer()   # downlaod nltk wordnet before runnng this line of code

for sentence in final_X:

    sentence = sentence.lower()                 # Converting to lowercase

    cleanr = re.compile('<.*?>')

    sentence = re.sub(cleanr, ' ', sentence)        #Removing HTML tags

    sentence = re.sub(r'[?|!|\'|"|#]',r'',sentence)

    sentence = re.sub(r'[.|,|)|(|\|/]',r' ',sentence)        #Removing Punctuations

    sentence = re.sub(r'[%]',r' ',sentence)

    sentence = re.sub(r'\d+',r' ',sentence)        #Removing Punctuations

    

    words = [snow.lemmatize(word) for word in sentence.split() if word not in stopwords.words('english')]   # Lemmatizing and removing stopwords

    temp.append(words)

    final_X = temp 
listwords=[]

for sentence in final_X:

    for key in sentence:

        listwords.append(key) 

# Creating a dataframe object from listoftuples

dfObj = pd.DataFrame(listwords,columns=['Word'])
#printing polarity scores of words in selection process notes

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk import sentiment

sid = SentimentIntensityAnalyzer()



for sentence in final_X:

    for key in sentence:

        scores = sid.polarity_scores(key)

        

        print('{1} and {0}'.format(key, scores))

        

        

!pip install afinn
# initialize afinn sentiment analyzer

from afinn import Afinn

af = Afinn()


sentiment_scores=[af.score(final_X) for final_X in sentence]



sentiment_category = ['positive' if score > 0 

                          else 'negative' if score < 0 

                              else 'neutral' 

                                  for score in sentiment_scores]







df = pd.DataFrame([list(dfObj['Word']), sentiment_scores, sentiment_category]).T

df.columns=['Word', 'sentiment_scores','sentiment_category']

df.head()
# viuslaize the sentiment scores against the sentiment cateogry to 

import seaborn as sns

f, (ax1)= plt.subplots(1, 1, figsize=(10, 4))

sp = sns.stripplot(x='sentiment_category', y="sentiment_scores", hue='sentiment_category',

                  data= df, ax=ax1)





t = f.suptitle('Visualizing  Sentiment', fontsize=14)
 

import seaborn as sns

f, (ax1,ax2) = plt.subplots(1, 2, figsize=(10,4))

sp = sns.countplot(x='sentiment_category', hue='sentiment_category',

                  data= df, ax=ax1)

sp1 = sns.countplot(x='sentiment_scores', hue='sentiment_scores',

                  data= df, ax=ax2)



t = f.suptitle('Visualizing  Sentiment', fontsize=14)
df=data[['job_title','selection_process_notes']]

df.head()
from pprint import pprint

# Convert to list

data = df.selection_process_notes.values.tolist()



# Remove new line characters

data = [re.sub('\s+', ' ', sent) for sent in data]



# Remove distracting single quotes

data = [re.sub(r'[.|,|)|(|\|/]', "", sent) for sent in data]



data = [re.sub(r'[?|!|\'|"|#]', "", sent) for sent in data]



data = [re.sub(r'\d+', "", sent) for sent in data]



data = [re.sub(r'[%]', "", sent) for sent in data]

# Remove Emails

data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]



pprint(data[:1])
# Gensim

import gensim

import gensim.corpora as corpora

from gensim.utils import simple_preprocess

from gensim.models import CoherenceModel



def sent_to_words(sentences):

    for sentence in sentences:

        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations



data_words = list(sent_to_words(data))



print(data_words[:1])
# Build the bigram and trigram models

bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.

trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  



# Faster way to get a sentence clubbed as a trigram/bigram

bigram_mod = gensim.models.phrases.Phraser(bigram)

trigram_mod = gensim.models.phrases.Phraser(trigram)



# See trigram example

print(trigram_mod[bigram_mod[data_words[0]]])
# spacy for lemmatization

import spacy

# Define functions for stopwords, bigrams, trigrams and lemmatization

def remove_stopwords(texts):

    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]



def make_bigrams(texts):

    return [bigram_mod[doc] for doc in texts]



def make_trigrams(texts):

    return [trigram_mod[bigram_mod[doc]] for doc in texts]



def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    """https://spacy.io/api/annotation"""

    texts_out = []

    for sent in texts:

        doc = nlp(" ".join(sent)) 

        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    return texts_out
stop_words = stopwords.words('english')

stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
import en_core_web_sm

spacy.load('en')
# Remove Stop Words

data_words_nostops = remove_stopwords(data_words)



# Form Bigrams

data_words_bigrams = make_bigrams(data_words_nostops)



# Initialize spacy 'en' model, keeping only tagger component (for efficiency)



#python -m spacy download en works only if you have administration rights, the installation folder should have admin rights



nlp = spacy.load('en', disable=['parser', 'ner'])





# Do lemmatization keeping only noun, adj, vb, adv

data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])



print(data_lemmatized[:1])
# Create Dictionary

id2word = corpora.Dictionary(data_lemmatized)



# Create Corpus

texts = data_lemmatized



# Term Document Frequency

corpus = [id2word.doc2bow(text) for text in texts]



# View

print(corpus[:1])

# Human readable format of corpus (term-frequency)

[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
id2word[0]
# Gensim

import gensim

import gensim.corpora as corpora

from gensim.utils import simple_preprocess

from gensim.models import CoherenceModel

# Enable logging for gensim - optional

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)



import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)
# Create Dictionary

id2word = corpora.Dictionary(data_lemmatized)



# Create Corpus: Term Document Frequency

corpus = [id2word.doc2bow(text) for text in data_lemmatized]



# Build LDA model

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,

                                           id2word=id2word,

                                           num_topics=10, 

                                           random_state=100,

                                           update_every=1,

                                           chunksize=10,

                                           passes=10,

                                           alpha='symmetric',

                                           iterations=100,

                                           per_word_topics=True)

# Compute Perplexity

print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score

coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')

coherence_lda = coherence_model_lda.get_coherence()

print('\nCoherence Score: ', coherence_lda)


def format_topics_sentences(ldamodel=None, corpus=corpus, texts=data):

    # Init output

    sent_topics_df = pd.DataFrame()



    # Get main topic in each document

    for i, row_list in enumerate(ldamodel[corpus]):

        row = row_list[0] if ldamodel.per_word_topics else row_list            

        # print(row)

        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        # Get the Dominant topic, Perc Contribution and Keywords for each document

        for j, (topic_num, prop_topic) in enumerate(row):

            if j == 0:  # => dominant topic

                wp = ldamodel.show_topic(topic_num)

                topic_keywords = ", ".join([word for word, prop in wp])

                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)

            else:

                break

    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']



    # Add original text to the end of the output

    contents = pd.Series(texts)

    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    return(sent_topics_df)





df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_lemmatized)



# Format

df_dominant_topic = df_topic_sents_keywords.reset_index()

df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

df_dominant_topic.head(10)
# Display setting to show more characters in column

pd.options.display.max_colwidth = 100



sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')



for i, grp in sent_topics_outdf_grpd:

    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 

                                             grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], 

                                            axis=0)



# Reset Index    

sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)



# Format

sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]



# Show

sent_topics_sorteddf_mallet.head(10)
doc_lens = [len(d) for d in df_dominant_topic.Text]



# Plot

plt.figure(figsize=(4,2), dpi=160)

plt.hist(doc_lens, bins = 1000, color='navy')

plt.text(750, 100, "Mean   : " + str(round(np.mean(doc_lens))))

plt.text(750,  95, "Median : " + str(round(np.median(doc_lens))))

plt.text(750,  90, "Stdev   : " + str(round(np.std(doc_lens))))

plt.text(750,  85, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))

plt.text(750,  80, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))

plt.gca().set(xlim=(0, 1000), ylabel='Number of Documents', xlabel='Document Word Count')

plt.tick_params(size=16)

plt.xticks(np.linspace(0,1000,9))

plt.title('Distribution of Document Word Counts', fontdict=dict(size=10))

plt.show()


import seaborn as sns

import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'



fig, axes = plt.subplots(2,2,figsize=(16,14), dpi=160, sharex=True, sharey=True)



for i, ax in enumerate(axes.flatten()):    

    df_dominant_topic_sub = df_dominant_topic.loc[df_dominant_topic.Dominant_Topic == i, :]

    doc_lens = [len(d) for d in df_dominant_topic_sub.Text]

    ax.hist(doc_lens, bins = 1000, color=cols[i])

    ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])

    sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())

    ax.set(xlim=(0, 1000), xlabel='Document Word Count')

    ax.set_ylabel('Number of Documents', color=cols[i])

    ax.set_title('Topic: '+str(i), fontdict=dict(size=16, color=cols[i]))



fig.tight_layout()

fig.subplots_adjust(top=0.90)

plt.xticks(np.linspace(0,1000,9))

fig.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=22)

plt.show()
# 1. Wordcloud of Top N words in each topic

from matplotlib import pyplot as plt

from wordcloud import WordCloud, STOPWORDS

import matplotlib.colors as mcolors



cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'



cloud = WordCloud(stopwords=stop_words,

                  background_color='white',

                  width=2500,

                  height=1800,

                  max_words=10,

                  colormap='tab10',

                  color_func=lambda *args, **kwargs: cols[i],

                  prefer_horizontal=1.0)



topics = lda_model.show_topics(formatted=False)



fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)



for i, ax in enumerate(axes.flatten()):

    fig.add_subplot(ax)

    topic_words = dict(topics[i][1])

    cloud.generate_from_frequencies(topic_words, max_font_size=300)

    plt.gca().imshow(cloud)

    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))

    plt.gca().axis('off')





plt.subplots_adjust(wspace=0, hspace=0)

plt.axis('off')

plt.margins(x=0, y=0)

plt.tight_layout()

plt.show()
from collections import Counter

topics = lda_model.show_topics(formatted=False)

data_flat = [w for w_list in data_lemmatized for w in w_list]

counter = Counter(data_flat)



out = []

for i, topic in topics:

    for word, weight in topic:

        out.append([word, i , weight, counter[word]])



df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        



# Plot Word Count and Weights of Topic Keywords

fig, axes = plt.subplots(2, 2, figsize=(16,10), sharey=True, dpi=160)

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

for i, ax in enumerate(axes.flatten()):

    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')

    ax_twin = ax.twinx()

    ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')

    ax.set_ylabel('Word Count', color=cols[i])

    ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)

    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)

    ax.tick_params(axis='y', left=False)

    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')

    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')



fig.tight_layout(w_pad=2)    

fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    

plt.show()
# Sentence Coloring of N Sentences

from matplotlib.patches import Rectangle



def sentences_chart(lda_model=lda_model, corpus=corpus, start = 0, end = 13):

    corp = corpus[start:end]

    mycolors = [color for name, color in mcolors.TABLEAU_COLORS.items()]



    fig, axes = plt.subplots(end-start, 1, figsize=(20, (end-start)*0.95), dpi=160)       

    axes[0].axis('off')

    for i, ax in enumerate(axes):

        if i > 0:

            corp_cur = corp[i-1] 

            topic_percs, wordid_topics, wordid_phivalues = lda_model[corp_cur]

            word_dominanttopic = [(lda_model.id2word[wd], topic[0]) for wd, topic in wordid_topics]    

            ax.text(0.01, 0.5, "Doc " + str(i-1) + ": ", verticalalignment='center',

                    fontsize=16, color='black', transform=ax.transAxes, fontweight=700)



            # Draw Rectange

            topic_percs_sorted = sorted(topic_percs, key=lambda x: (x[1]), reverse=True)

            ax.add_patch(Rectangle((0.0, 0.05), 0.99, 0.90, fill=None, alpha=1, 

                                   color=mycolors[topic_percs_sorted[0][0]], linewidth=2))



            word_pos = 0.06

            for j, (word, topics) in enumerate(word_dominanttopic):

                if j < 14:

                    ax.text(word_pos, 0.5, word,

                            horizontalalignment='left',

                            verticalalignment='center',

                            fontsize=16, color=mycolors[topics],

                            transform=ax.transAxes, fontweight=700)

                    word_pos += .009 * len(word)  # to move the word for the next iter

                    ax.axis('off')

            ax.text(word_pos, 0.5, '. . .',

                    horizontalalignment='left',

                    verticalalignment='center',

                    fontsize=16, color='black',

                    transform=ax.transAxes)       



    plt.subplots_adjust(wspace=0, hspace=0)

    plt.suptitle('Sentence Topic Coloring for Documents: ' + str(start) + ' to ' + str(end-2), fontsize=22, y=0.95, fontweight=700)

    plt.tight_layout()

    plt.show()



sentences_chart()    
# Sentence Coloring of N Sentences

def topics_per_document(model, corpus, start=0, end=1):

    corpus_sel = corpus[start:end]

    dominant_topics = []

    topic_percentages = []

    for i, corp in enumerate(corpus_sel):

        topic_percs, wordid_topics, wordid_phivalues = model[corp]

        dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]

        dominant_topics.append((i, dominant_topic))

        topic_percentages.append(topic_percs)

    return(dominant_topics, topic_percentages)



dominant_topics, topic_percentages = topics_per_document(model=lda_model, corpus=corpus, end=-1)            



# Distribution of Dominant Topics in Each Document

df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])

dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()

df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()



# Total Topic Distribution by actual weight

topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])

df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()



# Top 3 Keywords for each Topic

topic_top3words = [(i, topic) for i, topics in lda_model.show_topics(formatted=False) 

                                 for j, (topic, wt) in enumerate(topics) if j < 3]



df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])

df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)

df_top3words.reset_index(level=0,inplace=True)

from matplotlib.ticker import FuncFormatter



# Plot

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=120, sharey=True)



# Topic Distribution by Dominant Topics

ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc, width=.5, color='firebrick')

ax1.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))

tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x)+ '\n' + df_top3words.loc[df_top3words.topic_id==x, 'words'].values[0])

ax1.xaxis.set_major_formatter(tick_formatter)

ax1.set_title('Number of Documents by Dominant Topic', fontdict=dict(size=10))

ax1.set_ylabel('Number of Documents')

ax1.set_ylim(0, 1000)



# Topic Distribution by Topic Weights

ax2.bar(x='index', height='count', data=df_topic_weightage_by_doc, width=.5, color='steelblue')

ax2.set_xticks(range(df_topic_weightage_by_doc.index.unique().__len__()))

ax2.xaxis.set_major_formatter(tick_formatter)

ax2.set_title('Number of Documents by Topic Weightage', fontdict=dict(size=10))



plt.show()
import bokeh
# Get topic weights and dominant topics ------------



from sklearn.manifold import TSNE

from bokeh.plotting import figure, output_file, show

from bokeh.models import Label

from bokeh.io import output_notebook



# Get topic weights

topic_weights = []

for i, row_list in enumerate(lda_model[corpus]):

    topic_weights.append([w for i, w in row_list[0]])



# Array of topic weights    

arr = pd.DataFrame(topic_weights).fillna(0).values



# Keep the well separated points (optional)

arr = arr[np.amax(arr, axis=1) > 0.35]



# Dominant topic number in each doc

topic_num = np.argmax(arr, axis=1)



# tSNE Dimension Reduction

tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')

tsne_lda = tsne_model.fit_transform(arr)



# Plot the Topic Clusters using Bokeh

output_notebook()

n_topics = 4

mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])

plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), 

              plot_width=900, plot_height=700)

plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])

show(plot)
# Visualize the topics

# Plotting tools

import pyLDAvis

import pyLDAvis.gensim  # don't skip this

import matplotlib.pyplot as plt

# Visualize the topics

# Visualize the topics



pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)

vis