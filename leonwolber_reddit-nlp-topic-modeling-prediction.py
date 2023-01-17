import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

import re

import string



import nltk

from nltk.probability import FreqDist

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer

from nltk import pos_tag

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize



from wordcloud import WordCloud

from tqdm.auto import tqdm

import matplotlib.style as style

style.use('fivethirtyeight')

from sklearn.metrics import plot_roc_curve

from numpy import interp

from itertools import cycle

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
# load data



data = pd.read_csv("../input/dataisbeautiful/r_dataisbeautiful_posts.csv")
data.head()
data.info()
(data.isnull().sum() / len(data)) * 100
del data['id']

del data['author_flair_text']

del data['removed_by']

del data['total_awards_received']

del data['awarders']

del data['created_utc']

del data['full_link']
data = data.dropna()
data.head()
data.info()
# check out numeric columns



data.describe().T
len(data)
plt.figure(figsize=(13,5))



sns.kdeplot(data['score'], shade=  True)
print(len(data[data['score'] < 10]), 'Posts with less than 10 votes')

print(len(data[data['score'] > 10]), 'Posts with more than 10 votes')
plt.figure(figsize=(13,5))



sns.kdeplot(data['num_comments'], shade=  True)
print(len(data[data['num_comments'] < 10]), 'Posts with less than 10 comments')

print(len(data[data['num_comments'] > 10]), 'Posts with more than 10 comments')
# post with the most comments



data[data['score'] == data['score'].max()]['title'].iloc[0]




def remove_line_breaks(text):

    text = text.replace('\r', ' ').replace('\n', ' ')

    return text



#remove punctuation

def remove_punctuation(text):

    re_replacements = re.compile("__[A-Z]+__")  # such as __NAME__, __LINK__

    re_punctuation = re.compile("[%s]" % re.escape(string.punctuation))

    '''Escape all the characters in pattern except ASCII letters and numbers'''

    tokens = word_tokenize(text)

    tokens_zero_punctuation = []

    for token in tokens:

        if not re_replacements.match(token):

            token = re_punctuation.sub(" ", token)

        tokens_zero_punctuation.append(token)

    return ' '.join(tokens_zero_punctuation)



def remove_special_characters(text):

    text = re.sub('[^a-zA-z0-9\s]', '', text)

    return text



def lowercase(text):

    text_low = [token.lower() for token in word_tokenize(text)]

    return ' '.join(text_low)



def remove_stopwords(text):

    stop = set(stopwords.words('english'))

    word_tokens = nltk.word_tokenize(text)

    text = " ".join([word for word in word_tokens if word not in stop])

    return text



#remobe one character words

def remove_one_character_words(text):

    '''Remove words from dataset that contain only 1 character'''

    text_high_use = [token for token in word_tokenize(text) if len(token)>1]      

    return ' '.join(text_high_use)   

    

#%%

# Stemming with 'Snowball stemmer" package

def stem(text):

    stemmer = nltk.stem.snowball.SnowballStemmer('english')

    text_stemmed = [stemmer.stem(token) for token in word_tokenize(text)]        

    return ' '.join(text_stemmed)



def lemma(text):

    wordnet_lemmatizer = WordNetLemmatizer()

    word_tokens = nltk.word_tokenize(text)

    text_lemma = " ".join([wordnet_lemmatizer.lemmatize(word) for word in word_tokens])       

    return ' '.join(text_lemma)





#break sentences to individual word list

def sentence_word(text):

    word_tokens = nltk.word_tokenize(text)

    return word_tokens

#break paragraphs to sentence token 

def paragraph_sentence(text):

    sent_token = nltk.sent_tokenize(text)

    return sent_token    





def tokenize(text):

    """Return a list of words in a text."""

    return re.findall(r'\w+', text)



def remove_numbers(text):

    no_nums = re.sub(r'\d+', '', text)

    return ''.join(no_nums)







def clean_text(text):

    _steps = [

    remove_line_breaks,

    remove_one_character_words,

    remove_special_characters,

    lowercase,

    remove_punctuation,

    remove_stopwords,

    stem,

    remove_numbers

]

    for step in _steps:

        text=step(text)

    return text   

#%%



data['clean_title'] = pd.Series([clean_text(i) for i in tqdm(data['title'])])
words = data["clean_title"].values
ls = []



for i in words:

    ls.append(str(i))
ls[:5]
# The wordcloud of Cthulhu/squidy thing for HP Lovecraft

plt.figure(figsize=(16,13))

wc = WordCloud(background_color="black", max_words=1000, max_font_size= 200,  width=1600, height=800)

wc.generate(" ".join(ls))

plt.title("Most discussed terms", fontsize=20)

plt.imshow(wc.recolor( colormap= 'viridis' , random_state=17), alpha=0.98, interpolation="bilinear", )

plt.axis('off')
most_pop = data.sort_values('score', ascending =False)[['title', 'score']].head(12)



most_pop['score1'] = most_pop['score']/1000
plt.figure(figsize = (20,25))



sns.barplot(data = most_pop, y = 'title', x = 'score1', color = 'c')

plt.xticks(fontsize=27, rotation=0)

plt.yticks(fontsize=31, rotation=0)

plt.xlabel('Votes in Thousands', fontsize = 21)

plt.ylabel('')

plt.title('Most popular posts', fontsize = 30)
data.head()
most_com = data.sort_values('num_comments', ascending =False)[['title', 'num_comments', 'author']].head(12)

most_com['num_comments1'] = most_com['num_comments']/1000
type(most_com)
x = data.reset_index()

x[x['index'] == 92800]
most_com = most_com[most_com.author != 'dinoignacio']
plt.figure(figsize = (20,25))



sns.barplot(data = most_com, y = 'title', x = 'num_comments1', color = 'y')

plt.xticks(fontsize=28, rotation=0)

plt.yticks(fontsize=30, rotation=0)

plt.xlabel('Comments in Thousands', fontsize = 21)

plt.ylabel('')

plt.title('Most commented posts', fontsize = 30)
most_com.head(10)
n = data.sort_values('score', ascending =False)



n['score1'] = n['score']/1000

n['num_comments1'] = n['num_comments']/1000
plt.figure(figsize = (15,15))



sns.regplot(data = n, y = 'score1', x = 'num_comments1', color = 'purple')

plt.xticks(fontsize=14, rotation=0)

plt.yticks(fontsize=14, rotation=0)

plt.xlabel('Comments in Thousands', fontsize = 15)

plt.ylabel('Votes in Thousands')

plt.title('Comments and votes', fontsize = 14)
data[data['num_comments'] == data['num_comments'].max()]
import gensim

from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import STOPWORDS

from nltk.stem import WordNetLemmatizer, SnowballStemmer

from nltk.stem.porter import *

import numpy as np

np.random.seed(2018)

import nltk

stemmer = SnowballStemmer('english')
nltk.download('wordnet')
def lemmatize_stemming(text):

    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))



def preprocess(text):

    result = []

    for token in gensim.utils.simple_preprocess(text):

        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:

            result.append(lemmatize_stemming(token))

    return result
data['title'].iloc[1]
doc_sample = data['title'].iloc[1]

print('original document: ')



words = []



for word in doc_sample.split(' '):

    words.append(word)

    

    

print(words)

print('\n\n tokenized and lemmatized document: ')

print(preprocess(doc_sample))
data.info()
data['clean_title'] = data['clean_title'].astype(str)
words = []



for i in data['clean_title']:

        words.append(i.split(' '))
dictionary = gensim.corpora.Dictionary(words)



count = 0

for k, v in dictionary.iteritems():

    print(k, v)

    count += 1

    if count > 10:

        break
# Filter out tokens in the dictionary by their frequency.



dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in words]

bow_corpus[4310]
bow_doc_4310 = bow_corpus[4310]



for i in range(len(bow_doc_4310)):

    print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0], 

                                               dictionary[bow_doc_4310[i][0]], 

bow_doc_4310[i][1]))
from gensim import corpora, models



tfidf = models.TfidfModel(bow_corpus)

corpus_tfidf = tfidf[bow_corpus]



from pprint import pprint



for doc in corpus_tfidf:

    pprint(doc)

    break
lda_model = gensim.models.LdaMulticore(bow_corpus,

                                       num_topics=10,

                                       id2word=dictionary,

                                       passes=2,

                                       workers=2)
for idx, topic in lda_model.print_topics(-1):

    print('Topic: {} \nWords: {}'.format(idx, topic))
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf,

                                             num_topics=10,

                                             id2word=dictionary,

                                             passes=2,

                                             workers=4)



for idx, topic in lda_model_tfidf.print_topics(-1):

    print('Topic: {} Word: {}'.format(idx, topic))
unseen_document = 'How a Pentagon deal became an identity crisis for Google'

bow_vector = dictionary.doc2bow(preprocess(unseen_document))



for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):

    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
data['over_18'] = data['over_18'].astype(int)
data['over_18'] = pd.Categorical(data['over_18']) 
(data['over_18'].value_counts(normalize=True))
from sklearn.utils import resample


# Separate majority and minority classes

df_majority = data[data.over_18==0]

df_minority = data[data.over_18==1]

 

# Upsample minority class

df_minority_upsampled = resample(df_minority, 

                                 replace=True,     # sample with replacement

                                 n_samples=180000) # reproducible results

 

# Combine majority class with upsampled minority class

data_n = pd.concat([df_majority, df_minority_upsampled])

 

# Display new class counts

data_n['over_18'].value_counts()
(data_n['over_18'].value_counts(normalize=True))
from sklearn.model_selection import train_test_split

from sklearn import model_selection, naive_bayes, svm

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
processed_text = data_n['clean_title']
vectorizer = TfidfVectorizer()

tfidf = vectorizer.fit_transform(processed_text)

print(tfidf.shape)

print('\n')

#print(vectorizer.get_feature_names())
data_n['over_18']
y = data_n['over_18']
X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(tfidf, y, test_size=0.2, random_state=42)
# fit the training dataset on the NB classifier

Naive = naive_bayes.MultinomialNB()

Naive.fit(X_train_tf,y_train_tf)

# predict the labels on validation dataset

predictions_NB_tf = Naive.predict(X_test_tf)

# Use accuracy_score function to get the accuracy

print("Naive Bayes Accuracy -> ",accuracy_score(predictions_NB_tf, y_test_tf)*100)

print(classification_report(predictions_NB_tf,y_test_tf))
logmodel = LogisticRegression()

logmodel.fit(X_train_tf, y_train_tf)



predictions_LR_tf = logmodel.predict(X_test_tf)



print("LR Accuracy -> ",accuracy_score(predictions_LR_tf, y_test_tf)*100)

print(classification_report(predictions_LR_tf,y_test_tf))