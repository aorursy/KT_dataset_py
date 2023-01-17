import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import sys

import nltk

import sklearn

import re

import string

import heapq

import scipy.io

from array import *

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

%matplotlib inline

from plotly import graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff

from collections import Counter
df_train=pd.read_csv('../input/tweet-sentiment-extraction/train.csv',na_filter=False)

df_test=pd.read_csv('../input/tweet-sentiment-extraction/test.csv',na_filter=False)

FINAL=pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')
print(df_train.info())

print(df_train.head())
print(df_test.info())

print(df_test.head())
df_train.isna().sum()
df_test.isna().sum()
# check class distribution

classes = df_train['sentiment']

print(classes.value_counts())
df_train['Num_words_ST'] = df_train['selected_text'].apply(lambda x:len(str(x).split()))

df_train['Num_word_text'] = df_train['text'].apply(lambda x:len(str(x).split()))

df_train['difference_in_words'] = df_train['Num_word_text'] - df_train['Num_words_ST'] 

df_train.head()
# Plot the graph

df_train.sentiment.value_counts().plot(figsize=(12,5),kind='bar',color='orange');

plt.xlabel('Sentiment')

plt.ylabel(' Sentiments for Training Data')
def cleaning(text):



    text = str(text).lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    

    return text
df_train['text'] = df_train['text'].apply(lambda x:cleaning(x))

df_train['selected_text'] = df_train['selected_text'].apply(lambda x:cleaning(x))
# removing stopwords

from nltk.corpus import stopwords

nltk.download('stopwords')

def remove(x):

    return [y for y in x if y not in stopwords.words('english')]
df_train['temp_list_T'] = df_train['text'].apply(lambda x:str(x).split())

df_train['temp_list_T'] = df_train['temp_list_T'].apply(lambda x:remove(x))
df_train['temp_list_ST'] = df_train['selected_text'].apply(lambda x:str(x).split())

df_train['temp_list_ST'] = df_train['temp_list_ST'].apply(lambda x:remove(x))
# calculation of jaccard score

def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / ((len(a) + len(b) - len(c))+0.1)



jaccard_score1=[]



for ind,row in df_train.iterrows():

    sentence1 = row.text

    sentence2 = row.selected_text



    jaccard_score = jaccard(sentence1,sentence2)

    jaccard_score1.append([sentence1,sentence2,jaccard_score])



    

# putting jaccard output in main train file

Jaccard_score = pd.DataFrame(jaccard_score1,columns=["text","selected_text","jaccard_score"])

df_train = df_train.merge(Jaccard_score,how='outer')

df_train.head()
Pos = df_train[df_train['sentiment']=='positive']

Ne = df_train[df_train['sentiment']=='negative']

N = df_train[df_train['sentiment']=='neutral']
#most positive words finder

top = Counter([item for sublist in Pos['temp_list_ST'] for item in sublist])

temp_positive = pd.DataFrame(top.most_common(20))

temp_positive.columns = ['Common_words','count']

temp_positive.style.background_gradient(cmap='Blues')
# most number of negative words finder

top = Counter([item for sublist in Ne['temp_list_ST'] for item in sublist])

temp_negative = pd.DataFrame(top.most_common(20))

temp_negative = temp_negative.iloc[1:,:]

temp_negative.columns = ['Common_words','count']

temp_negative.style.background_gradient(cmap='Blues')
#MosT number of neutral words finder

top = Counter([item for sublist in N['temp_list_ST'] for item in sublist])

temp_neutral = pd.DataFrame(top.most_common(20))

temp_neutral = temp_neutral.loc[1:,:]

temp_neutral.columns = ['Common_words','count']

temp_neutral.style.background_gradient(cmap='Blues')
def polar(data):

    

    training_data = data['text']

    training_data_sentiment = data['sentiment']

    selected_text_processed = []

    analyser = SentimentIntensityAnalyzer()



    

    for i in range(len(training_data)):

        text = re.sub(r'http\S+', '', str(training_data.iloc[i]))

    

        score = []

    

        if(training_data_sentiment.iloc[i] == "positive"):

            

            words = re.split(' ', text)

            for w in words:

                score.append(analyser.polarity_scores(w)['compound'])

            

            maximum = np.argmax(score)

            word = words[maximum]

            selected_text_processed.append(word)

        

        if(training_data_sentiment.iloc[i] == "negative"):

            

            words = re.split(' ', text)

            for w in words:

                score.append(analyser.polarity_scores(w)['compound'])

       

            maximum = np.argmin(score)

            word = words[maximum]

            selected_text_processed.append(word)

        

        if(training_data_sentiment.iloc[i] == "neutral"):

        

            selected_text_processed.append(text)

            

    return selected_text_processed            
trainST = polar(df_train)

len(trainST)
# Calculation of jaccard value

train_selected_data = df_train['selected_text']

average = 0;

for i in range(0,len(train_selected_data)):

    jaccard_score = jaccard(str(trainST[i]),str(train_selected_data[i]))

    average = jaccard_score+average 

print('Training accuracy')

print(average/len(trainST))
# using our made func to find words with high polar value

testST = polar(df_test)
#FINAL

FINAL['selected_text'] =testST

FINAL.head(20)
FINAL.to_csv('submission.csv',index = False)