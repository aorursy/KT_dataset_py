# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importing required packages

import re,nltk

from nltk.corpus import stopwords

from nltk.tokenize.treebank import TreebankWordDetokenizer

from nltk.tokenize import word_tokenize

import spacy

from nltk.sentiment.vader import SentimentIntensityAnalyzer
#downloading the required packages from nltk downloader

nltk.download(['punkt','stopwords','vader_lexicon'])
#defining a function to read and preprocess data

def preprocessor(text):

    #using regex to remove http from the dataframe

    text=str(text).lower()

    text=re.sub('http\S+','',text)

    return text
#reading the trainset

trainset=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
#obtaining cleaned text

trainset['cleaned_data']=trainset['text'].apply(preprocessor)
#viewing trainset's first 5 rows

trainset.head()
#dropping the original text

trainset.drop(['text'],1,inplace=True)
trainset.head()
#function for Sentiment Intensity Analyser and returning the processed_text

def polarity_determiner(df_process):

    train_data = df_process['cleaned_data']

    train_data_sentiment = df_process['sentiment']

    #initialising a list that contains all the processed text

    selected_text_processed = []

    #initialising the Sentiment Intensity Analyser

    #this will determine the polarity of each word

    analyser = SentimentIntensityAnalyzer()

    for j in range(0 , len(train_data)):

        #using regex to remove http from the train_data

        text = re.sub(r'http\S+', '', str(train_data.iloc[j]))

        #for neutral similarity, all text is appended 

        if(train_data_sentiment.iloc[j] == "neutral" or len(text.split()) < 2):

            selected_text_processed.append(str(text))

        #for sentiments like positive and negative, only words that have the highest polarity are determined as these words strongly determine the sentiment

        if(train_data_sentiment.iloc[j] == "positive" and len(text.split()) >= 2):

            aa = re.split(' ', text)

        

            ss_arr = ""

            #assigning an initial polarity of 0

            polar = 0

            for qa in range(0,len(aa)):

                score = analyser.polarity_scores(aa[qa])

                if score['compound'] >polar:

                    polar = score['compound']

                    ss_arr = aa[qa]

            if len(ss_arr) != 0:

                selected_text_processed.append(ss_arr)   

            if len(ss_arr) == 0:

                selected_text_processed.append(text)

        #repeating the same case for negative sentiment

        if(train_data_sentiment.iloc[j] == "negative"and len(text.split()) >= 2):

            aa = re.split(' ', text)

        

            ss_arr = ""

            polar = 0

            for qa in range(0,len(aa)):

                score = analyser.polarity_scores(aa[qa])

                if score['compound'] <polar:

                    polar = score['compound']

                    ss_arr = aa[qa]

            if len(ss_arr) != 0:

                selected_text_processed.append(ss_arr)   

            if len(ss_arr) == 0:

                selected_text_processed.append(text)  

    return selected_text_processed
selected_train=polarity_determiner(trainset)
len(selected_train)
#reading the testset into a dataframe

testset=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
#same for test dataset

testset['cleaned_data']=testset['text'].apply(preprocessor)
selected_test=polarity_determiner(testset)
text_id=testset['textID']
idlist=[text_id.iloc[i] for i in range(len(text_id))]
df_sub=pd.DataFrame({'textID':idlist,'selected_text':selected_test})
df_sub.head()
df_sub.to_csv('/kaggle/working/submission.csv',index=False)