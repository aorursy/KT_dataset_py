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
# Importing required libraries



import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

from plotly import graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff

import nltk

nltk.download("popular")

from nltk.corpus import stopwords

import re

from nltk.tokenize import word_tokenize,sent_tokenize

from nltk.sentiment.vader import SentimentIntensityAnalyzer

import string

from collections import Counter
# Reading the data 

train_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv',delimiter=',',encoding='latin-1', na_filter=False)

test_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv',delimiter=',',encoding='latin-1', na_filter=False)

train_data.head()
train_data.info()
train_data.describe()
# Positive tweet

print("Positive Tweet example :",train_data[train_data['sentiment']=='positive']['text'].values[0])

#negative_text

print("Negative Tweet example :",train_data[train_data['sentiment']=='negative']['text'].values[0])

#neutral_text

print("Neutral tweet example  :",train_data[train_data['sentiment']=='neutral']['text'].values[0])
train_data.isna().sum()
train_data["sentiment"].value_counts()
#Number Of words in Selected Text

train_data['Num_words_ST'] = train_data['selected_text'].apply(lambda x:len(str(x).split())) 

#Number Of words in main text

train_data['Num_word_text'] = train_data['text'].apply(lambda x:len(str(x).split())) 

#Difference in Number of words text and Selected text

train_data['difference_in_words'] = train_data['Num_word_text'] - train_data['Num_words_ST'] 
train_data.head()
# Plotting the graph for Total Number of Individual Sentiment in Training Data

train_data.sentiment.value_counts().plot(figsize=(12,5),kind='bar',color='red');

plt.xlabel('Sentiment')

plt.ylabel('Total Number Of Individual Sentiment in Training Data')
test_data["sentiment"].value_counts()
# Plotting the graph for Total Number of Individual Sentiment in Testing Data

test_data.sentiment.value_counts().plot(figsize=(12,5),kind='bar',color='green');

plt.xlabel('Sentiment')

plt.ylabel('Total Number Of Individual Sentiment in Testing Data')
# Here we are plotting the number of words in each tweet sentiment-vise.

fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))

tweet_len=train_data[train_data['sentiment']=='positive']['text'].str.split().map(lambda x: len(x))

ax1.hist(tweet_len,color='green')

ax1.set_title('positive tweets')



tweet_len=train_data[train_data['sentiment']=='neutral']['text'].str.split().map(lambda x: len(x))

ax2.hist(tweet_len,color='gray')

ax2.set_title('neutral tweets')



tweet_len=train_data[train_data['sentiment']=='negative']['text'].str.split().map(lambda x: len(x))

ax3.hist(tweet_len,color='red')

ax3.set_title('negative tweets')



fig.suptitle('Words in a tweet')

plt.show()
# Function for pre-procesing.



stop_words = set(stopwords.words('english'))

def clean_tweets(x):

    # removing the hyperlinks

    clean1 = re.sub('https?://[A-Za-z0-9./]+','',x)

    # removing the hashtags

    clean2 = re.sub('#[A-Za-z0-9]+','',clean1)

    # removing @

    clean3 = re.sub('@[A-Za-z0-9]','',clean2)

    # removing punctuations and lower case conversion

    clean4 = re.sub(r'[^\w\s]','',clean3).lower()

    words = word_tokenize(clean4)

    # removing stopwords

    words = [w for w in words if not w in stop_words]

    sent = ' '.join(words)

    return sent
train_data["clean_text"] = train_data["text"].apply(clean_tweets)

train_data["clean_ST"] = train_data["selected_text"].apply(clean_tweets)
# This function would calculate jaccard score.

def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / ((len(a) + len(b) - len(c))+0.1)
# Here we are calculating jaccard score for each tweet where 1st sentence would be text and second sentence would be selected text.

# score close to says that the text and the selected text are almost similar.

jaccard_score1=[]



for ind,row in train_data.iterrows():

    sentence1 = row.text

    sentence2 = row.selected_text



    jaccard_score = jaccard(sentence1,sentence2)

    jaccard_score1.append([sentence1,sentence2,jaccard_score])
# Here we are creating a dataframe of Jaccard_score and then merging it with our main dataframe Train_data.

Jaccard_score = pd.DataFrame(jaccard_score1,columns=["text","selected_text","jaccard_score"])

train_data = train_data.merge(Jaccard_score,how='outer')
train_data.head()
# Here we have calculated Jaccard score sentiment-vise.

# From the below scores, we can say that for NEUTRAL SENTIMENT- Jaccard score is almost equal to 1 meaning the text and selected_text are almmost Equal.

train_data.groupby('sentiment').mean()['jaccard_score']
# Finding out the most common words in selected_text

# Counter Returns an itertool for all the elements with positive count in the Counter object.

train_data['temp_list'] = train_data['selected_text'].apply(lambda x:str(x).split())

top = Counter([item for sublist in train_data['temp_list'] for item in sublist])

temp = pd.DataFrame(top.most_common(30))

temp.columns = ['Common_words','count']

temp.style.background_gradient(cmap='Blues')
# Ploting the bar grapgh count of most common words(top 30) of selected_text

fig = px.bar(temp, x="count", y="Common_words", title='Commmon Words in Selected Text', orientation='h', 

             width=900, height=800,color='Common_words')

fig.show()
# Function for fetching words with highest polarity.

def high_polarity_words(data):

    

    training_data = data['text']

    training_data_sentiment = data['sentiment']

    selected_text_processed = []

    analyser = SentimentIntensityAnalyzer()

    

    for i in range(0 , len(training_data)):

        text = re.sub(r'http\S+', '', str(training_data.iloc[i]))

        

        if(training_data_sentiment.iloc[i] == "neutral"):

            selected_text_processed.append(str(text))   ## Since already the Neutral tweets have jaccard score almost close to 1, we do not process those tweets.

        

        # Here what we are doing is....... 

        # We will get those words which have high polarity for being positive.

        if(training_data_sentiment.iloc[i] == "positive"):

            orig_text = re.split(' ', text)  # Converting each row (string text) into list.

            #print(orig_text)

        

            high_words_arr = ""           # array to store High polarity words

            polarity = 0

            for j in range(0,len(orig_text)):

                score = analyser.polarity_scores(orig_text[j])   # Here score is a dictionary having 4 key value pairs negative, neutral , postive and compound.

                if score['compound'] >polarity:               # Setting up a threshold which is polarity(compound) greater than 0 then store that word because it has high polarity for being postive.

                    polarity = score['compound']              # here we have set Polarity > 0 as Positive and Polarity < 0 as Negative.

                    high_words_arr = orig_text[j]

            if len(high_words_arr) != 0:                         # Checking the len of selected_Word if not equal to 0, append that.

                selected_text_processed.append(high_words_arr)   

            if len(high_words_arr) == 0:                         # Chekcing the len of selected_word if equal to 0 then append the original text because there is no use of appending the selected_text because it is of length 0.

                selected_text_processed.append(text)

        

        # Here what we are doing is....... 

        # We will get those words which have high polarity for being negative.

        if(training_data_sentiment.iloc[i] == "negative"):

            orig_text = re.split(' ', text)

        

            high_words_arr = ""          # array to store High polarity words

            polarity = 0

            for j in range(0,len(orig_text)):

                score = analyser.polarity_scores(orig_text[j])

                if score['compound'] <polarity:               # Here score is a dictionary having 4 key value pairs negative, neutral , postive and compound.

                    polarity = score['compound']

                    high_words_arr = orig_text[j]

            if len(high_words_arr) != 0:                           # Checking the len of selected_Word if not equal to 0, append that.

                selected_text_processed.append(high_words_arr)   

            if len(high_words_arr) == 0:                           # Chekcing the len of selected_word if equal to 0 then append the original text because there is no use of appending the selected_text because it is of length 0. 

                selected_text_processed.append(text) 

                

    return selected_text_processed
train_selected_text = high_polarity_words(train_data)
len(train_selected_text)
train_selected_data = train_data['selected_text']

len(train_selected_data)
# Calculating the final Jaccard Score for training dataset.

train_selected_data = train_data['selected_text']

average = 0;

for i in range(0,len(train_selected_data)):

    jaccard_score = jaccard(str(train_selected_text[i]),str(train_selected_data[i]))

    average = jaccard_score+average     # Summing up all jaccard score

print('-----Training Data accuracy-----')

print(average/len(train_selected_text))   # Accuracy
# running function on test data to find high polarity words of respective sentiment.

test_selected_text = high_polarity_words(test_data)
# Submission File should have textID and seleceted_text

sample = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")

sample.loc[:, 'selected_text'] = test_selected_text

sample.to_csv("submission.csv", index=False)