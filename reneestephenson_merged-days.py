import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from wordcloud import STOPWORDS

stopwords = set(STOPWORDS)



import re



from collections import Counter



import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#First 12 cells are reading, cleaning and filtering most of the data.

#Read data into variables the number after df represents the day in april

df9=pd.read_csv('/kaggle/input/coronavirus-covid19-tweets-early-april/2020-04-09 Coronavirus Tweets.CSV')

df4=pd.read_csv('/kaggle/input/coronavirus-covid19-tweets-early-april/2020-04-04 Coronavirus Tweets.CSV')

df5=pd.read_csv('/kaggle/input/coronavirus-covid19-tweets-early-april/2020-04-05 Coronavirus Tweets.CSV')

df14=pd.read_csv('/kaggle/input/coronavirus-covid19-tweets-early-april/2020-04-14 Coronavirus Tweets.CSV')

df14
#Drop the extra columns

df9.drop(['status_id','user_id','screen_name','source','reply_to_status_id','reply_to_user_id','is_retweet','place_full_name','place_type','reply_to_screen_name','is_quote','followers_count','friends_count','account_lang','account_created_at','verified'],axis=1, inplace = True)

df4.drop(['status_id','user_id','screen_name','source','reply_to_status_id','reply_to_user_id','is_retweet','place_full_name','place_type','reply_to_screen_name','is_quote','followers_count','friends_count','account_lang','account_created_at','verified'],axis=1, inplace = True)

df5.drop(['status_id','user_id','screen_name','source','reply_to_status_id','reply_to_user_id','is_retweet','place_full_name','place_type','reply_to_screen_name','is_quote','followers_count','friends_count','account_lang','account_created_at','verified'],axis=1, inplace = True)

df14.drop(['status_id','user_id','screen_name','source','reply_to_status_id','reply_to_user_id','is_retweet','place_full_name','place_type','reply_to_screen_name','is_quote','followers_count','friends_count','account_lang','account_created_at','verified'],axis=1, inplace = True)
#Filter out all other languages. English only

df9=df9[(df9.lang == "en")].reset_index(drop = True)

df9.drop(['country_code','lang'],axis=1,inplace=True)



df4=df4[(df4.lang == "en")].reset_index(drop = True)

df4.drop(['country_code','lang'],axis=1,inplace=True)



df5=df5[(df5.lang == "en")].reset_index(drop = True)

df5.drop(['country_code','lang'],axis=1,inplace=True)



df14=df14[(df14.lang == "en")].reset_index(drop = True)

df14.drop(['country_code','lang'],axis=1,inplace=True)
#Data preprocessing - df9 - TAKES FOREVER TO RUN!

for i in range(df9.shape[0]) :

    df9['text'][i] = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(#[A-Za-z0-9]+)", " ", df9['text'][i]).split()).lower()

    

df9
#Data preprocessing - df4 - TAKES FOREVER TO RUN!

for i in range(df4.shape[0]) :

    df4['text'][i] = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(#[A-Za-z0-9]+)", " ", df4['text'][i]).split()).lower()

df4
#Data preprocessing - df5 - TAKES FOREVER TO RUN!

for i in range(df5.shape[0]) :

    df5['text'][i] = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(#[A-Za-z0-9]+)", " ", df5['text'][i]).split()).lower()

df5
#Data preprocessing - df14 - TAKES FOREVER TO RUN!

for i in range(df14.shape[0]) :

    df14['text'][i] = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(#[A-Za-z0-9]+)", " ", df14['text'][i]).split()).lower()

df14
#Forgot to drop some columns

df9.drop(['created_at', 'favourites_count', 'retweet_count'],axis=1, inplace = True)



df4.drop(['created_at', 'favourites_count', 'retweet_count'],axis=1, inplace = True)



df5.drop(['created_at', 'favourites_count', 'retweet_count'],axis=1, inplace = True)



df14.drop(['created_at', 'favourites_count', 'retweet_count'],axis=1, inplace = True)
stopwords
#Remove Stop Words

df9['text'] = df9['text'].apply(lambda df9: ' '.join([word for word in df9.split() if word not in stopwords]))



df4['text'] = df4['text'].apply(lambda df4: ' '.join([word for word in df4.split() if word not in stopwords]))



df5['text'] = df5['text'].apply(lambda df5: ' '.join([word for word in df5.split() if word not in stopwords]))



df14['text'] = df14['text'].apply(lambda df14: ' '.join([word for word in df14.split() if word not in stopwords]))

#Remove empty rows

df9 = df9[df9['text'].notnull()]



df4 = df4[df4['text'].notnull()]



df5 = df5[df5['text'].notnull()]



df14 = df14[df14['text'].notnull()]
#Output cleaned and sorted data into csv files

df9.to_csv('2020-04-09.csv')



df4.to_csv('2020-04-04.csv')



df5.to_csv('2020-04-05.csv')



df14.to_csv('2020-04-14.csv')
#The csv files were downloaded and merged in windows cmd. The merge file was uploaded to kaggle

#Read merged csv file into one variable

dataset=pd.read_csv('/kaggle/input/project/all.csv', sep='delimiter', header=None)

#Merged file changed text column header to 0. Should be read as dataset['text']. Rename the header:

dataset = dataset.rename(columns={0: 'text'})

dataset
#Filter the texts. Takes out rows that do not contain symptoms

dataset = dataset[dataset['text'].str.contains("fever" or "cough" or "tierd" or "sore" or "aches" or "pains" or "headache" or "diarrhoea")]

dataset
#Split all the words 

words = []

words = [word for i in dataset.text for word in i.split()]

words
#Create a list of all the symptoms. Drop all the words that are not in the list

drops = ['fever', 'cough',  'tierd', ' sore', 'aches', 'pains','headache', 'diarrhoea']

for word in list(words):

    if word not in drops:

        words.remove(word)
#Check if the words were dropped 

words
#Count the frequency of the words 

freq = Counter(words).most_common(30)

freq = pd.DataFrame(freq)

freq.columns = ['word', 'frequency']

freq
#Create bar chart 

plt.figure(figsize = (10, 10))

sns.barplot(y="word", x="frequency",data=freq );