# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from langdetect import detect



from sklearn.feature_extraction.text import CountVectorizer

from nltk import ngrams

from nltk.corpus import stopwords



plt.style.use('fivethirtyeight') 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# Any results you write to the current directory are saved as output.
stocks=pd.read_csv("../input/financial-tweets/stocks_cleaned.csv")

stockerbot=pd.read_csv("../input/financial-tweets/stockerbot-export.csv",error_bad_lines=False) #Some lines prevent the file from being loaded, we'll just skip them here

stockerbot=stockerbot.drop(columns=['id'])

stockerbot["timestamp"] = pd.to_datetime(stockerbot["timestamp"])

stockerbot["text"] = stockerbot["text"].astype(str)

stockerbot["url"] = stockerbot["url"].astype(str)



stockerbot["company_names"] = stockerbot["company_names"].astype("category")

stockerbot["symbols"] = stockerbot["symbols"].astype("category")

stockerbot["source"] = stockerbot["source"].astype("category")

stockerbot.dtypes
stockerbot['date'] = stockerbot['timestamp'].dt.date

stockerbot['time'] = stockerbot['timestamp'].dt.time
stockerbot.isnull().any() #company_namaes & url have missing values

stockerbot[stockerbot['company_names'].isnull()] #1 line

stockerbot[stockerbot['url'].isnull()] #6369 lines : yet they are verified



#stockerbot  = stockerbot[stockerbot['verified'] == True] #Dropping unverified tweets (~10% of all tweets)



ts = pd.Series(stockerbot['date'].values, index=stockerbot['date'])

stockerbot['date'].value_counts().sort_values()  # Why is there such a spike on 2018-07-18? More than 90% of the tweets take place from 07-16 to 07-18.
total_companies = stockerbot["symbols"].value_counts() #Quoted companies

total_sources = stockerbot["source"].value_counts() #Different sources : 

print(total_sources)

total_sources.head(50).plot.bar() #Allows us to see the biggest sources and how they are leveled. 

#3 biggest sources are quoteed ~900 times, three nexts ~620 times, then it falls to ~350



#Room for improvement: Check the sources confidence, see how much tweets are verified, etc.





print(total_companies)

total_companies.head(50).plot.bar()
stop = stopwords.words('english')

stop.append("RT")

url_regex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"

handle_regex= "^@?(\w){1,15}$"





stockerbot['text']=stockerbot['text'].str.replace(url_regex, '')

stockerbot['text']=stockerbot['text'].str.replace(handle_regex, '')



stockerbot['text']=stockerbot['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
stockerbot = stockerbot[stockerbot["source"] != "test5f1798"]
word_vectorizer = CountVectorizer(ngram_range=(2,2), analyzer='word')

sparse_matrix = word_vectorizer.fit_transform(stockerbot['text'])

frequencies = sum(sparse_matrix).toarray()[0]
top_2grams = pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency']).sort_values(by=['frequency'],ascending=False)

top_2grams

top_2grams.head(50).plot.bar() #Allows us to see the biggest sources and how they are leveled. 

word_vectorizer = CountVectorizer(ngram_range=(3,3), analyzer='word')

sparse_matrix = word_vectorizer.fit_transform(stockerbot['text'])

frequencies = sum(sparse_matrix).toarray()[0]
top_3grams = pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency']).sort_values(by=['frequency'],ascending=False)

top_3grams

top_3grams.head(50).plot.bar()

# what is "amp" we had "amp amp" first and now "amp amp amp"
unique_companies=np.array(stockerbot["symbols"].unique())



companies_df={}







for company in unique_companies:

    if (len(stockerbot[stockerbot["symbols"] == company])>=65): # 50% of the companies has been quoted at least 65 times (thanks to total_companies.describe())

        word_vectorizer = CountVectorizer(ngram_range=(2,3), analyzer='word')

        sparse_matrix = word_vectorizer.fit_transform(stockerbot.loc[stockerbot["symbols"] == company,['text']]['text'])

        frequencies = sum(sparse_matrix).toarray()[0]

        

        ### Now let's run bi/tri gram analysis (for each company and keep the 10 most relevant for each.)

        companies_df[company]=pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency']).sort_values(by=['frequency'],ascending=False).head(10)    





companies_df
