# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import nltk

from nltk.corpus import stopwords

import re

  

# for Stemming propose  

from nltk.stem.porter import PorterStemmer 

from nltk.corpus import stopwords 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

stock_data = pd.read_csv("../input/DJIA_table.csv") 

news_data = pd.read_csv("../input/RedditNews.csv") 

news_data = news_data.merge(stock_data, on='Date')



# Any results you write to the current directory are saved as output.
# Remove all the special characters

news_data['News'] = [re.sub(r'\W', ' ', row) for row in news_data['News']]



# remove all single characters

news_data['News'] = [re.sub(r'\s+[a-zA-Z]\s+', ' ', row) for row in news_data['News']]



# Remove single characters from the start

news_data['News'] = [re.sub(r'\^[a-zA-Z]\s+', ' ', row) for row in news_data['News']]



# Substituting multiple spaces with single space

news_data['News'] = [re.sub(r'\s+', ' ', row, flags=re.I) for row in news_data['News']]



# Removing prefixed 'b'

news_data['News'] = [re.sub(r'^b\s+', '', row) for row in news_data['News']]



# Removing numbers

news_data['News'] = [re.sub(r'\d+', '', row) for row in news_data['News']]



news_data['News'] = [row.lower().split() for row in news_data['News']]
grouped_news = pd.DataFrame(columns=['Date','Stacked News','Open','High','Low','Close','Volume','Adj Close'])

grouped_news['Date'] = news_data.groupby('Date')['Date'].apply(set)

grouped_news['Stacked News'] = news_data.groupby('Date')['News'].apply(list)

grouped_news['Open'] = news_data.groupby('Date')['Open'].unique()

grouped_news['High'] = news_data.groupby('Date')['High'].unique()

grouped_news['Low'] = news_data.groupby('Date')['Low'].unique()

grouped_news['Close'] = news_data.groupby('Date')['Close'].unique()

grouped_news['Volume'] = news_data.groupby('Date')['Volume'].unique()

grouped_news['Adj Close'] = news_data.groupby('Date')['Adj Close'].unique()
grouped_news['Date'] = grouped_news['Date'].astype(str)

grouped_news['Date'] = [row.strip("{}''") for row in grouped_news['Date']]



grouped_news['Open'] = grouped_news['Open'].astype(str)

grouped_news['Open'] = [row.strip("[]") for row in grouped_news['Open']]

grouped_news['Open'] = grouped_news['Open'].astype(float)



grouped_news['High'] = grouped_news['High'].astype(str)

grouped_news['High'] = [row.strip("[]") for row in grouped_news['High']]

grouped_news['High'] = grouped_news['High'].astype(float)



grouped_news['Low'] = grouped_news['Low'].astype(str)

grouped_news['Low'] = [row.strip("[]") for row in grouped_news['Low']]

grouped_news['Low'] = grouped_news['Low'].astype(float)



grouped_news['Close'] = grouped_news['Close'].astype(str)

grouped_news['Close'] = [row.strip("[]") for row in grouped_news['Close']]

grouped_news['Close'] = grouped_news['Close'].astype(float)



grouped_news['Volume'] = grouped_news['Volume'].astype(str)

grouped_news['Volume'] = [row.strip("[]") for row in grouped_news['Volume']]

grouped_news['Volume'] = grouped_news['Volume'].astype(float)



grouped_news['Adj Close'] = grouped_news['Adj Close'].astype(str)

grouped_news['Adj Close'] = [row.strip("[]") for row in grouped_news['Adj Close']]

grouped_news['Adj Close'] = grouped_news['Adj Close'].astype(float)



grouped_news['Increase/Decrease'] = grouped_news['Close'].diff()

grouped_news.dropna(inplace=True)
import gensim 

from gensim.models import Word2Vec 



for itr,row in enumerate(grouped_news['Stacked News']):

    model = gensim.models.Word2Vec(grouped_news['Stacked News'][itr], size=300, window=5, min_count=1, workers=4)

    grouped_news['Stacked News'][itr] = [model[word] for word in grouped_news['Stacked News'][itr]]
grouped_news['Vectorized News'] = grouped_news['Stacked News']

for itr,row in enumerate(grouped_news['Vectorized News']):

    grouped_news['Vectorized News'][itr] = [np.mean(array) for array in grouped_news['Vectorized News'][itr]]
for itr,row in enumerate(grouped_news['Vectorized News']):

    min = np.min(grouped_news['Vectorized News'][itr])

    max = np.max(grouped_news['Vectorized News'][itr])

    grouped_news['Vectorized News'][itr] = [(row-min)/(max-min) for row in grouped_news['Vectorized News'][itr]]
grouped_news['Vectorized News'] = [np.mean(list_array) for list_array in grouped_news['Vectorized News']]
grouped_news_copy = grouped_news.copy()
grouped_news_copy.drop(['Date', 'Stacked News', 'Open', 'High', 'Low', 'Close', 'Volume',

       'Adj Close', 'Increase/Decrease'], axis=1, inplace=True)
X_train_Reg, X_test_Reg, y_train_Reg, y_test_Reg = train_test_split(grouped_news_copy, grouped_news['Increase/Decrease'].astype('int'), test_size=0.33, random_state=42)
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error



my_model_Reg = XGBRegressor()

my_model_Reg.fit(X_train_Reg, y_train_Reg.astype('int'), verbose=False)
predicted_Reg = my_model_Reg.predict(X_test_Reg)

print("Mean Error for Regression : ", np.mean(np.subtract(y_test_Reg, predicted_Reg)))

print("Root Mean Squared Error for Regression : ", np.sqrt(mean_squared_error(y_test_Reg, predicted_Reg)))