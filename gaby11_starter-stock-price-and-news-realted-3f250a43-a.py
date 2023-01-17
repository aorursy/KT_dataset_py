import sys

import matplotlib.pyplot as plt # plotting

plt.style.use('ggplot')

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression



import warnings

warnings.filterwarnings("ignore")
df1 = pd.read_csv('../input/MicrosoftFinalData.csv', delimiter=',')

df1.dataframeName = 'MicrosoftFinalData.csv'

nRow, nCol = df1.shape
df2 = pd.read_csv('../input/MicrosoftNewsStock.csv', delimiter=',')

df2.dataframeName = 'MicrosoftNewsStock.csv'

nRow, nCol = df2.shape
df2.head(5)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import unicodedata

sid = SentimentIntensityAnalyzer()



neg, neu, pos, compound = [], [], [], []



def sentiment(df):

    for i in range(len(df)):

        sen = unicodedata.normalize('NFKD', train.iloc[i]['News'])

        ss = sid.polarity_scores(sen)

        neg.append(ss['neg'])

        neu.append(ss['neu'])

        pos.append(ss['pos'])

        compound.append(ss['compound'])

    df['neg'] = neg

    df['neu'] = neu

    df['pos'] = pos

    df['compound'] = compound
def if_news(column):

    if column == 0:

        return 0

    else:

        return 1
df1['if_news'] = df1['compound'].apply(if_news)



df_weekly = df1[['Date','Close', 'compound', 'neg', 'pos', 'if_news']]

df_weekly['Date'] = pd.to_datetime(df_weekly['Date'])

df_weekly.set_index('Date',inplace=True)
def take_last(array_like):

    return array_like[-1]
output = df_weekly.resample('W', # Weekly resample

                    how={

                         'Close': take_last,

                         'compound': 'mean',

                         'neg': 'mean',

                         'pos': 'mean',

                         'if_news': 'max'

                    }, 

                    loffset=pd.offsets.timedelta(days=-6))
output.head(5)
def split(df, X, y, seed=42):

    """

    Split the data into a training and test set

    

    The training data should span the date range from 1/1/2018 to 6/30/2018

    The test data should span the date range from 7/1/2018 to 7/31/2018

    

    Parameters

    ----------

    X: DataFrame containing the independent variable(s) (i.e, features, predictors)

    y: DataFrame containing the dependent variable (i.e., the target)

    

    Optional

    --------

    seed: Integer used as the seed for a random number generator

      You don't necessarily NEED to use a random number generator but, if you do, please use the default value for seed

    

    Returns

    -------

    X_train: DataFrame containing training data for independent variable(s)

    X_test:  DataFrame containing test data for independent variable(s)

    y_train: DataFrame containing training data for dependent variable

    y_test:  DateFrame containing test data for dependent variable

    """

    # IF  you need to use a random number generator, use rng.

    rng = np.random.RandomState(seed)

    

    # Create the function body according to the spec

    train_range = (np.where(df.index<='2015-06-30'))

    test_range = (np.where(df.index>'2015-06-30'))

    

    # Split X and Y into train and test sets

    X_train, y_train = X.iloc[train_range], y.iloc[train_range]

    X_test, y_test = X.iloc[test_range], y.iloc[test_range]



    # Change the return statement as appropriate

    return X_train, X_test, y_train, y_test
def pd2ndarray( dfList ):

    """

    For each DataFrame in the list dfList, prepare the ndarray needed by the sklearn model

    

    Parameters

    ----------

    dfList: List of DataFrames

    

    Returns

    --------

    ndList: a list of ndarrays

    """

    

    # Create the function body according to the spec

    ndList = []

    for i in dfList:

        ndList.append(i.values)

    

    # Change the return statement as appropriate

    return ndList
pos_name = ['pos_1','pos_2', 'pos_3','pos_4', 'pos_5', 'pos_6', 'pos_7', 'pos_8','pos_9', 'pos_10', 'pos_11', 'pos_12','pos_13']

neg_name = ['neg_1','neg_2', 'neg_3','neg_4', 'neg_5', 'neg_6', 'neg_7','neg_8','neg_9','neg_10','neg_11','neg_12','neg_13']

if_news_name = ['if_news_1','if_news_2', 'if_news_3','if_news_4', 'if_news_5', 'if_news_6', 'if_news_7','if_news_8','if_news_9','if_news_10','if_news_11', 'if_news_12','if_news_13']
def cal_lag(df, k):

    for i in range(k):

        df[pos_name[i]] = df['pos'].shift(i+1)

        df[neg_name[i]] = df['neg'].shift(i+1)

        df[if_news_name[i]] = df['if_news'].shift(i+1)
output['ret'] = output['Close'].pct_change()

cal_lag(output, 13)

output = output.dropna()
X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []



# # Split the data into a training and a test set

def lag_split(df, k):

    for i in range(k):

        independent = [if_news_name[i], pos_name[i], neg_name[i]]

        X = df.loc[:, independent]

        y = df.loc[:, ['ret'] ]

        X_train, X_test, y_train, y_test = split(output, X, y)

        X_train, X_test, y_train, y_test = pd2ndarray( [X_train, X_test, y_train, y_test] )

        X_train_list.append(X_train)

        X_test_list.append(X_test)

        y_train_list.append(y_train)

        y_test_list.append(y_test)
lag_split(output, 13)
alpha, gamma, beta, delta = [], [], [], []



for i in range(13):

    reg = LinearRegression().fit(X_train_list[i], y_train_list[i])

    alpha.append(reg.intercept_[0])

    gamma.append(reg.coef_[0][0])

    beta.append(reg.coef_[0][1])

    delta.append(reg.coef_[0][2])
week = list(range(1,14))

gamma_cum = np.cumsum(gamma)

beta_cum = np.cumsum(beta)

delta_cum = np.cumsum(delta)
poly = np.polyfit(week,gamma_cum,5)

poly_1 = np.poly1d(poly)(week)

poly = np.polyfit(week,beta_cum,5)

poly_2 = np.poly1d(poly)(week)

poly = np.polyfit(week,delta_cum,5)

poly_3 = np.poly1d(poly)(week)
plt.figure(figsize=(8,6))

plt.title('Cumulative Weekly Average Regression Coefficients')

plt.xlabel('Week')

plt.ylabel('Regression Coefficient')

plt.plot(week,poly_1, label='News Effect') # return premium for firms that have neutral published news over firms with no news

plt.plot(week,poly_2, label='Positive Sentiment') # excess return on positive sentiment with lag k

plt.plot(week,poly_3, label='Negative Sentiment') # excess return on negative sentiment with lag k 

plt.legend()

plt.show()
# plt.plot(np.cumsum(alpha), label='alpha') # return with no news at lag k

plt.figure(figsize=(8,6))

plt.plot(week,np.cumsum(gamma), label='News Effect') # return premium for firms that have neutral published news over firms with no news

plt.plot(week,np.cumsum(beta), label='Positive Sentiment') # excess return on positive sentiment with lag k

plt.plot(week,np.cumsum(delta), label='Negative Sentiment') # excess return on negative sentiment with lag k 

plt.legend()

plt.show()
from nltk.tokenize import word_tokenize

text = "All models are wrong, but some are useful."

print(word_tokenize(text))
from nltk.corpus import stopwords

word_list = word_tokenize(text)

filtered_word_list = word_list[:] #make a copy of the word_list

for word in word_list: # iterate over word_list

    if word in stopwords.words('english'): 

        filtered_word_list.remove(word)
filtered_word_list
from nltk.stem import PorterStemmer

from nltk.stem import LancasterStemmer



porter = PorterStemmer()

lancaster=LancasterStemmer()

stemmed_list = []



print("{0:20}{1:20}{2:20}".format("Word","Porter Stemmer","lancaster Stemmer"))

for word in filtered_word_list:

    print("{0:20}{1:20}{2:20}".format(word,porter.stem(word),lancaster.stem(word)))

    stemmed_list.append(porter.stem(word))

    
import nltk

from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()



stemmed_list

print("{0:20}{1:20}".format("Word","Lemma"))

for word in stemmed_list:

    print ("{0:20}{1:20}".format(word,wordnet_lemmatizer.lemmatize(word)))
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import unicodedata

sid = SentimentIntensityAnalyzer()
sen = unicodedata.normalize('NFKD', text)

ss = sid.polarity_scores(sen)
ss