%matplotlib inline

import pandas as pd

import numpy as np

import re

import os

import math

import seaborn as sns

import matplotlib.pyplot as plt

import multiprocessing as mp

from datetime import datetime

from collections import Counter

from scipy.fftpack import fft



from fbprophet import Prophet

from statsmodels.tsa.arima_model import ARIMA

import warnings



kaggle_on = True



if kaggle_on:

    path = '../input/'

else:

    path = 'data/'



df_train = pd.read_csv(path + 'train_1.csv', nrows=150000)



null_df = df_train.isnull().sum(axis=1)

df_train = df_train.fillna(0)

print('Len of data: ', len(df_train.index))

print('Number of pages with nan values: ', len(null_df[null_df > 0]))

print('Number of pages with all nan values: ', len(null_df[null_df == len(df_train.columns)-1]))
# Convert page views to integers

for col in df_train.columns[1:]:

    df_train[col] = pd.to_numeric(df_train[col], downcast='integer')
cols = df_train.columns[1:-1]



def filter_df(df, word):

    df_new = df[df['Page'].str.contains(word)]

    apple_pages = df_new.Page.values

    df_new = df_new[cols].transpose()

    df_new[word] = df_new.values.sum(axis=1)

    return df_new[[word]]



word_to_filter_by = ['Apple_Inc', 'Microsoft', 'Facebook', 'Google']



df_companies = pd.DataFrame()

for word in word_to_filter_by:

    df_tmp = filter_df(df_train, word)

    df_companies = pd.concat([df_companies, df_tmp], axis=1)



print(df_companies.idxmax(axis=0))

df_companies.plot()



# mark Apple releases and other important dates during time period

if False:

    holidays = ['2015-11-26', '2015-12-25']

    stock_dates = ['2016-08-10']

    apple_dates = ['2015-07-15', '2015-09-09', '2015-09-25', '2015-10-13', '2015-10-26', '2015-11-11',

                  '2016-03-31', '2016-04-19', '2016-09-16', '2016-10-27', '2016-12-19']

    for date in holidays + stock_dates:

        plt.axvline(df_companies.index.get_loc(date), color='black', linestyle='solid')

plt.show()    
for col in df_companies.columns:

    print(col)

    fig, (ax0, ax1) = plt.subplots(nrows=2)

    df_companies[col].hist(bins=100, ax=ax0)

    q = df_companies[col].quantile(0.99)

    df_companies_filt = df_companies[df_companies[col] < q]

    df_companies_filt[col].hist(bins=100)

    

    fig, (ax00, ax11) = plt.subplots(nrows=2)

    df_companies[col].plot(ax=ax00)

    df_companies_filt[col].plot(ax=ax11)

    plt.show()