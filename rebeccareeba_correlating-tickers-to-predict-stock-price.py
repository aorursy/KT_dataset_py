import datetime as dt

from datetime import datetime

import matplotlib.pyplot as plt

from matplotlib import style

import pandas as pd

import pandas_datareader.data as web

import plotly.offline as py

import plotly.graph_objs as go

import matplotlib.dates as mdates

from plotly import tools

import plotly.figure_factory as ff

from collections import Counter
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

df = pd.read_csv('../input/all_stocks_5yr.csv')

df['date'] = pd.to_datetime(df['date'])

df.head()



# Any results you write to the current directory are saved as output.
plt.figure(figsize=(15, 5));

plt.subplot(1,2,1);

plt.plot(df[df.Name == 'AAPL'].open.values, color='red', label='open')

plt.plot(df[df.Name == 'AAPL'].close.values, color='green', label='close')

plt.plot(df[df.Name == 'AAPL'].low.values, color='blue', label='low')

plt.plot(df[df.Name == 'AAPL'].high.values, color='black', label='high')

plt.title('stock price')

plt.xlabel('time [days]')

plt.ylabel('price')

plt.legend(loc='best')

#plt.show()



plt.subplot(1,2,2);

plt.plot(df[df.Name == 'AAPL'].volume.values, color='black', label='volume')

plt.title('stock volume')

plt.xlabel('time [days]')

plt.ylabel('volume')

plt.legend(loc='best');
df_pivot=df.pivot(index='date', columns='Name', values='close')

df_pivot.head()
def visualize_data():



    df_corr = df_pivot.corr()

    data = df_corr.values

    fig = plt.figure()

    ax = fig.add_subplot(1,1,1)

    

    heatmap = ax.pcolor(data, cmap = plt.cm.RdYlGn)

    fig.colorbar(heatmap)

    ax.set_xticks(np.arange(data.shape[0]+0.5,minor = False))

    ax.set_yticks(np.arange(data.shape[1]+0.5,minor = False))

    ax.invert_yaxis()

    ax.xaxis.ticks_top()

    

    column_labels = df_corr.columns

    row_labels = df_corr.index

    

    ax.set_xticklabels(column_labels)

    ax.set_ylabels(row_labels)

    plt.xticks(rotation = 90)

    heatmap.set_clim(-1,1)

    plt.tight_layout()

    plt.show

    

    #print(df_corr.head())

    

visualize_data()
def process_data_for_labels(ticker):

    hm_days = 7

    

    tickers = df_pivot.columns.values.tolist()

    df_pivot.fillna(0, inplace = True)

    

    for i in range(1, hm_days+1): #we add 1 to hm_days as it starts from 0

        #Custon colum which gives i days in the future the value of the 

        # stock in percentchange

        df_pivot['{}_{}d'.format(ticker,i)] = (df_pivot[ticker].shift(-i)- df_pivot[ticker])/df_pivot[ticker]

        

    df_pivot.fillna(0,inplace = True)

    

    return tickers,df_pivot

process_data_for_labels('AAPL')
df.head()
def buy_sell_hold(*args): # *args lets you pass any number of parameter

    cols = [c for c in args]#if you pass collumns this will go row by row

    # here we will pass the whole week of percent changes

    requirement = 0.02

    #if our stock prices changes more than 2% in the next seven days

    for col in cols:

        if col > 0.028:

            return 1 #buy

        if col < -0.027 :

            return -1 #sell

        

    return 0 #hold
def extract_featuresets(ticker):

    tickers, df_pivot =  process_data_for_labels(ticker)# returns the tickers and a df with features(% change in price)



    df_pivot['{}_target'.format(ticker)] = list(map(buy_sell_hold,df_pivot['{}_1d'.format(ticker)],

                                                            df_pivot['{}_2d'.format(ticker)],

                                                            df_pivot['{}_3d'.format(ticker)],

                                                            df_pivot['{}_4d'.format(ticker)],

                                                            df_pivot['{}_5d'.format(ticker)],

                                                            df_pivot['{}_6d'.format(ticker)],

                                                            df_pivot['{}_7d'.format(ticker)]))

    vals = df_pivot['{}_target'.format(ticker)].values.tolist()

    str_vals = [str(i) for i in vals]

    print('Data spread:', Counter(str_vals)) # this will give the distribution

    # eg 1500 may br buy 800 may be sell, basically we will have more buy than sell

    # we want to see the spread, we want to see how acurate is our classifier

    # because we need to beat buy buybuy

    

    df_pivot.fillna(0, inplace= True)

    df_pivot = df_pivot.replace([np.inf, -np.inf],np.nan)# say if the the company went out of bussiness or if the

    #company just came to the market we put nan

    

    df_pivot.dropna(inplace = True)

    

    # create the feature sets and labels seperately

    df_vals = df_pivot[[ticker for ticker in tickers]].pct_change()

    df_vals = df_vals.replace([np.inf, -np.inf], 0)

    df_vals.fillna(0, inplace =True)

    

    # if you pass future values to the classifier as one of the featers

    # the classifer will learn this is important and will give you a false postive

    # be explicit which columns are the classifiers or you will get a very high accuracy

    X = df_vals.values

    y = df_pivot['{}_target'.format(ticker)].values

    

    return X, y, df



extract_featuresets('AAPL')
from sklearn import svm, neighbors 

from sklearn import model_selection

# cross validate helps shuffle and divide traning and testing data

from sklearn.ensemble import VotingClassifier, RandomForestClassifier

# Voting Classifier not one classifier but many classifier and vote to see which is best
# final function for doing typical machine learning

def do_ml(ticker):

    X,y,df = extract_featuresets(ticker)

# X is the percent change data for all the companies

# y is the target _1,0,1

    X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.2)

    

    #create a classifier

    #clf = neighbors.KNeighborsClassifier()# Simple classifier

    #use 3 classifiers and make them vote to see the best

    clf = VotingClassifier([('lsvc', svm.LinearSVC()), #linear Support Vector Machine

                            ('knn',neighbors.KNeighborsClassifier()),#k nearest neighbors

                            ('rfor', RandomForestClassifier())])

    

    # X is the feature and y is the target

    # we are trying to fit the input data to the target we arre setting

    clf.fit(X_train,y_train)

    # if you are happy with the confidence you dont need tto retrain the modle

    # you can pickle it

    confidence = clf.score(X_test, y_test)

    predictions = clf.predict(X_test)

    

    print('Predicted spread: ',Counter(predictions))

    print('Accuracy:', confidence)

    

    return confidence



do_ml('AAPL')