# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import sys

import warnings



if not sys.warnoptions:

    warnings.simplefilter("ignore")

    

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

_dir_ = '../input/'



import os

# print(os.listdir(_dir_))



# Any results you write to the current directory are saved as output.



all_stocks = pd.read_csv(_dir_+'all_stocks_5yr.csv')



# printing information of csv files

print(all_stocks.info())



all_stocks
# cleaning data



# filling null values with most repeated

all_stocks['open'] = all_stocks['open'].fillna(all_stocks['open'].value_counts().index[0]).astype(float)

all_stocks['high'] = all_stocks['high'].fillna(all_stocks['high'].value_counts().index[0]).astype(float)

all_stocks['low'] = all_stocks['low'].fillna(all_stocks['low'].value_counts().index[0]).astype(float)

all_stocks['close'] = all_stocks['close'].fillna(all_stocks['close'].value_counts().index[0]).astype(float)



# setting date to year and month from date column

all_stocks['Year'] = all_stocks["date"].str.split("-", n = 2, expand = True)[0].astype(int)

all_stocks['Month'] = all_stocks["date"].str.split("-", n = 2, expand = True)[1].astype(int)

all_stocks = all_stocks.drop('date',axis=1)



all_stocks
from sklearn import model_selection,metrics,preprocessing

from sklearn import naive_bayes,ensemble

preprocess = preprocessing.LabelEncoder()

def model(data,category=0,month=0):

    X = data.drop(['close'],axis=1)

#     print(X)

    y = preprocess.transform(data['close'])



    # Create Navie Bayes Model

    nb = naive_bayes.GaussianNB()

    nbmodel = nb.fit(X, y)



    # Create Random Forest Model

    rf = ensemble.RandomForestRegressor()  

    rfmodel = rf.fit(X, data['close'])

    

    return {'model':{'nb':nbmodel,'rf':rfmodel},'Name':category,'Month':month}



def predict(model,test):

    print(test)

    return {'Navie Bayes':model['nb'].predict(test),'Random Forest':model['rf'].predict(test)}
preprocess.fit(all_stocks['close'])

def groupByMonth(i):

    # i will have rows of particular month

    x = model(i.drop(['Name','Year','volume'],axis=1),i.Name.iloc[0],i.Month.iloc[0])

    return x

def groupByName(i):

    # i will have rows of particular company

    x = i.groupby(i['Month']).apply(groupByMonth)

    return x



output = all_stocks.groupby(all_stocks['Name']).apply(groupByName)



# output.to_csv("submission.csv", index="False")

output
test = [[184.03,197.86,184.04,189.46,'','FDX',2019,4]]

df = pd.DataFrame(test,columns=['open','high','low','close','volume','Name','Year','Month'])

print(df)
x = (output.index.tolist())

#print(x)

x = x.index((df['Name'][0],int(df['Month'][0])))

print(x)



array = output.iloc[x]

print(array)



value = predict(array['model'],df.drop(['Name','Year','volume','close'],axis=1))

print('\nNavie Bayes:',preprocess.inverse_transform(value['Navie Bayes'])[0],' Random Forest:',value['Random Forest'][0])