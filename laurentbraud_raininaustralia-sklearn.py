# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))

import math



import sklearn

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error



#TODO  https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

#from keras.models import Sequential

#from keras.layers import Dense

#from keras.layers import LSTM



# print(tf.__version__)

# fix random seed for reproducibility

np.random.seed(7)
mydateparser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d")

df = pd.read_csv("../input/weatherAUS.csv", parse_dates=['Date'], date_parser=mydateparser)

df.head(5)
#df.describe()

df.info()
#Location

#

print(len(df["Location"].unique()))

df["Location"].unique()

# todo : place them on a map, distance vetween town, distance to ocean (in W / E/ / S / Noth)
#df.describe('')

#

# MinTemp et MaxTemp

%matplotlib inline

import matplotlib.pyplot as plt

df.hist(bins=50, figsize=(20,15))

plt.show()







#df.info()
from sklearn.model_selection import train_test_split

# V1 : Totalement aléatoire.



train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

train_set.dropna()



# https://www.kaggle.com/jsphyg/weather-dataset-rattle-package/discussion/78316#latest-495878

train_set = train_set.drop('RISK_MM', axis=1)

test_set = test_set.drop('RISK_MM', axis=1)



X_train = train_set

def removeUnusedVariable(df):

    df.drop('Location', axis=1, inplace=True)

    #df.drop('Date', axis=1, inplace=True)

    

removeUnusedVariable(X_train)



# Modification vers numérique

X_train["RainToday"] = X_train["RainToday"].map({'No': 0, 'Yes': 1})

X_train["RainTomorrow"] = X_train["RainTomorrow"].map({'No': 0, 'Yes': 1})

X_train.head(5)







#import time

# a/ As a number 1-12 is month revelant ? => No

# b/ is a numer 1-6 

import datetime

def processDate(df):

    df['Month'] = df['Date'].dt.month

    #datetime.date(2008, 12, 25)

    #df['SummerDay1'] = df['Date'] - datetime.date(df['Date'].dt.year, 6, 21)

    df['SummerDay'] = df['Month'].map({1: 6, 2:5, 3:4, 4:3, 5:2, 6:1, 7:1, 8:2, 9:3, 10:4, 11:5, 12:6})

    df.drop('Month', axis=1, inplace=True)

    df.drop('Date', axis=1, inplace=True)



    

processDate(X_train)

X_train.head(5)
# return 2 values :  

#   -1 to 1 for West to East

#  - 1 to 1 for North to South

# winDirection : exemple S, NE, NNE (Max : 3 caracter)





#df["WindGustDirCode"] = df["WindGustDir"]

def convertWindDirectionGlobal(winDirection) :

    nbValue = len(winDirection)

    # 6  : 3 * 2 * 1 (nb caractere commun)

    coef = 6 / nbValue

    dirWE = winDirection.count("W") * -1

    dirWE += winDirection.count("E")

    dirWE = (dirWE * coef) / 6

    

    dirNS = winDirection.count("N")* -1

    dirNS += winDirection.count("S")

    dirNS = (dirNS * coef) / 6

    return dirWE, dirNS



def convertWindDirectionAxe(winDirection, neg, pos) :

    nbValue = len(winDirection)

    # 6  : 3 * 2 * 1 (nb caractere commun)

    coef = 6 / nbValue

    dir = winDirection.count(neg) * -1

    dir += winDirection.count(pos)

    dir = (dir * coef) / 6

    

    return dir



        

#from sklearn.preprocessing import FunctionTransformer

#attr_adder = WindAttributesConverter()

#df["WindGustDirCode"] = df["WindGustDir"].str.len()

#df["WindGustDirCode"] = attr_adder.transform(df)

#df.head(5)**

#attr_adder = FunctionTransformer(convertWindDirectionGlobal)



def changeColumnWinDir(df, name):

    # WE => 2 valeurs opposées. Resultat sera 0

    df[name].fillna("WE", inplace=True)

    df[name + "WE"] = df[name].map(lambda x: convertWindDirectionAxe(x, "W", "E") , na_action='ignore' )

    df[name + "NS"] = df[name].map(lambda x: convertWindDirectionAxe(x, "N", "S") , na_action='ignore' )

    df.drop(name, axis=1, inplace=True)

    

def changeAllColumnWinDir(df):

    changeColumnWinDir(df, "WindGustDir")

    changeColumnWinDir(df, "WindDir9am")

    changeColumnWinDir(df, "WindDir3pm")

    #categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am']

    #df = pd.get_dummies(df, columns=categorical_columns)

    

    return df

X_train = changeAllColumnWinDir(X_train)







#X_train["WindGustDir"] 

#X_train["WindGustDirNS"] = X_train["WindGustDir"]

#X_train["WindGustDirNS"][1][0]



X_train.head(10)

#df2 = X_train2

# correlation

#corr_matrix = df2.corr()

#corr_matrix

#df["Evaporation"].value_counts()
#from pandas.plotting import scatter_matrix

#attributes = ["RainTomorrow","RainToday","WindGustDirWE","WindGustDirNS"]

#scatter_matrix(X_train[attributes], figsize=(12,8))
# separate X and y

y_train = X_train['RainTomorrow']

X_train = X_train.drop('RainTomorrow', axis=1)
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")

imputer.fit(X_train)

X = imputer.transform(X_train)

X_train2 = pd.DataFrame(X, columns=X_train.columns)

X_train2.head(5)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train_std=sc.fit_transform(X_train2)

X_train2 = pd.DataFrame(X_train_std, columns=X_train2.columns)

X_train2.head(5)

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score





def alterDataSet(df):

    removeUnusedVariable(df)

    processDate(df)

    df = changeAllColumnWinDir(df)

    df["RainToday"] = df["RainToday"].map({'No': 0, 'Yes': 1})

    X_bis = imputer.transform(df)

    X2 = pd.DataFrame(X_bis, columns=df.columns)



    X_std=sc.transform(X2)

    X2 = pd.DataFrame(X_std, columns=df.columns)

    

    return X2



# separate X and y

X_test = test_set.drop('RainTomorrow', axis=1)

y_test = test_set['RainTomorrow']

y_test = y_test.map({'No': 0, 'Yes': 1})

# apply change on X_test

X_test2 = alterDataSet(X_test)

X_test2.shape

from sklearn.linear_model import SGDClassifier

model = SGDClassifier(random_state=42)

model.fit(X_train2, y_train)



res = model.predict(X_test2)
#

#confusion_matrix(y_test, res)

#

print(f1_score(y_test, res))

print(accuracy_score(y_test, res))





# Sans les mois : f1=0.5604687356953219   accuracy= 0.8311825310313302

# Avec les mois : f1=0.5570107152669658   accuracy= 0.8299166637364183

# Avec les mois (distance par rapport à mi-année): f1=0.5735630163473369  accuracy= 0.829389219030205

# ==> Ne pas utiliser la date comme data

from sklearn.tree import DecisionTreeClassifier



model_dt = DecisionTreeClassifier(random_state=42)

model_dt.fit(X_train2,y_train)

y_pred = model_dt.predict(X_test2)

score = accuracy_score(y_test,y_pred)

print(score)
#result = sklearn.tree.export_graphviz(model_dt, out_file=None)                
print(os.listdir("..")) #tree.dot

#from graphviz import Digraph

print(model_dt)

#import graphviz



#with open("tree_1.dot") as f:

    #dot_graph = f.read()



# remove the display(...)



#graphviz.Source(result)


