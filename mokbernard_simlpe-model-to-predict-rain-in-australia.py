import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Import the required packages

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import preprocessing, model_selection , svm

from sklearn.linear_model import LinearRegression
#Lets read the data

df = pd.read_csv('../input/weatherAUS.csv', index_col=0)
#Start exploring the data

df_1 = df.copy()

df_1.head()
#Seems that a number of data are categorical

df_1.info()
#And there are a good number of null value

df_1.isnull().sum()
#Lets handle the NA in a simple method by filling them as NA

df_1 = df_1.fillna(0)
#Seems that our label is in text format

df_1['RainTomorrow'].unique()
#Lets transform them into numbers

TrueFalse = {'Yes': 1,'No': 0, 0:-99999} 

df_1['RainToday'] = [TrueFalse[item] for item in df_1['RainToday']]

df_1['RainTomorrow'] = [TrueFalse[item] for item in df_1['RainTomorrow']]
#The wind gust directions will need a similar treatment

#Create a list with all kinds of wind direction

dir_1 = list(df_1['WindGustDir'].unique())

dir_2 = list(df_1['WindDir9am'].unique())

dir_3 = list(df_1['WindDir3pm'].unique())

all_dir = dir_1 + dir_2 + dir_3

all_dir_dedup = list(set(all_dir))



#Create a dictionary with these wind direction to numbers for further mapping

dir_dict = {}

for num, winddir in enumerate(all_dir_dedup):

    dir_dict[winddir] = num



#Map these wind direction number

df_1['WindGustDir'] = df_1['WindGustDir'].map(dir_dict)

df_1['WindDir9am'] = df_1['WindDir9am'].map(dir_dict)

df_1['WindDir3pm'] = df_1['WindDir3pm'].map(dir_dict)
df_1.fillna(value = -99999, inplace=True)
#We would like to conduct some analysis on the variables

#lets drop the location and create a heat map

df_1_noloc = df_1.drop(columns = ['Location'])

df_1_noloc = df_1_noloc[df_1_noloc.columns].astype(float)
#Lets create a heatmap with seaborn

#Seems that humidity and 

dimension = (20,10)

fig, ax = plt.subplots(figsize = dimension)

sns.set(style='white')

sns.heatmap(df_1_noloc.reset_index(drop=True).corr(), annot=True)
#Lets also look at the distribution of data with histogram

fig, ax= plt.subplots(figsize=(20,12))

hist = df_1_noloc.hist(ax = ax)

#Now we will create some very simple machine learning models

x = np.array(df_1_noloc.drop(['RainTomorrow'], 1))

y = np.array(df_1_noloc['RainTomorrow'])

x = preprocessing.scale(x)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)
#Linear Regression Model

clf=LinearRegression()

clf.fit(x_train, y_train)

confidence = clf.score(x_test, y_test)

print(confidence)
#SVC

clf=svm.SVC()

clf.fit(x_train, y_train)

confidence = clf.score(x_test, y_test)

print(confidence)
#Here it seems that SVC is a better model than linear regression to predict the rainfall in Australia