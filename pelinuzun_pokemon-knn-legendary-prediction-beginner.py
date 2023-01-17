# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import ipywidgets

import altair as alt

alt.renderers.enable('notebook')

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#load data

data = pd.read_csv("../input/pokemon.csv", skipinitialspace=True)
#print out top 5 data from dataframe

data.head()
#check info about the dataframe

data.info()

#check NaN values

data.isnull().sum()
#we can drop NaN values from our dataframe

data = data.dropna()
data.isnull().sum()
#correlation between features

f, ax = plt.subplots(figsize = (18, 18))

sns.heatmap(data.corr(), annot = True, linewidth = .5, fmt = ".1f", ax = ax)

plt.show() 
data1 = data.loc[:, ["Attack", "Speed", "Defense", "HP"]]

data1.plot()

plt.show()
data1.plot(subplots = True)

plt.show()
#we can also use scatter graph

data.plot(kind = "scatter", x = "Attack", y = "Defense", color = "b", linewidth = 1)

plt.xlabel("Attack")

plt.ylabel("Defense")

plt.show()
#we can plot histogram graph

data.Attack.plot(kind = "hist", bins = 50, figsize = (15, 15))

plt.xlabel("Attack")

plt.show()
#we can see count, mean, std, min, max values of numeric data

data.describe()
#we can also plot outlier 

data.boxplot(column = "Attack", by = "Legendary")

plt.show()
@ipywidgets.interact

def plot(color = ['red', 'steelblue', 'green', 'blue']):

    (sns.barplot(y = 'Name', x = 'Attack', data = data.head(10), orient = 'h', color = color))
#we can also create pivot table

pd.pivot_table(data,index=["Name"])

#we have to remove Name

data = data.drop('Name', axis = 1)
#we have to give numerical values to Legendary column

data.iloc[:, 5:6] = data.iloc[:, 5:6].apply(LabelEncoder().fit_transform)

from collections import Counter

count  = pd.Series(data['Type 1'].str.replace('[\[\]\']','').str.split(',').map(Counter).sum())

ln_list = range(0, len(count))

categorize = list(ln_list)

type(categorize[0])

#we also have to categorize Type 1 and Type 2

#data['Type 1'] = pd.Categorical(data['Type 1'], categories= data['Type 1'].unique()).codes

        
data.iloc[:, 1:3] = data.iloc[:, 1:3].apply(LabelEncoder().fit_transform)

data.head()

data = data.drop('#', axis = 1)
#assign target value

target = data['Legendary']

#drop target value from the dataframe

data = data.drop('Legendary', axis = 1)

#create KNN object

knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(data, target)

KNeighborsClassifier(algorithm = 'auto', leaf_size = 30, 

                    metric = "minkowski", metric_params = None, n_jobs = None, 

                    n_neighbors = 5, p = 2, weights = 'uniform')
poke = [157,133,30,70,120,100,135,95, 0]

poke = np.array(poke).reshape(1, -1)

poke
prediction = knn.predict(poke)

result = 'Legendary' if prediction == True else 'Not Legendary'

print(result)
