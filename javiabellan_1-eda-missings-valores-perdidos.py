import numpy   as np 

import pandas  as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import missingno as msno



%matplotlib inline
df = pd.read_csv("../input/murcia-beer-challenge/beer_train.csv")

df.head()
missings   = df.isnull().sum().sort_values(ascending=False)

percentage = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending=False)

pd.concat([missings, percentage], axis=1, keys=['Missings', 'Percentage'])
msno.bar(df);
msno.matrix(df);
msno.heatmap(df);
msno.dendrogram(df);