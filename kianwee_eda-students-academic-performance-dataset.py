# Importing required packages 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set() # Making sns as default for plots 
# Importing dataset into notebook

data = pd.read_csv('../input/xAPI-Edu-Data/xAPI-Edu-Data.csv')



# Viewing dataset

print(data.head())



# Viewing shape

print(data.shape)
data.isnull().sum()
data.describe(include = 'all')
g = sns.pairplot(data,hue = 'Class', hue_order= ['H', 'M', 'L'])

g.map_diag(sns.distplot)

g.add_legend()

g.fig.suptitle('FacetGrid plot', fontsize = 20)

g.fig.subplots_adjust(top= 0.9);
plt.figure(figsize=(10,5))

sns.countplot(data = data, x = 'gender', hue = 'Class', hue_order= ['H', 'M', 'L']).set_title("Graph showing number of males and females among different classes", fontsize = 20)
plt.figure(figsize=(15,5))



sns.countplot(data = data, x ='NationalITy', hue = 'Class', hue_order= ['H', 'M', 'L']).set_title("Graph showing number of countries among different classes", fontsize = 20)
plt.figure(figsize=(15,5))



sns.countplot(data = data, x ='StageID', hue = 'Class', hue_order= ['H', 'M', 'L']).set_title("Graph showing number of stages among different classes", fontsize = 20)
plt.figure(figsize=(15,5))



sns.countplot(data = data, x ='Topic', hue = 'Class', hue_order= ['H', 'M', 'L']).set_title("Graph showing number of topics among different classes", fontsize = 20)
lst = ['Topic','Class', 'raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']

data1 = data[lst]

data1.groupby('Topic').median()
g = sns.catplot(x="Class", y="raisedhands",col_wrap=3, col="Topic", order = ['H', 'M', 'L'],

                data= data, kind="box",

                height=5, aspect=0.8);
plt.figure(figsize=(10,5))





sns.countplot(data = data, x = 'StudentAbsenceDays', hue = 'Class', hue_order = ['H', 'M', 'L']).set_title("Graph showing student absence days among different classes")
plt.figure(figsize=(10,5))



sns.countplot(data = data, x = 'Relation', hue = 'Class', hue_order = ['H', 'M', 'L']).set_title("Graph showing which parent produce better students", fontsize = 15)