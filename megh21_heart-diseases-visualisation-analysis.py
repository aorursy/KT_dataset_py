import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

sns.set_style('darkgrid')

os.getcwd()
df1 = pd.read_csv('../input/heart.csv')
#########Size of the dataset#######

df1.shape
###########Lets take a look at the first 5 rows of our data########

df1.head()
#########Information About the Dataset#########

df1.info()

#########We can see that except oldpeak all the values are in int64###########
#########Stats about the dataset########

df1.describe()
##########Lets take a look at how the data is correlated#########

plt.figure(figsize=(10,8))

viz1 = sns.heatmap(df1.corr(), annot = True , cmap = 'Greens',linewidths=2)

plt.show()

#####Not much of correlation in the data######
######Since we have age in the dataset, Let us take a look at age distribution##########

viz2 = sns.distplot(df1.age , color = 'green' , )

#########We can see that most of the people are in the age group of 40 to 60#########
##############Lets start digging into the data#############
##########Lets pair age with heart diseases#############

#########Lets convert age into bins#############

age_bins = [20,30,40,50,60,70,80]

df2 = pd.cut(df1.age , bins = age_bins)

df1['bin_age'] = df2

df1.head()

viz3=sns.countplot(x='bin_age',data=df1 ,palette='husl',linewidth=3 , hue = 'target')

viz3.set_title("Patients Based on Age Groups")

plt.show(viz3)

#########The graph indicates most of the heart patients are in the age groups of 40's and 50's########
##########Let us take a look at how other parameters in the dataset look when paired with heart diseases#######

viz4=sns.countplot(x= 'cp',data=df1 ,palette='husl',linewidth=3 , hue = 'target')

viz4.set_title("Patients Based on Chest Pain Type")

plt.show(viz4)

##It is Obvious from the graph that most of the heart patients get chest pain type 2##
##############Cholestrol and Heart Diseases############

plt.figure(figsize=(7,6))

chol_bins = [100,150,200,250,300,350,400,450]

df3 = pd.cut(df1.chol , bins = chol_bins)

df1['bins_chol'] = df3

viz5=sns.countplot(x= 'bins_chol',data=df1 ,palette='husl',linewidth=3 , hue = 'target')

viz5.set_title("Patients Based on Cholestrol Serum Levels in Blood")

plt.show(viz5)
#########Patients Based on Fasting Blood Sugar Levels#########

viz5=sns.countplot(x= 'fbs',data=df1 ,palette='husl',linewidth=3 , hue = 'target')

viz5.set_title("Patients Based on Fasting Blood Sugar Levels")

plt.show(viz4)

##The graph points out that when fsb levels are above 120mg/dl its less likely to have a heart diesase
#########Patients Based on Thal Levels#########

viz6=sns.countplot(x= 'thal',data=df1 ,palette='husl',linewidth=3 , hue = 'target')

viz6.set_title("Patients Based on Thal Levels")

plt.show(viz6)
#########Avg Age of Heart Patients Based on Gender#######

plt.figure(figsize=(10,8))

viz7 = sns.boxenplot(x='sex', y= 'age', data=df1, hue = 'target', palette='husl')
plt.figure(figsize=(10,7))

viz8 = sns.pointplot(x='sex', y= 'age', data=df1, hue = 'target', palette='husl', capsize = 0.1)
plt.figure(figsize=(10,7))

viz9 = sns.lineplot(data = df1 , x = 'age', y='cp',hue='target' , ci = None , palette='husl')
viz10 = sns.countplot(data = df1 , x = 'slope' , hue = 'target', palette='husl' )
viz11 = sns.lmplot(data = df1 , x= 'age', y = 'oldpeak', ci=None,fit_reg=False,size=8,hue='target',aspect=2, palette='husl')