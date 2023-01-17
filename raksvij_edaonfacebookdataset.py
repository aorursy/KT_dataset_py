import os

print(os.listdir("../input"))
import numpy as np

import pandas as pd





dataset = pd.read_csv("../input/pseudo_facebook.tsv",sep="\t")

dataset.head()
dataset.shape #to check the no. of rows and columns present in a dataset
#!pip install pandas-profiling
import pandas_profiling as pp

pp.ProfileReport(dataset)
data_age1 = dataset.groupby('age').mean()

data_age1.head()
from scipy import stats, integrate

import matplotlib.pyplot as plt



import seaborn as sns



%matplotlib inline
data_age1.reset_index(inplace=True)

data_age1.head()
dataset['age'].value_counts().head()
sns.boxplot(data_age1.age) # BoxPlotOfAge
plt.bar(data_age1.age,data_age1.tenure)

plt.xlabel("Age")

plt.ylabel("Tenure")
data_age1[data_age1['tenure']==data_age1['tenure'].max()]

#Displays the row with maximum Tenure value

#“data_age1” – data frame obtained by grouping the mean of “age” values
datacount = dataset.groupby('age').count()

datacount = datacount.reset_index()

data1000 = datacount[datacount['tenure']>=1000] #random column taken to bifercate the age groups who has more than 1000 facebook users 

data1000.loc[:, ['age','tenure']].head()
dataset_1k = data_age1.loc[data_age1['age'].isin(data1000['age'])]

dataset_1k[dataset_1k['tenure']==dataset_1k['tenure'].max()]
data2000 = datacount[datacount['tenure']>=2000]

dataset_2k = data_age1.loc[data_age1['age'].isin(data2000['age'])]

dataset_2k[dataset_2k['tenure']==dataset_2k['tenure'].max()]
data_age1[data_age1['friend_count']==data_age1['friend_count'].max()]
dataset_2k[dataset_2k['friend_count']==dataset_2k['friend_count'].max()]
dataset_1k[dataset_1k['friend_count']==dataset_1k['friend_count'].max()]
dataset_18 = dataset[dataset['age']==18]

dataset_18_M = dataset_18[dataset_18['gender']=='male']

dataset_18_F = dataset_18[dataset_18['gender']=='female']

print(dataset_18_F.shape)

print(dataset_18_M.shape)
dataset_gender_male = dataset[dataset['gender']=='male']

dataset_gender_female = dataset[dataset['gender']=='female']
dataset_gender_male.shape
dataset_gender_female.shape
data_age1['mobile_surfing'] = data_age1.mobile_likes+data_age1.mobile_likes_received

data_age1['web_surfing'] = data_age1.www_likes+data_age1.www_likes_received

data_age1.head()
plt.figure(figsize=(10,7))

plt.plot("age","mobile_surfing",'bv--',data=data_age1)

plt.plot("age","web_surfing",'r*-',data=data_age1)

plt.xlabel('Age')

plt.ylabel('Mobile and Web Surfing')

plt.title('Age Vs Surfing')

plt.legend()
data_age1[data_age1.mobile_surfing>data_age1.web_surfing].shape
data_age1[data_age1['mobile_surfing']==data_age1['mobile_surfing'].max()]
dataset_2k['mobile_surfing'] = dataset_2k.mobile_likes+dataset_2k.mobile_likes_received

dataset_2k['web_surfing'] = dataset_2k.www_likes+dataset_2k.www_likes_received
dataset_2k[dataset_2k['mobile_surfing']==dataset_2k['mobile_surfing'].max()]
dataset_2k[dataset_2k['mobile_likes']==dataset_2k['mobile_likes'].max()]
print(dataset_gender_male.mobile_likes.sum())

print(dataset_gender_male.www_likes.sum())

print(dataset_gender_female.mobile_likes.sum())

print(dataset_gender_female.www_likes.sum())
print(dataset_gender_male.likes_received.sum())

print(dataset_gender_female.likes_received.sum())