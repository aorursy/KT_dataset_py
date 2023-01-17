import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import re

import nltk

import nltk as nlp



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import warnings

warnings.filterwarnings("ignore")
df=pd.read_csv("../input/all-space-missions-from-1957/Space_Corrected.csv")
df.head()
df.tail()
df.drop(["Unnamed: 0","Unnamed: 0.1"],axis=1,inplace=True)
df.head()
df.columns
df=df.rename(columns={"Company Name":"company_name","Location":"location","Datum":"datum","Detail":"detail","Status Rocket":"status_rocket"," Rocket":"rocket","Status Mission":"status_mission"})
df.head()
df.info()
df.isna().sum()
df.drop(["rocket"],axis=1,inplace=True)
df.isna().sum()
df['datum'] = pd.to_datetime(df['datum'])
df.head()
df.status_mission.unique()
df.status_rocket.unique()
df.head()
print(df['company_name'].value_counts(dropna=False))
df.status_mission.dropna(inplace = True)

labels = df.status_mission.value_counts().index

colors = ['green','red',"blue","black"]

explode = [0.1,0,0,0.5]

sizes = df.status_mission.value_counts().values



# visual cp

plt.figure(0,figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('status_mission According to success Type',color = 'blue',fontsize = 15)
df.status_rocket.dropna(inplace = True)

labels = df.status_rocket.value_counts().index

colors = ['red','green']

explode = [0,0.2]

sizes = df.status_rocket.value_counts().values



# visual cp

plt.figure(0,figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('status_rocket According to status',color = 'blue',fontsize = 15)
detail_list=[]

for detail in df.detail:

    detail=re.sub("[^a-zA-Z]"," ",detail)

    detail=detail.lower()

    detail=nltk.word_tokenize(detail)

    lemma  = nlp.WordNetLemmatizer()

    detail=[lemma.lemmatize(word) for word in detail]

    detail=" ".join(detail)

    detail_list.append(detail)
df.detail.value_counts
#bag of words

from sklearn.feature_extraction.text import CountVectorizer

max_features =50

count_vectorizer =CountVectorizer(max_features=max_features,stop_words="english")

sparce_matrix = count_vectorizer.fit_transform(detail_list).toarray()

print("The 50 most commonly used {} words: {} ".format(max_features,count_vectorizer.get_feature_names()))
#from sklearn.preprocessing import LabelEncoder

#labelEncoder_Y=LabelEncoder()

#df.iloc[:,4]=labelEncoder_Y.fit_transform(df.iloc[:,4].values)
#from sklearn.preprocessing import LabelEncoder

#labelEncoder_Y=LabelEncoder()

#df.iloc[:,5]=labelEncoder_Y.fit_transform(df.iloc[:,5].values)
plt.figure(figsize=(20,4))

sns.barplot(x=df['company_name'].value_counts().index,y=df['company_name'].value_counts().values)

plt.title('company_name other rate')

plt.ylabel('Rates')

plt.legend(loc=0)

plt.xticks(rotation=75)

plt.show()
df.columns
location = df["location"]

df["location"] = [i.split(".")[0].split(",")[-1].strip() for i in location]
df.head()

plt.figure(figsize=(25,4))

sns.barplot(x=df['location'].value_counts().index,y=df['location'].value_counts().values)

plt.title('location other rate')

plt.ylabel('Rates')

plt.legend(loc=0)

plt.xticks(rotation=60)

plt.show()
det = df["detail"]

df["detail"] = [i.split(" | ")[0].split(",")[-1].strip() for i in det]
plt.figure(figsize=(25,4))

sns.barplot(x=df['detail'][:50].value_counts().index,y=df['detail'][:50].value_counts().values)

plt.title('detail other rate')

plt.ylabel('Rates')

plt.legend(loc=0)

plt.xticks(rotation=90)

plt.show()