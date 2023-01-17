import numpy as np

import pandas as pd 

import re #library to clean data

import nltk #Natural Language tool kit

import matplotlib.pyplot as plt

import seaborn as sns 

import os 

import datetime

from nltk.corpus import stopwords #to remove stopword

from nltk.stem.porter import PorterStemmer 

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
os.listdir('../input/google-play-store-apps')
data=pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")

data.head()
data.isnull().sum().sort_values(ascending=False)
data_review=pd.read_csv("../input/google-play-store-apps/googleplaystore_user_reviews.csv")

data_review.head()
data_review.isnull().sum().sort_values(ascending=False)
print('data size: ',len(data))

print('data_review size: ',len(data_review))
data=data.dropna(subset=['Rating','Current Ver','Android Ver','Content Rating'])
fig,ax=plt.subplots(2,2,figsize=(25,16))

sns.barplot(x=data.Category.value_counts(),y=data.Category.value_counts().index,ax=ax[0,0])

ax[0,0].set_title("Counts of Category",size=20)

ax[0,0].set_xlabel("")



data.Reviews=data.Reviews.astype('int')

sns.barplot(x=data.groupby(['Category'])['Reviews'].agg('sum').sort_values(ascending=False),y=data.groupby(['Category'])['Reviews'].agg('sum').sort_values(ascending=False).index,ax=ax[0,1])

ax[0,1].set_title("Number of reviews by Category",size=20)

ax[0,1].set_ylabel("")



data['new_install']=data.Installs.apply(lambda x:x.split('+')[0].strip(',').replace(',',''))

data.new_install=data.new_install.astype('int')



sns.barplot(x=data.groupby(['Category'])['new_install'].agg('sum').sort_values(ascending=False),y=data.groupby(['Category'])['new_install'].agg('sum').sort_values(ascending=False).index,ax=ax[1,0])

ax[1,0].set_title("Number of installs by Category",size=20)

ax[1,0].set_ylabel("")

ax[1,0].set_xlabel("")



sns.boxplot(y="Category",x="Rating",data=data,ax=ax[1,1])

ax[1,1].set_ylabel("")

ax[1,1].set_title("Distribution of rating by Category",size=20)
plt.figure(figsize=(8,8))

corr = data.corr()

sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns,annot=True)

plt.title("correlation plot",size=28)
fig,ax=plt.subplots(1,2,figsize=(25,16))

sns.barplot(x=data.Size.value_counts()[:10],y=data.Size.value_counts()[:10].index,ax=ax[0])

ax[0].set_title("Top 10 Size by counts",size=20)

ax[0].set_xlabel("")



sns.boxplot(y="Size",x="Rating",data=data[data.Size.isin(list(data.Size.value_counts()[:10].index))],ax=ax[1])

ax[1].set_ylabel("")

ax[1].set_title("Distribution of rating by Size for Top 10",size=20)
data['new_Price']=data.Price.apply(lambda x:  x.strip('$') if x!='0' else x.strip(''))

data.new_Price=data.new_Price.astype(float)
fig,ax=plt.subplots(1,2,figsize=(25,16))

sns.barplot(x=data['Content Rating'].value_counts(),y=data['Content Rating'].value_counts().index,ax=ax[0])

ax[0].set_title("Counts of Content Rating",size=20)

ax[0].set_xlabel("")



sns.barplot(x=data.groupby(['Content Rating'])['new_Price'].agg('sum'),y=data.groupby(['Content Rating'])['new_Price'].agg('sum').index,ax=ax[1])

ax[1].set_title("Total Price by Content Rating",size=20)

ax[1].set_ylabel("")

ax[1].set_xlabel("Total Price")

data_review=data_review.dropna(subset=['Translated_Review'])

data_review=data_review.reindex(range(len(data_review)), method='ffill')

headline_text_new=[]#Initialize empty array to append clean text

for i in range(len(data_review)):

    headline=re.sub('[^a-zA-Z]',' ',data_review['Translated_Review'][i]) 

    headline=headline.lower() #convert to lower case

    headline=headline.split() #split to array(default delimiter is " ")

    ps=PorterStemmer() #creating porterStemmer object to take main stem of each word

    headline=[ps.stem(word) for word in headline if not word in set(stopwords.words('english'))] #loop for stemming each word  in string array at ith row

    headline_text_new.extend(headline)
wordcloud = WordCloud(background_color="black",max_words=200,max_font_size=40,random_state=10).generate(str(headline_text_new))



plt.figure(figsize=(20,15))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()