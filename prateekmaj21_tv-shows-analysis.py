#Importing libraries



import re

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import random

from wordcloud import WordCloud, STOPWORDS

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#reading the data

data=pd.read_csv("/kaggle/input/tv-shows-on-netflix-prime-video-hulu-and-disney/tv_shows.csv")
data.head()
#looking at the data



data.info()
#Converting the percentages to number



data['Rotten Tomatoes'] = data['Rotten Tomatoes'].str.rstrip('%').astype('float')
#Removing the "+" sign from age rating



data["Age"] = data["Age"].str.replace("+","")
#Conveting it to numeric 



data['Age'] = pd.to_numeric(data['Age'],errors='coerce')
#Final data



data.head()
#Data info



data.info()
#only the data will complete column values available

#later use



df=data.dropna()
#Taking the values



titles=data["Title"].values
#Joining into a single string



text=' '.join(titles)
len(text)
#How it looks



text[1000:1500]
#Removing the punctuation



text = re.sub(r'[^\w\s]','',text)
len(text)
#Punctuation has been removed



text[1000:1500]
#Creating the tokenizer

tokenizer = nltk.tokenize.RegexpTokenizer('\w+')
#Tokenizing the text

tokens = tokenizer.tokenize(text)
len(tokens)
#Now the words have been converted to tokens



tokens[1000:1010]
#now we shall make everything lowercase for uniformity

#to hold the new lower case words



words = []



# Looping through the tokens and make them lower case

for word in tokens:

    words.append(word.lower())
#Stop words are generally the most common words in a language.

#English stop words from nltk.



stopwords = nltk.corpus.stopwords.words('english')
words_new = []



#Now we need to remove the stop words from the words variable

#Appending to words_new all words that are in words but not in sw



for word in words:

    if word not in stopwords:

        words_new.append(word)
#The frequency distribution of the words



freq_dist = nltk.FreqDist(words_new)
#Frequency Distribution Plot

plt.subplots(figsize=(20,12))

freq_dist.plot(50)
#converting into string



res=' '.join([i for i in words_new if not i.isdigit()])
#wordcloud



plt.subplots(figsize=(16,10))

wordcloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          max_words=100,

                          width=1400,

                          height=1200

                         ).generate(res)





plt.imshow(wordcloud)

plt.title('TV Show Title WordCloud 100 Words')

plt.axis('off')

plt.show()
#wordcloud



plt.subplots(figsize=(16,10))

wordcloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          max_words=500,

                          width=1400,

                          height=1200

                         ).generate(res)





plt.imshow(wordcloud)

plt.title('TV Show Title WordCloud 500 Words')

plt.axis('off')

plt.show()
#Lets compare the both
#Raw text



len(text)
#Text from tokenized words



len(res)


plt.subplots(figsize=(16,10))

wordcloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          max_words=100,

                          width=1400,

                          height=1200

                         ).generate(text)





plt.imshow(wordcloud)

plt.title('TV Show Title WordCloud 100 Words')

plt.axis('off')

plt.show()
text[1000:1500]


plt.subplots(figsize=(16,10))

wordcloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          max_words=500,

                          width=1400,

                          height=1200

                         ).generate(text)





plt.imshow(wordcloud)

plt.title('TV Show Title WordCloud 500 Words')

plt.axis('off')

plt.show()
data.head()
data.info()
#overall year of release analysis



plt.subplots(figsize=(8,6))

sns.distplot(data["Year"],kde=False, color="blue")
#overall year of release analysis



plt.subplots(figsize=(8,6))

sns.distplot(data["Age"],kde=False, color="blue")
print("TV Shows with highest IMDb ratings are= ")

print((data.sort_values("IMDb",ascending=False).head(20))['Title'])
#barplot of rating

plt.subplots(figsize=(8,6))

sns.barplot(x="IMDb", y="Title" , data= data.sort_values("IMDb",ascending=False).head(20))
print("TV Shows with lowest IMDb ratings are= ")

print((data.sort_values("IMDb",ascending=True).head(20))['Title'])
#barplot of rating

plt.subplots(figsize=(8,6))

sns.barplot(x="IMDb", y="Title" , data= data.sort_values("IMDb",ascending=True).head(20))
#Overall data of IMDb ratings



plt.figure(figsize=(16, 6))



sns.scatterplot(data=data['IMDb'])

plt.ylabel("Rating")

plt.xlabel('Movies')

plt.title("IMDb Rating Distribution")
print("TV Shows with highest Rotten Tomatoes scores are= ")

print((data.sort_values("Rotten Tomatoes",ascending=False).head(20))['Title'])
#barplot of rating

plt.subplots(figsize=(8,6))

sns.barplot(x="Rotten Tomatoes", y="Title" , data= data.sort_values("Rotten Tomatoes",ascending=False).head(20))
print("TV Shows with lowest Rotten Tomatoes scores are= ")

print((data.sort_values("Rotten Tomatoes",ascending=True).head(20))['Title'])
#barplot of rating

plt.subplots(figsize=(8,6))

sns.barplot(x="Rotten Tomatoes", y="Title" , data= data.sort_values("Rotten Tomatoes",ascending=True).head(20))
#Overall data of Rotten Tomatoes scores



plt.figure(figsize=(16, 6))

sns.scatterplot(data=data['Rotten Tomatoes'])

plt.ylabel("Rotten Tomatoes score")

plt.xlabel('Movies')

plt.title("Rotten Tomatoes Score Distribution")
#selecting netflix shows

netflix=data[data["Netflix"]==1]
print("Number of shows on Netflix= ", len(netflix))
plt.subplots(figsize=(8,6))

sns.distplot(netflix["Year"],kde=False, color="blue")
plt.subplots(figsize=(8,6))

sns.distplot(netflix["Age"],kde=False, color="blue")
plt.subplots(figsize=(8,6))

sns.distplot(netflix["IMDb"],kde=False, color="blue")
plt.subplots(figsize=(8,6))

sns.distplot(netflix["Rotten Tomatoes"],kde=False, color="blue")
print("Netflix Shows with highest IMDb ratings are= ")

print((netflix.sort_values("IMDb",ascending=False).head(10))['Title'])
print("Netflix Shows with lowest IMDb ratings are= ")

print((netflix.sort_values("IMDb",ascending=True).head(10))['Title'])
print("Netflix Shows with highest Rotten Tomatoes score are= ")

print((netflix.sort_values("Rotten Tomatoes",ascending=False).head(10))['Title'])
print("Netflix Shows with lowest Rotten Tomatoes score are= ")

print((netflix.sort_values("Rotten Tomatoes",ascending=True).head(10))['Title'])
#Taking the title and rating data



netflix1=netflix.sort_values("IMDb",ascending=False).head(100)[['Title',"IMDb"]]

netflix1.head()
#Converting it into a tuple



tuples_netflix_imdb = [tuple(x) for x in netflix1.values]
#Looks like this



tuples_netflix_imdb[0:10]
#Making a wordcloud



wordcloud_netflix_imdb = WordCloud(width=1400,height=1200).generate_from_frequencies(dict(tuples_netflix_imdb))
plt.subplots(figsize=(12,12))

plt.imshow(wordcloud_netflix_imdb)

plt.title("TV Shows based on IMDb rating(Top 100)")
#Taking the title value and Rotten Tomatoes Score



netflix2=netflix.sort_values("Rotten Tomatoes",ascending=False).head(100)[['Title',"Rotten Tomatoes"]]

netflix2.head()
#Converting to Tuple



tuples_netflix_tomatoes = [tuple(x) for x in netflix2.values]
#Word Cloud generation



wordcloud_netflix_tomatoes = WordCloud(width=1400,height=1200).generate_from_frequencies(dict(tuples_netflix_tomatoes))
plt.subplots(figsize=(12,12))

plt.imshow(wordcloud_netflix_tomatoes)



plt.title("TV Shows based on Rotten Tomatoes Score(Top 100)")
#Taking the relevant data



ratings=data[["Title",'IMDb',"Rotten Tomatoes"]]

ratings.head()
len(ratings)
ratings.info()
#Removing the data



ratings=ratings.dropna()
ratings["IMDb"]=ratings["IMDb"]*10
#New data



ratings.head()
#Input data



X=ratings[["IMDb","Rotten Tomatoes"]]
X.head()
#Scatterplot of the input data



plt.figure(figsize=(10,6))

sns.scatterplot(x = 'IMDb',y = 'Rotten Tomatoes',  data = X  ,s = 60 )

plt.xlabel('IMDb rating (multiplied by 10)')

plt.ylabel('Rotten Tomatoes') 

plt.title('IMDb rating (multiplied by 10) vs Rotten Tomatoes Score')

plt.show()
#Importing KMeans from sklearn



from sklearn.cluster import KMeans
wcss=[]



for i in range(1,11):

    km=KMeans(n_clusters=i)

    km.fit(X)

    wcss.append(km.inertia_)
#The elbow curve



plt.figure(figsize=(12,6))



plt.plot(range(1,11),wcss)



plt.plot(range(1,11),wcss, linewidth=2, color="red", marker ="8")



plt.xlabel("K Value")

plt.xticks(np.arange(1,11,1))

plt.ylabel("WCSS")



plt.show()
#this is known as the elbow graph , the x axis being the number of clusters

#the number of clusters is taken at the elbow joint point

#this point is the point where making clusters is most relevant

#the numbers of clusters is kept at maximum
#Taking 4 clusters



km=KMeans(n_clusters=4)
#Fitting the input data



km.fit(X)
#predicting the labels of the input data



y=km.predict(X)
#adding the labels to a column named label



ratings["label"] = y
#The new dataframe with the clustering done



ratings.head()
#Scatterplot of the clusters



plt.figure(figsize=(10,6))

sns.scatterplot(x = 'IMDb',y = 'Rotten Tomatoes',hue="label",  

                 palette=['green','orange','red',"blue"], legend='full',data = ratings  ,s = 60 )



plt.xlabel('IMDb rating(Multiplied by 10)')

plt.ylabel('Rotten Tomatoes score') 

plt.title('IMDb rating(Multiplied by 10) vs Rotten Tomatoes score')

plt.show()
print('Number of Cluster 0 TV Shows are=')

print(len(ratings[ratings["label"]==0]))

print("--------------------------------------------")

print('Number of Cluster 1 TV Shows are=')

print(len(ratings[ratings["label"]==1]))

print("--------------------------------------------")

print('Number of Cluster 2 TV Shows are=')

print(len(ratings[ratings["label"]==2]))

print("--------------------------------------------")

print('Number of Cluster 3 TV Shows are=')

print(len(ratings[ratings["label"]==3]))

print("--------------------------------------------")
print('TV Shows in cluster 0')



print(ratings[ratings["label"]==0]["Title"].values)
print('TV Shows in cluster 1')



print(ratings[ratings["label"]==1]["Title"].values)
print('TV Shows in cluster 2')



print(ratings[ratings["label"]==2]["Title"].values)
print('TV Shows in cluster 3')



print(ratings[ratings["label"]==3]["Title"].values)