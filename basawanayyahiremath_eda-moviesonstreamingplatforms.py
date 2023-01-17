import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
im= pd.read_csv('/kaggle/input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv')
im.head()
im.shape
im.describe()
#data.describe() function will only display numarical data
#here we can see counts, max value etc and we can see the missing values also in count section 
im.info()
#data.info provides counts of all columns and also display type of data
sns.heatmap(im.isnull())
#visually we see the missing values using heatmap 
sns.countplot(x='Netflix', data=im)
im["Netflix"].value_counts()
# counts of values

sns.countplot(x='Hulu', data=im)
im["Hulu"].value_counts()
sns.countplot(x='Prime Video', data=im)
im["Prime Video"].value_counts()
sns.countplot(x='Disney+', data=im)
im["Disney+"].value_counts()
sns.countplot(x='Age', data=im)
sns.pairplot(im)
fig=plt.gcf()
fig.set_size_inches(20,20)
#as most of them are just categorical data so correlation is not there 
sns.scatterplot(x="Year", y="Runtime", hue="Age",data=im)
fig=plt.gcf()
fig.set_size_inches(10,10)
#we can see the outlier and the hue is Age. we can see that more films are made as time proceeds, this can be VISUALIZED using distribution graph

sns.distplot(im['Year'])
#by seeing the distribution graph we can say that most movies are made in period of 2000 to 2020 than 1940 to 2000

plt.figure(figsize=(15,7))
chains=im['Language'].value_counts()[:20]#change this value to see more result
sns.barplot(x=chains,y=chains.index,palette='Set2')
plt.title("Languages most commonly made ",size=20,pad=20)
plt.xlabel("Counts",size=15)
# English movies are made more in world and it followed by Hindi movies 
plt.figure(figsize=(15,7))
chains=im['Directors'].value_counts()[:20]
sns.barplot(x=chains,y=chains.index,palette='Set1')
plt.title("Most movies made by Director ",size=20,pad=20)
plt.xlabel("Counts",size=15)
plt.figure(figsize=(15,7))
chains=im['Genres'].value_counts()[:20]
sns.barplot(x=chains,y=chains.index,palette='Set2')
plt.title("Genres",size=20,pad=20)
plt.xlabel("Counts",size=15)
#it looks like people like more Drama movies compared to action movies
plt.figure(figsize=(15,7))
chains=im['Country'].value_counts()[:20]
sns.barplot(x=chains,y=chains.index,palette='Set3')
plt.title("Most movies made by Country",size=20,pad=20)
plt.xlabel("Counts",size=15)
#here we can see that USA makes more movies and follwed by India
sns.distplot(im['IMDb'], bins=20)
#this is the distribution graph of IMDb rating but we cant campare to Rotten Tomatoes because if see the heatmap
#there are so missing values, so we drop NaN values and compare.

#so about this distribution graph we can see avrage is about 6 rating
# Here Rotten Tomatoes is object value so we have to convert it to float. we do that by removing % symbol 
im_copy = im.copy(deep = True)
im_copy['Rotten Tomatoes'].unique()
im_copy= im_copy.loc[im_copy['Rotten Tomatoes'] !='NEW']
im_copy=im_copy.loc[im_copy['Rotten Tomatoes'] !='-'].reset_index(drop=True)
remove_slash = lambda x:x.replace('%','') if type(x)==np.str else x
im_copy['Rotten Tomatoes']=im_copy['Rotten Tomatoes'].apply(remove_slash).str.strip().astype('float')
im_copy['Rotten Tomatoes'].head()
im_copy.isnull().sum()
im_copy.dropna(how='any', inplace=True)
im_copy.shape
# initally 16744 rows, after dropna there are only 3301
sns.distplot(im_copy['IMDb'])
sns.distplot(im_copy['Rotten Tomatoes'])
# clearly we can see that distribution graph are not same that means rating differs for same movies