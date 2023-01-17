# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn import preprocessing
from scipy.sparse import hstack
import pandas_profiling
#load dataset
df = pd.read_csv("/kaggle/input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv")
df = df.iloc[:, 1:]    #remove unnamed index column
df.head()
df.info()
df.Type.unique()
df.isnull().sum()
miss = pd.DataFrame(df.isnull().sum())
miss = miss.rename(columns={0: "miss_count"})
miss["missing%"] = (miss.miss_count/len(df.ID))*100
miss
#mssing %>50%
df.drop(['Rotten Tomatoes', 'Age'], axis =1, inplace=True)
# dropping NA from following columns
df.dropna(subset=[
    'IMDb', 'Directors', 'Genres', 'Country', 'Language', 'Runtime'
], inplace = True)

df.reset_index(inplace=True, drop=True)

#converting into object tyype

df.Year = df.Year.astype("object")
df.info()
#check distribution of year using distplot
plt.figure(figsize = (20, 5))
sns.distplot(df['Year'])
plt.show()

#distribution of IMDB Rating
plt.figure(figsize = (20,5))
sns.distplot(df['IMDb'])
plt.show()

# lets plot the length of movies
sns.distplot(df['Runtime'])
plt.show()
def movie_count(platform, count=False):
    """A function to count the movies in differnet Streaming platform"""
    if count==False:
        print('Platform {} Count : {}'.format(platform, df[platform].sum()))
        
    else:
        return df[platform].sum()
#lets see the count
movie_count('Netflix')
movie_count('Prime Video')
movie_count('Disney+')
#lets find on each plaform
platform = 'Prime Video', 'Netflix', 'Hulu', 'Disney+'
s = [movie_count('Prime Video', count = True),
     movie_count('Netflix', count = True),
     movie_count('Hulu', count = True),
     movie_count('Disney+', count = True),
    ]

explode= (0.1, 0.1, 0.1, 0.1)

#plot
fig1, ax1 = plt.subplots()
ax1.pie(s, 
       labels = platform,
       autopct = '%1.1f%%',
       explode = explode,
       shadow = True,
       startangle=100)

ax1.axis = ('equal')
plt.title('Distribution of OTT')
plt.show()
# split genre by ',' and then stack it one after the other.
#apply will create the mupltiple columns for each genre and 'stack' will stack them
#in single column

g = df['Genres'].str.split(',').apply(pd.Series,1).stack()

g.index = g.index.droplevel(-1)

#assign name
g.name = 'Genres'

#delete column

del df['Genres']

#join new column
df_genres = df.join(g)


plt.figure(figsize=(20,10))
sns.countplot(x='Genres', data = df_genres)
plt.xticks(rotation=90)
plt.title("Genre wise movies")
plt.show()
#same process as above 
#split-->apply-->stack()

c = df['Country'].str.split(',').apply(pd.Series, 1).stack()
c.index = c.index.droplevel(-1)

#assign name
c.name = 'Country'

#delete column
del df['Country']

df_country = df.join(c)
df_country.head()

#plot
df_country['Country'].value_counts()[:10].plot(kind="bar", figsize=(15,5))
plt.show()
#split
l = df['Language'].str.split(',').apply(pd.Series,1).stack()

l.index = l.index.droplevel(-1)

l.name = "Language"

del df['Language']

df_language = df.join(l)
df_language['Language'].value_counts()[:10].plot(kind = 'bar', figsize =(15,3))
plt.show()
df.columns
#apply melt() function --> converts wide dataframe into a long dataframe
df_Imdb = pd.melt(df, id_vars = ['ID', 'Title', 'Year', 'IMDb', 'Type', 'Runtime'], var_name = 'platform')

df_Imdb = df_Imdb[df_Imdb.value==1]
df_Imdb.drop(columns=["value"], axis=1, inplace=True)

#plot the graph
g = sns.FacetGrid(df_Imdb, col="platform")
g.map(plt.hist, "IMDb")
plt.show()
# we have already droped the Age 
#lets reload the data
df = pd.read_csv("/kaggle/input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv")
df=df.iloc[:,1:]
df.ID = df.ID.astype("object")

# melting platform columns to create visualization
df2 = pd.melt(df, id_vars=["ID","Title","Year","Age","IMDb","Rotten Tomatoes","Type","Runtime"], var_name="platform")
df2 = df2[df2.value==1]
df2.drop(columns=["value"],axis=1,inplace=True)
df2.head(5)
# Total runtime in different platform

ax = sns.barplot(x="platform", y="Runtime",hue="Age", estimator=sum, data=df2)

#load data again
df = pd.read_csv("/kaggle/input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv")
df = df.iloc[:, 1:]
#find missing values
miss = pd.DataFrame(df.isnull().sum())
miss = miss.rename(columns={0: "miss_count"})
miss["missing%"] = (miss.miss_count/len(df.ID))*100
miss
#mssing %>50%
df.drop(['Rotten Tomatoes', 'Age'], axis =1, inplace=True)
# dropping NA from following columns
df.dropna(subset=[
    'IMDb', 'Directors', 'Genres', 'Country', 'Language', 'Runtime'
], inplace = True)

df.reset_index(inplace=True, drop=True)

#converting into object tyype

df.Year = df.Year.astype("object")
df.ID = df.ID.astype("object")
#select variable
numerical_df = df.select_dtypes(include=['float64', "int64"])
#use preprocessing

#create minmax scaler
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

#create dataframe
numerical_df_max = pd.DataFrame((scaler.fit_transform(numerical_df)))

#assign column names
numerical_df_max.columns = numerical_df.columns

numerical_df_max.head()
from sklearn.metrics.pairwise import cosine_similarity

#compute cosine similarity
sig = cosine_similarity(numerical_df_max, numerical_df_max)

#reverse maping of indices and titles
indices = pd.Series(df.index, index = df['Title']).drop_duplicates()
indices.head()
def give_recomendation(title, sig = sig):
    """return the index of series of indices"""
    #get the index corresponding to original_title
    idx = indices[title]
    
    #get the pairwise similarity scores
    sig_scores = list(enumerate(sig[idx]))
    
    #sort the movies
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    
    #scores of 10 most similar movies
    movie_indices = [i[0] for i in sig_scores]
    #top 10 most similar movies
    return df['Title'].iloc[movie_indices]
#lets try
give_recomendation("The Matrix", sig= sig)
#the function performs all the important preprocessing steps
def preprocess(df):
    
    #combining all text columns
    # Selecting all object data type and storing them in list
    s = list(df.select_dtypes(include=['object']).columns)
    
    
    # Removing ID and Title column
    s.remove("Title")
    s.remove("ID")
    
    # Joining all text/object columns using commas into a single column
    df['all_text']= df[s].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)

    # Creating a tokenizer to remove unwanted elements from our data like symbols and numbers
    token = RegexpTokenizer(r'[a-zA-Z]+')

    # Converting TfidfVector from the text
    cv = TfidfVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
    text_counts= cv.fit_transform(df['all_text'])

    # Aelecting numerical variables
    ndf = df.select_dtypes(include=['float64',"int64"])

    # Scaling Numerical variables
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

    # Applying scaler on our data and converting i into a data frame
    ndfmx = pd.DataFrame((scaler.fit_transform(ndf)))
    ndfmx.columns=ndf.columns    

    # Adding our adding numerical variables in the TF-IDF vector
    IMDb = ndfmx.IMDb.values[:, None]
    X_train_dtm = hstack((text_counts, IMDb))
    
    Netflix = ndfmx.Netflix.values[:, None]
    X_train_dtm = hstack((X_train_dtm, Netflix))
    
    Hulu = ndfmx.Hulu.values[:, None]
    X_train_dtm = hstack((X_train_dtm, Hulu))
    
    Prime = ndfmx["Prime Video"].values[:, None]
    X_train_dtm = hstack((X_train_dtm, Prime))
    
    Disney = ndfmx["Disney+"].values[:, None]
    X_train_dtm = hstack((X_train_dtm, Disney))
    
    Runtime = ndfmx.Runtime.values[:, None]
    X_train_dtm = hstack((X_train_dtm, Runtime))
    
    return X_train_dtm
# Preprocessing data
mat = preprocess(df)
mat.shape
# using cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

#compute the sigmoid kernel
sig2= cosine_similarity(mat, mat)

#reverse mapping of indices and movie titles
indices = pd.Series(df.index, index=df['Title']).drop_duplicates()
give_recomendation("The Matrix", sig = sig2)