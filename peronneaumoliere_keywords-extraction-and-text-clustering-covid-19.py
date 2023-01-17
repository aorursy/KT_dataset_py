# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import warnings

warnings.filterwarnings('ignore')

#load the data

df=pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')
#first five rows

df.head()
#dataframe info

df.info()
#Dataframe describe

df.describe()
#create a new dataframe with most important columns for us

df=df[['publish_time','authors','title','abstract']]

df.head()
df.info()
#check the total null cell for the column of abstract

df['abstract'].isnull().sum()
#delete rows where  abstract are null

df.dropna(subset=['abstract'], inplace=True)

df.info()
#Fetch word count for each abstract

df['word_count'] = df['abstract'].apply(lambda x: len(str(x).split(" ")))

df.head()
#Descriptive statistics of word counts

df.describe()
#Identify common words (20 top words)

freq = pd.Series(' '.join(df['abstract']).split()).value_counts()[:20]

freq
#plot the most 20 common words

freq.plot()
#Identify uncommon words (top 20)

freq1 =  pd.Series(' '.join(df['abstract']).split()).value_counts()[-20:]

freq1
#Import the required libraries for the text processing

import re

import nltk

from nltk.stem.porter import PorterStemmer

from nltk.tokenize import RegexpTokenizer

nltk.download('stopwords')

from nltk.corpus import stopwords

nltk.download('wordnet') 

from nltk.stem.wordnet import WordNetLemmatizer
#Removing stopwords

    ##Creating a list of stop words and adding custom stopwords

stop_words = set(stopwords.words("english"))



    ##Creating a list of custom stopwords (all other words you want to remove from the text)

new_words = ["using", "show", "result", "also", "iv", "one", 'however',"two", "new", "previously", "shown"]

stop_words = stop_words.union(new_words)
#carry out the pre-processing tasks step-by-step to get a cleaned and normalised text corpus:

corpus = []

for i in list(df.index.values): # list of index of the dataframe [0,1,2......]'

    #Remove punctuations

    text = re.sub('[^a-zA-Z]', ' ', df['abstract'][i])

    #Convert to lowercase

    text = text.lower()

    #remove tags

    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

    #remove special characters and digits

    text=re.sub("(\\d|\\W)+"," ",text)

    #Convert to list from string

    text = text.split()

    #Stemming

    ps=PorterStemmer()

    #Lemmatisation

    lem = WordNetLemmatizer()

    text = [lem.lemmatize(word) for word in text if not word in  

            stop_words] 

    text = " ".join(text)

    corpus.append(text)
#View corpus item

corpus[1000]
#Word cloud: Vizualize the corpus (frequency or the importance of each word)

#from os import path

#from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

%matplotlib inline

%matplotlib inline

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stop_words,

                          max_words=100,

                          max_font_size=70, 

                          random_state=42

                         ).generate(str(corpus))

print(wordcloud)

fig = plt.figure(1,figsize=(20,10))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
#Identify common words in the corpus  (20 top words)

freq = pd.Series(' '.join(corpus).split()).value_counts()[:20]

freq
#plot the result (top 20 words in the corpus)

#Convert most freq words to dataframe for plotting bar plot

top_words = pd.Series(' '.join(corpus).split()).value_counts()[:20]

top_df = pd.DataFrame(top_words).reset_index()

top_df.columns=["Word", "Freq"]



#Barplot of most freq words

import seaborn as sns

sns.set(rc={'figure.figsize':(13,8)})

g = sns.barplot(x="Word", y="Freq", data=top_df)

g.set_title('Top 20 words in the corpus')

g.set_xticklabels(g.get_xticklabels(), rotation=30)

#Creating a vector of word counts

from sklearn.feature_extraction.text import CountVectorizer

import re

cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))

X=cv.fit_transform(corpus)
#shape of X

X.shape
#print a list of 10 vocabulary from the list of vocabulary

list(cv.vocabulary_.keys())[:10]
#Uni-grams

    #Most frequently occuring words

def get_top_unigram_words(corpus, n=None):

    vec = CountVectorizer().fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], 

                       reverse=True)

    return words_freq[:n]

    #Convert most freq words to dataframe for plotting bar plot

top_words = get_top_unigram_words(corpus, n=20)

top_df = pd.DataFrame(top_words)

top_df.columns=["Word", "Freq"]

    #Barplot of most freq words

import seaborn as sns

sns.set(rc={'figure.figsize':(13,8)})

g = sns.barplot(x="Word", y="Freq", data=top_df)

g.set_title('Top 20 Uni_grams')

g.set_xticklabels(g.get_xticklabels(), rotation=30)
#Bi_grams

    #Most frequently occuring Bi-grams

def get_top_bi_grams_words(corpus, n=None):

    vec1 = CountVectorizer(ngram_range=(2,2), max_features=4000).fit(corpus)

    bag_of_words = vec1.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec1.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], 

                reverse=True)

    return words_freq[:n]

top2_words = get_top_bi_grams_words(corpus, n=20)

top2_df = pd.DataFrame(top2_words)

top2_df.columns=["Bi-gram", "Freq"]



    #Barplot of most freq Bi-grams

import seaborn as sns

sns.set(rc={'figure.figsize':(13,8)})

h=sns.barplot(x="Bi-gram", y="Freq", data=top2_df)

h.set_title('Top 20 Bi_grams')

h.set_xticklabels(h.get_xticklabels(), rotation=45)
#Tri_Grams

    #Most frequently occuring Tri-grams

def get_top_n3_words(corpus, n=None):

    vec1 = CountVectorizer(ngram_range=(3,3), max_features=4000).fit(corpus)

    bag_of_words = vec1.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec1.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], 

                reverse=True)

    return words_freq[:n]

top3_words = get_top_n3_words(corpus, n=20)

top3_df = pd.DataFrame(top3_words)

top3_df.columns=["Tri-gram", "Freq"]

print(top3_df)

    #Barplot of most freq Tri-grams

import seaborn as sns

sns.set(rc={'figure.figsize':(13,8)})

j=sns.barplot(x="Tri-gram", y="Freq", data=top3_df)

j.set_title('Top 20 Tri_grams')

j.set_xticklabels(j.get_xticklabels(), rotation=45)
#4_Grams

    #Most frequently occuring 4-grams

def get_top_n4_words(corpus, n=None):

    vec1 = CountVectorizer(ngram_range=(3,3), max_features=4000).fit(corpus)

    bag_of_words = vec1.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec1.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], 

                reverse=True)

    return words_freq[:n]

top3_words = get_top_n4_words(corpus, n=20)

top3_df = pd.DataFrame(top3_words)

top3_df.columns=["4-gram", "Freq"]

print(top3_df)

    #Barplot of most freq Tri-grams

import seaborn as sns

sns.set(rc={'figure.figsize':(13,8)})

l=sns.barplot(x="4-gram", y="Freq", data=top3_df)

l.set_title('Top 20 4_grams')

l.set_xticklabels(j.get_xticklabels(), rotation=45)
#Converting to a matrix of integers

from sklearn.feature_extraction.text import TfidfTransformer

 

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)

tfidf_transformer.fit(X)



# get feature names

feature_names=cv.get_feature_names()
# Define Function for sorting tf_idf in descending order



from scipy.sparse import coo_matrix

def sort_coo(coo_matrix):

    tuples = zip(coo_matrix.col, coo_matrix.data)

    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

 

def extract_topn_from_vector(feature_names, sorted_items, topn=10):

    """get the feature names and tf-idf score of top n items"""

    

    #use only topn items from vector

    sorted_items = sorted_items[:topn]

 

    score_vals = []

    feature_vals = []

    

    # word index and corresponding tf-idf score

    for idx, score in sorted_items:

        

        #keep track of feature name and its corresponding score

        score_vals.append(round(score, 3))

        feature_vals.append(feature_names[idx])

 

    #create a tuples of feature,score

    #results = zip(feature_vals,score_vals)

    results= {}

    for idx in range(len(feature_vals)):

        results[feature_vals[idx]]=score_vals[idx]

    

    return results
#Extract the keywords for the abstract number 304 (1)

abstract_335=corpus[335]

    #generate tf-idf for the given document

tf_idf_vector_abstract_335=tfidf_transformer.transform(cv.transform([abstract_335]))

#sort the tf-idf vectors by descending order of scores



sorted_items=sort_coo(tf_idf_vector_abstract_335.tocoo())



#extract only the top n; n here is 5

keywords=extract_topn_from_vector(feature_names,sorted_items,5)

    

 

# now print the results

print("\nAbstract 335:")

print(abstract_335)

print("\nKeywords:")

for k in keywords:

    print(k,keywords[k])
#sort the tf-idf vectors by descending order of scores

tf_idf_vector_corpus=tfidf_transformer.transform(cv.transform(corpus))

keywords=[]

for b in tf_idf_vector_corpus:

    sorted_items=sort_coo(b.tocoo())

    keywords.append(extract_topn_from_vector(feature_names,sorted_items,5))
#add the keywords for each abstract in the Dataframe

pd.set_option('display.width', None)

pd.set_option('display.max_colwidth', -1)

pd.set_option('display.max_columns', None)

df['keywords']=keywords

df1=df.drop(columns='word_count', axis=1)

df1.head()
#Use the the algorith MinisBatch as a Classifier

    #Import the required libraries

import sklearn

import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
#predict the cluster

X1=tf_idf_vector_corpus



#Make the prediction for 10 clusters

k = 10



kmeans = MiniBatchKMeans(n_clusters=k)

y_pred = kmeans.fit_predict(X1)

y=y_pred
from sklearn.decomposition import PCA



pca = PCA(n_components=3)

pca_result = pca.fit_transform(X1.toarray())
#Vizualize the clusters

# sns settings

sns.set(rc={'figure.figsize':(15,15)})

# colors

palette = sns.color_palette("bright", len(set(y)))

# plot

sns.scatterplot(pca_result[:,0], pca_result[:,1], hue=y, legend='full', palette=palette)

plt.title("Covid-19 Abstracts - Clustered (K-Means)")

plt.show()
#vizualize in 3D

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



ax = plt.figure(figsize=(16,10)).gca(projection='3d')

ax.scatter(

    xs=pca_result[:,0], 

    ys=pca_result[:,1], 

    zs=pca_result[:,2], 

    c=y, 

    cmap='tab10'

)

ax.set_xlabel('PCA_1')

ax.set_ylabel('PCA_2')

ax.set_zlabel('PCA_3')

plt.title("Covid-19 Abstracts - Clustered (K-Means)")

plt.show()
df1['cluster']=y

df1.head()
#Generate the size of each cluster

df1.groupby('cluster').apply(len)