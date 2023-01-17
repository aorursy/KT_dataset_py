# Load required libraries

import numpy as np

import pandas as pd

import sklearn

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import SGDClassifier

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn.cluster import KMeans, MiniBatchKMeans  # MiniBatchKMeans really helps to fasten processing time

from nltk import wordpunct_tokenize

from nltk.stem import WordNetLemmatizer

import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS

import math as math
class LemmaTokenizer(object):

    """Custom tokenizer class that stems tokens"""

    def __init__(self):

        self.wnl = WordNetLemmatizer()

    def __call__(self,doc):

        return [self.wnl.lemmatize(t) for t in wordpunct_tokenize(doc) if len(t.strip()) > 1]

    

def show_topn(classifier,vectorizer,categories,n):

    """Returns the top n features that characterize eachc category"""

    feature_names = np.asarray(vectorizer.get_feature_names())

    for i, category in enumerate(categories):

        topn = np.argsort(classifier.coef_[i])[-n:]

        print('{}: {}'.format(category,", ".join(feature_names[topn])))

        

def save_topn(classifier,vectorizer,categories,n,outdict):

    """Returns the top n features that characterize eachc category, and save result in outdict"""

    feature_names = np.asarray(vectorizer.get_feature_names())

    for i, category in enumerate(categories):

        topn = np.argsort(classifier.coef_[i])[-n:]

        outdict[i] = feature_names[topn]

# read in a few columns from the data and show the top of the resulting dataframe

df = pd.read_csv('../input/listings.csv', usecols = ['id', 'name', 'space', 'description', 'neighborhood_overview', 'neighbourhood_cleansed'])



df.head()
# Check the full text in each of the column

for i in range(len(df.columns)):

    print(df.columns[i],": ")

    print(df.iloc[0,i])

    print('=======================')
# let's combine the name, space, description, and neighborhood_overview into a new column

df['combined_description'] = df.apply(lambda x: '{} {} {} {}'.format(x['name'], x['space'], x['description'], x['neighborhood_overview']), axis=1)

print(df.loc[0,'combined_description'])
# Transform combined_description into tfidf format

tfidf = TfidfVectorizer(ngram_range=(1,2),stop_words='english',tokenizer=LemmaTokenizer())

tfidf.fit(df['combined_description'])

DescTfidf = tfidf.transform(df['combined_description'])
# I added a chart to replace tabulation in the original notebook



neighborRank = df.groupby(by='neighbourhood_cleansed').count()[['id']].sort_values(by='id', ascending=False)

# print(neighborRank)

plt.figure(figsize=(10,10))

g = sns.barplot(y=neighborRank.index,x=neighborRank["id"])

# The line below adds the value label in each bar

[g.text(p[1]+1,p[0],p[1], color='black') for p in zip(g.get_yticks(), neighborRank["id"])]

plt.title('Number of Listings in Each Neighbourhood')
# Create K-Means using MiniBatchKMeans. The MiniBatch version works much faster than regular KMeans

kmeans6 = MiniBatchKMeans(n_clusters=6)

DescKmeans6 = kmeans6.fit_predict(DescTfidf.todense())
# Combine description, cluster, and neighborhood into one dataframe. 

FullDescKmeans6 = pd.concat([pd.DataFrame(DescKmeans6),df[['combined_description','neighbourhood_cleansed']]],axis=1)

FullDescKmeans6.columns = ['Cluster','Description','Neighbourhood']  

print(FullDescKmeans6.head())
# Show and plot the number of listings in each cluster

ClusterCount = FullDescKmeans6['Cluster'].value_counts().sort_index()

ClusterCount = pd.DataFrame(ClusterCount)

ClusterCount.columns=['NumListings']

g = sns.barplot(x=FullDescKmeans6['Cluster'].value_counts().index,y=FullDescKmeans6['Cluster'].value_counts())

[g.text(p[0]-0.15,p[1]+5,p[1], color='black') for p in zip(g.get_xticks(), ClusterCount["NumListings"])]

plt.title('Number of Listings in Each Description-based Clusters')
# Create crosstab between Cluster and Neighbourhood 

ctab = pd.crosstab(index=FullDescKmeans6['Neighbourhood'],columns=FullDescKmeans6['Cluster'])

plt.figure(figsize=(10,10))

sns.heatmap(ctab,annot=True,cmap='Blues', fmt='g')

plt.title("Crosstab of Cluster and Neighbourhood")
# Let's take a look at the full description from a couple of listings

#for i in range(6):

#    subset = FullDescKmeans6[FullDescKmeans6['Cluster']==i]

#    print('We are at cluster..')

#    print(i)

#    for j in range(1):

#        print(subset.iloc[j,1])

#        print('--------------------------------')
# Pipeline to identify top 30 words that are "best predictor" of a cluster

pipeline = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2), stop_words='english', tokenizer=LemmaTokenizer())),

                     ('clf', SGDClassifier(loss='hinge', penalty='l2',

                                           alpha=1e-3, n_iter=5, random_state=42)),

])

modelSegment = pipeline.fit(df['combined_description'],FullDescKmeans6['Cluster'])

# Create wordcloud based on top-30 words

Keywords6 = {}

save_topn(modelSegment.named_steps['clf'], modelSegment.named_steps['tfidf'], [str(i) for i in range(6)], 30,outdict=Keywords6)

fig,axes=plt.subplots(2,3,figsize=(30,12))

for i in range(6):

    wordlist = list(Keywords6[i])

    wc = WordCloud(background_color='white',max_words=30,relative_scaling=0.2).generate(" ".join(wordlist))

    print(wc)

    axes[math.floor(i/3),i%3].imshow(wc)
# Now we'll try to create a cluster of 12. Hopefully we could get a well spread clustering

kmeans12 = MiniBatchKMeans(n_clusters=12,batch_size=128)

DescKmeans12 = kmeans12.fit_predict(DescTfidf.todense())
FullDescKmeans12 = pd.concat([pd.DataFrame(DescKmeans12),df[['combined_description','neighbourhood_cleansed']]],axis=1)

FullDescKmeans12.columns = ['Cluster','Description','Neighbourhood']

g = sns.barplot(x=FullDescKmeans12['Cluster'].value_counts().index,y=FullDescKmeans12['Cluster'].value_counts()) 

plt.title("Number of listings in each cluster")
# Create crosstab between Cluster and Neighbourhood 

ctab = pd.crosstab(index=FullDescKmeans12['Neighbourhood'],columns=FullDescKmeans12['Cluster'])

plt.figure(figsize=(10,10))

sns.heatmap(ctab,annot=True,cmap='Blues', fmt='g')

plt.title("Crosstab of Cluster and Neighbourhood")
# I previously use a regular print-out of the words, but now I am using a wordcloud instead

modelSegment = pipeline.fit(df['combined_description'],FullDescKmeans12['Cluster'])

# show_topn(modelSegment.named_steps['clf'], modelSegment.named_steps['tfidf'], [str(i) for i in range(12)], 20)
# Create wordcloud based on top-30 words

Keywords12 = {}

save_topn(modelSegment.named_steps['clf'], modelSegment.named_steps['tfidf'], [str(i) for i in range(12)], 30,outdict=Keywords12)

fig,axes=plt.subplots(4,3,figsize=(20,20))

for i in range(12):

    wordlist = list(Keywords12[i])

    wc = WordCloud(background_color='white',max_words=30,relative_scaling=0.2).generate(" ".join(wordlist))

    print(wc)

    axes[math.floor(i/3),i%3].imshow(wc)