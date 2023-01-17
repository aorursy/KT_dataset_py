# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cluster import KMeans 

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import PCA

from sklearn.preprocessing import normalize

from sklearn.metrics import pairwise_distances

from nltk.tokenize import word_tokenize

from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords

from bs4 import BeautifulSoup

from scipy.stats import multivariate_normal as mvn

import nltk

import os

import random









import string

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# email module has some useful functions

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('fivethirtyeight')



import os, sys, email,re

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/emails.csv',nrows = 35000)

df.shape
# create list of email objects

emails = list(map(email.parser.Parser().parsestr,df['message']))



# extract headings such as subject, from, to etc..

headings  = emails[0].keys()



# Goes through each email and grabs info for each key

# doc['From'] grabs who sent email in all emails

for key in headings:

    df[key] = [doc[key] for doc in emails]
##Useful functions

def get_raw_text(emails):

    email_text = []

    for email in emails.walk():

        if email.get_content_type() == 'text/plain':

            email_text.append(email.get_payload())

    return ''.join(email_text)



df['body'] = list(map(get_raw_text, emails))

df.head()

df['user'] = df['file'].map(lambda x: x.split('/')[0])
df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)

df.head()

df.dtypes
df['Month'] = df['Date'].dt.month

df['Year'] = df['Date'].dt.year

df['Day'] = df['Date'].dt.dayofweek



# reduce the sample period

# looks like the total number of emails really ramped up in 2000 and 2001

indices = (df['Year'] > 1995) & (df['Year'] <= 2004)

plt.figure(figsize = (10,6))

figure1 = df.loc[indices].groupby('Year')['body'].count().plot()
df.Year.agg({'max': max, 'min': min})

# really should not be dates up to 2044

df[df['Year']==2044]
plt.figure(figsize = (10,6))

figure2 = df.groupby('Month')['body'].count().plot()
plt.figure(figsize = (10,6))

figure3 = df.groupby('Day')['body'].count().plot()
#Unique to and From

print('Total number of emails: %d' %len(df))

print('------------')

print('Number of unique received: %d '%df['To'].nunique())

print('------------')

print('Number of unique Sent: %d '%df['From'].nunique())
top_10_frequent = df.groupby('user')['file'].count().sort_values(ascending = False)[:30]

top_10_frequent
plt.figure(figsize = (10,6))

top_10_frequent.plot(kind = 'bar')
df.groupby(['user', 'Year'])['file'].count()
def split_data(data):

    if data is not None:

        temp = data.split(',')

        if len(temp) == 1:

            return 'Direct'

        else:

            return 'Multiple'

    else:

        return 'Empty'

df['Direct_or_multi'] = df['To'].apply(split_data)
df.groupby('user')['Direct_or_multi'].value_counts().sort_values(ascending=False)[:15]
def clean_column(data):

    if data is not None:

        stopwords_list = stopwords.words('english')

        #exclusions = ['RE:', 'Re:', 're:']

        #exclusions = '|'.join(exclusions)

        data =  data.lower()

        data = re.sub('re:', '', data)

        data = re.sub('-', '', data)

        data = re.sub('_', '', data)

        # Remove data between square brackets

        data =re.sub('\[[^]]*\]', '', data)

        # removes punctuation

        data = re.sub(r'[^\w\s]','',data)

        data = re.sub(r'\n',' ',data)

        data = re.sub(r'[0-9]+','',data)

        # strip html 

        p = re.compile(r'<.*?>')

        data = re.sub(r"\'ve", " have ", data)

        data = re.sub(r"can't", "cannot ", data)

        data = re.sub(r"n't", " not ", data)

        data = re.sub(r"I'm", "I am", data)

        data = re.sub(r" m ", " am ", data)

        data = re.sub(r"\'re", " are ", data)

        data = re.sub(r"\'d", " would ", data)

        data = re.sub(r"\'ll", " will ", data)

        data = re.sub('forwarded by phillip k allenhouect on    pm', '',data)

        data = re.sub(r"httpitcappscorpenroncomsrrsauthemaillinkaspidpage", "", data)

        

        data = p.sub('', data)

        if 'forwarded by:' in data:

            data = data.split('subject')[1]

        data = data.strip()

        return data

    return 'No Subject'





df['Subject_new'] = df['Subject'].apply(clean_column)

df['body_new'] = df['body'].apply(clean_column)
df['body_new'].head(5)
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)

to_add = ['FW', 'ga', 'httpitcappscorpenroncomsrrsauthemaillinkaspidpage', 'cc', 'aa', 'aaa', 'aaaa',

         'hou', 'cc', 'etc', 'subject', 'pm']



for i in to_add:

    stopwords.add(i)
wordcloud = WordCloud(

                          collocations = False,

                          width=1600, height=800,

                          background_color='white',

                          stopwords=stopwords,

                          max_words=150,

                          #max_font_size=40, 

                          random_state=42

                         ).generate(' '.join(df['Subject_new'])) # can't pass a series, needs to be strings and function computes frequencies

print(wordcloud)

plt.figure(figsize=(9,8))

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
stemmer = PorterStemmer()

def stemming_tokenizer(str_input):

    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()

    words = [porter_stemmer.stem(word) for word in words]

    return words
def tokenize_and_stem(text):

    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token

    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

    filtered_tokens = []

    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)

    for token in tokens:

        if re.search('[a-zA-Z]', token):

            filtered_tokens.append(token)

    stems = [stemmer.stem(t) for t in filtered_tokens]

    return stems
from sklearn.feature_extraction.text import TfidfVectorizer

data = df['body_new']

# data.head()



tf_idf_vectorizor = TfidfVectorizer(stop_words = stopwords,#tokenizer = tokenize_and_stem,

                             max_features = 5000)

%time tf_idf = tf_idf_vectorizor.fit_transform(data)

tf_idf_norm = normalize(tf_idf)

tf_idf_array = tf_idf_norm.toarray()

pd.DataFrame(tf_idf_array, columns=tf_idf_vectorizor.get_feature_names()).head()
# initial_centroids = np.random.permutation(tf_idf_array.shape[0])[:3]

# initial_centroids

# centroids = tf_idf_array[initial_centroids]

# centroids.shape

# dist_to_centroid =  pairwise_distances(tf_idf_array,centroids, metric = 'euclidean')

# cluster_labels = np.argmin(dist_to_centroid, axis = 1)
class Kmeans:

    """ K Means Clustering

    

    Parameters

    -----------

        k: int , number of clusters

        

        seed: int, will be randomly set if None

        

        max_iter: int, number of iterations to run algorithm, default: 200

        

    Attributes

    -----------

       centroids: array, k, number_features

       

       cluster_labels: label for each data point

       

    """

    

    def __init__(self, k, seed = None, max_iter = 200):

        self.k = k

        self.seed = seed

        if self.seed is not None:

            np.random.seed(self.seed)

        self.max_iter = max_iter

        

            

    

    def initialise_centroids(self, data):

        """Randomly Initialise Centroids

        

        Parameters

        ----------

        data: array or matrix, number_rows, number_features

        

        Returns

        --------

        centroids: array of k centroids chosen as random data points 

        """

        

        initial_centroids = np.random.permutation(data.shape[0])[:self.k]

        self.centroids = data[initial_centroids]



        return self.centroids

    

    

    def assign_clusters(self, data):

        """Compute distance of data from clusters and assign data point

           to closest cluster.

        

        Parameters

        ----------

        data: array or matrix, number_rows, number_features

        

        Returns

        --------

        cluster_labels: index which minmises the distance of data to each

        cluster

            

        """

        

        if data.ndim == 1:

            data = data.reshape(-1, 1)

        

        dist_to_centroid =  pairwise_distances(data, self.centroids, metric = 'euclidean')

        self.cluster_labels = np.argmin(dist_to_centroid, axis = 1)

        

        return  self.cluster_labels

    

    

    def update_centroids(self, data):

        """Computes average of all data points in cluster and

           assigns new centroids as average of data points

        

        Parameters

        -----------

        data: array or matrix, number_rows, number_features

        

        Returns

        -----------

        centroids: array, k, number_features

        """

        

        self.centroids = np.array([data[self.cluster_labels == i].mean(axis = 0) for i in range(self.k)])

        

        return self.centroids

    

    

    def convergence_calculation(self):

        """

        Calculates 

        

        """

        pass

    

    def predict(self, data):

        """Predict which cluster data point belongs to

        

        Parameters

        ----------

        data: array or matrix, number_rows, number_features

        

        Returns

        --------

        cluster_labels: index which minmises the distance of data to each

        cluster

        """

        

        return self.assign_clusters(data)

    

    def fit_kmeans(self, data):

        """

        This function contains the main loop to fit the algorithm

        Implements initialise centroids and update_centroids

        according to max_iter

        -----------------------

        

        Returns

        -------

        instance of kmeans class

            

        """

        self.centroids = self.initialise_centroids(data)

        

        # Main kmeans loop

        for iter in range(self.max_iter):



            self.cluster_labels = self.assign_clusters(data)

            self.centroids = self.update_centroids(data)          

            if iter % 100 == 0:

                print("Running Model Iteration %d " %iter)

        print("Model finished running")

        return self    
from sklearn.datasets import make_blobs

# create blobs

data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.6, random_state=50)

# create np array for data points

points = data[0]

# create scatter plot

plt.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap='viridis')

plt.xlim(-15,15)

plt.ylim(-15,15)



X = data[0]

X[2]
temp_k  = Kmeans(4, 1, 600)

temp_fitted  = temp_k.fit_kmeans(X)

new_data = np.array([[1.066, -8.66],

                    [1.87876, -6.516],

                    [-1.59728965,  8.45369045],

                    [1.87876, -6.516]])

temp_fitted.predict(new_data)
sklearn_pca = PCA(n_components = 2)

Y_sklearn = sklearn_pca.fit_transform(tf_idf_array)

test_e = Kmeans(3, 1, 600)

%time fitted = test_e.fit_kmeans(Y_sklearn)

predicted_values = test_e.predict(Y_sklearn)



plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], c=predicted_values, s=50, cmap='viridis')



centers = fitted.centroids

plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.6);
from sklearn.cluster import KMeans

n_clusters = 3

sklearn_pca = PCA(n_components = 2)

Y_sklearn = sklearn_pca.fit_transform(tf_idf_array)

kmeans = KMeans(n_clusters= n_clusters, max_iter=600, algorithm = 'auto')

%time fitted = kmeans.fit(Y_sklearn)

prediction = kmeans.predict(Y_sklearn)



plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1],c=prediction ,s=50, cmap='viridis')



centers2 = fitted.cluster_centers_

plt.scatter(centers2[:, 0], centers2[:, 1],c='black', s=300, alpha=0.6);
number_clusters = range(1, 7)



kmeans = [KMeans(n_clusters=i, max_iter = 600) for i in number_clusters]

kmeans



score = [kmeans[i].fit(Y_sklearn).score(Y_sklearn) for i in range(len(kmeans))]

score = [i*-1 for i in score]



plt.plot(number_clusters, score)

plt.xlabel('Number of Clusters')

plt.ylabel('Score')

plt.title('Elbow Method')

plt.show()
def get_top_features_cluster(tf_idf_array, prediction, n_feats):

    labels = np.unique(prediction)

    dfs = []

    for label in labels:

        id_temp = np.where(prediction==label) # indices for each cluster

        x_means = np.mean(tf_idf_array[id_temp], axis = 0) # returns average score across cluster

        sorted_means = np.argsort(x_means)[::-1][:n_feats] # indices with top 20 scores

        features = tf_idf_vectorizor.get_feature_names()

        best_features = [(features[i], x_means[i]) for i in sorted_means]

        df = pd.DataFrame(best_features, columns = ['features', 'score'])

        dfs.append(df)

    return dfs

dfs = get_top_features_cluster(tf_idf_array, prediction, 20)
import seaborn as sns

plt.figure(figsize=(8,6))

sns.barplot(x = 'score' , y = 'features', orient = 'h' , data = dfs[0][:15])
plt.figure(figsize=(8,6))

sns.barplot(x = 'score' , y = 'features', orient = 'h' , data = dfs[1][:15])
plt.figure(figsize=(8,6))

sns.barplot(x = 'score' , y = 'features', orient = 'h' , data = dfs[2][:15])
for i, df in enumerate(dfs):

    df.to_csv('df_'+str(i)+'.csv')
def plot_features(dfs):

    fig = plt.figure(figsize=(14,12))

    x = np.arange(len(dfs[0]))

    for i, df in enumerate(dfs):

        ax = fig.add_subplot(1, len(dfs), i+1)

        ax.set_title("Cluster: "+ str(i), fontsize = 14)

        ax.spines["top"].set_visible(False)

        ax.spines["right"].set_visible(False)

        ax.set_frame_on(False)

        ax.get_xaxis().tick_bottom()

        ax.get_yaxis().tick_left()

        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))

        ax.barh(x, df.score, align='center', color='#40826d')

        yticks = ax.set_yticklabels(df.features)

    plt.show();

plot_features(dfs)