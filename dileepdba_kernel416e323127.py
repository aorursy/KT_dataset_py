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
docmetadata= pd.read_csv ('/kaggle/input/CORD-19-research-challenge/metadata.csv')
docmetadata.head()
# Import all the json files

cord_19_folder = '/kaggle/input/CORD-19-research-challenge/metadata.csv'

list_of_files = []; # only going to take those from pdf_json! not pmc_json

for root, dirs, files in os.walk(cord_19_folder):

    for name in files:

        if (name.endswith('.json')):

            full_path = os.path.join(root, name)

            list_of_files.append(full_path)

sorted(list_of_files)

print('done')
simple_schema = ['cord_uid','source_x', 'title', 'abstract', 'authors', 'full_text_file', 'url']

def make_clickable(address):

    '''Make the url clickable'''

    return '<a href="{0}">{0}</a>'.format(address)

def preview(text):

    '''Show only a preview of the text data.'''

    return text[:30] + '...'



format_ = {'title': preview, 'abstract': preview, 'authors': preview, 'url': make_clickable}



print("Number of articles in the metadata excel is : " , len(docmetadata))
docmetadata.info()
docmetadata.isnull().sum()
import seaborn as sns

sns.heatmap(docmetadata.isnull(),yticklabels=False,cbar=False,cmap="viridis")
docmetadata['source_x'].value_counts()
docmetadata['abstract'].describe(include='all')
docmetadata.drop_duplicates(['abstract'], inplace=True)

docmetadata['abstract'].describe(include='all')
## conda install -c plotly plotly



def abstract_len(a):

    if type(a) is str:

        return len(a.split())

    else:

        return 0



docmetadata["abstract_words"] = docmetadata["abstract"].apply(abstract_len)

num_of_word = docmetadata.query("abstract_words != 0 and abstract_words <500")["abstract_words"]



sns.distplot(num_of_word)


docmetadata['publish_time'].describe()
#define a function that returns the year component from date

def get_year(dt):

    dt=str(dt)

    year= dt.split('-')

    return year[0]



#Check the count of articles published by year in this dataset



#domainplot= sns.countplot(yearvise_articles)

#docmetadata.groupby(docmetadata['publish_time'].apply(get_year))['cord_uid'].count().sort_values(ascending=False).head(20)

domainplot=docmetadata.groupby(docmetadata['publish_time'].apply(get_year))['cord_uid'].count().plot(linestyle='--', marker='o', color='m')

domainplot.set_xticklabels(domainplot.get_xticklabels(),rotation=90, ha="right",fontsize=20)
from tqdm import tqdm

Covid = docmetadata[docmetadata['publish_time'].str.contains('2019') | docmetadata['publish_time'].str.contains('2020')]
Covid= Covid[Covid["abstract"].notna()] #.dropna(subset=["abstract"])
Covid.shape
Covid.shape
Covid.head()
Covid.shape
from nltk.tokenize import RegexpTokenizer

from stop_words import get_stop_words

from nltk.stem.porter import PorterStemmer
tokenizer = RegexpTokenizer(r'\w+')



# create English stop words list

en_stop = get_stop_words('en')





# Create p_stemmer of class PorterStemmer

p_stemmer = PorterStemmer()

    

# create sample documents



abstracts=Covid.abstract
# list for tokenized documents in loop

texts = []

# loop through document list

for i in abstracts:

    

    # clean and tokenize document string

    raw = i.lower()

    tokens = tokenizer.tokenize(raw)



    # remove stop words from tokens

    stopped_tokens = [i for i in tokens if not i in en_stop]

    

    # stem tokens

    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

    

    #remove tokens less than length 3

    stemmed_len_tokens = [i for i in stemmed_tokens if len(i)>4]

    

    # add tokens to list

    texts.append(stemmed_len_tokens)

print(len(texts))
Covid["processed_punctuation"]=texts
def joining(row):

    return ' '.join(row)

Covid['processed']=Covid.processed_punctuation.apply(joining)



Covid.head()
from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize(text, maxx_features):

    

    vectorizer = TfidfVectorizer(max_features=maxx_features)

    #vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(text)

    return X
text = Covid['processed'].values

X = vectorize(text,1000)

X.shape
from sklearn.decomposition import PCA



pca = PCA(n_components=0.90, random_state=42)

X_reduced = pca.fit_transform(X.toarray())

X_reduced.shape
import scipy.cluster.hierarchy as sch


dendrogram = sch.dendrogram(sch.linkage(X_reduced, method="ward"))

plt.title("Dendrogram")

plt.xlabel("Words")

plt.ylabel("Euclidean Distance")

plt.show()


dendrogram = sch.dendrogram(sch.linkage(X_reduced, method="ward"))

plt.title("Dendrogram")

plt.xlabel("Words")

plt.ylabel("Euclidean Distance")

plt.axhline(y=20,color="r",linestyle="--")

plt.show()
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=2, linkage="ward",affinity="euclidean")
pred= hc.fit_predict(X_reduced)


pred
Covid["HierarchicalCluster"]=pred
Covid["HierarchicalCluster"].value_counts()
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2, init = 'k-means++',random_state=35)
kmeans.fit(X_reduced)
pred=kmeans.predict(X_reduced)
kmeans.inertia_
from sklearn import metrics

metrics.silhouette_score(X_reduced, pred)
 

    

for i in range(1,21):

 kmeans = KMeans(n_clusters= i, init='k-means++', random_state=42,max_iter=10000,tol=0.000001)

 kmeans.fit(X_reduced)

 wcss = (kmeans.inertia_)



 #inertia_ is the formula used to segregate the data points into clusters

#Visualizing the ELBOW method to get the optimal value of K 

plt.figure(figsize=(20,10))

plt.plot(range(1,21), wcss)

plt.title('The Elbow Method')

plt.xlabel('no of clusters')

plt.ylabel('wcss')

plt.show()

   
import matplotlib.pyplot as plt