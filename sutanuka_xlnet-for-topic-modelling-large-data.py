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
from tqdm import tqdm

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

%matplotlib inline



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

import re

import string



import gensim

import gensim.corpora as corpora

from gensim.utils import simple_preprocess

from gensim.models import CoherenceModel

from gensim.models import LsiModel

from gensim.models.coherencemodel import CoherenceModel



from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer 

import nltk

from nltk.tokenize import word_tokenize



import torch

print(torch.__version__)
!pip install flair
pip --version
from flair.embeddings import FlairEmbeddings, DocumentPoolEmbeddings, BertEmbeddings, XLNetEmbeddings
import wandb
#from flair.embeddings import Sentence

from flair.data import Sentence
# initialise embedding classes

flair_embedding_forward = FlairEmbeddings("news-forward")

flair_embedding_backward = FlairEmbeddings("news-backward")

#bert_embedding = BertEmbeddings()

xlnet_embedding = XLNetEmbeddings("xlnet-large-cased")
# combine word embedding models

document_embeddings = DocumentPoolEmbeddings([xlnet_embedding, flair_embedding_backward, flair_embedding_forward])
data = pd.read_csv('/kaggle/input/npr-data/npr.csv')

#df = data['Text']

#data_1 = data.tolist()

print(data.shape)
test_data = pd.read_csv('/kaggle/input/reviews/reviews.csv')

test_data_doc = test_data['Text'].values.tolist()
df = data[500:900]

print(df.shape)

print(df.head())
documents = df['Article'].values.tolist()

print(type(documents))

print(type(documents[0]))

print(len(documents))
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()
def preprocess_data(doc_set):

    

    texts = []

    for i in doc_set:

        

        #Lowercase

        input_str = i.lower()

    

        #Remove numbers

        input_str_1 = re.sub(r'\d+', '', input_str)

    

        #Remove Punctuation

        #input_str_2 = input_str_1.translate(str.maketrans('', '', string.punctuation))

        input_str_2 = re.sub('[^a-zA-Z0-9]{3,}', ' ', input_str_1)

        input_str_2 = re.sub('\[.*?\]', '', input_str_2)

        input_str_2 = re.sub('[%s]' % re.escape(string.punctuation), '', input_str_2)

        input_str_2 = re.sub('\w*\d\w*', '', input_str_2)

        input_str_2 = re.sub('[‘’“”…]', '', input_str_2)

        input_str_2 = re.sub('\n', '', input_str_2)

    

        #Remove stopwords

        word_tokens = word_tokenize(input_str_2) 

        input_str_3 = [word for word in word_tokens if word not in stop_words]

    

        #Lemmatize

        input_str_4 = [lemmatizer.lemmatize(word, pos ='v') for word in input_str_3]

        

        output = (' '.join(input_str_4))

        

        texts.append(output)

        #return(input_str_2)

        

    return(texts)
clean_text=preprocess_data(documents)
clean_text[13]
print(type(clean_text[0]))

print(type(clean_text))

len(clean_text)

#print(type(Cleaned_Data[0]))

#print(type(Cleaned_Data))

#len(Cleaned_Data)
df['Article_clean'] = clean_text
df['Article_clean'][13]
df.head()
# set up empty tensor

X = torch.empty(size=(len(df.index), 6144)).cuda()
type(document_embeddings)
%%time

# fill tensor with embeddings

i=0

for text in tqdm(df['Article_clean']):

    sentence = Sentence(text)

    document_embeddings.embed(sentence)

    embedding = sentence.get_embedding()

    X[i] = embedding

    i += 1
X = X.cpu().detach().numpy()
#pca = PCA(n_components=300)
#X_red = pca.fit_transform(X)
#var= pca.explained_variance_ratio_
#var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

#plt.plot(var1)
#pca = PCA(n_components=300)

#X_red = pca.fit_transform(X)
X_red = X
# fitting multiple k-means algorithms and storing the values in an empty list

SSE = []

for cluster in range(1,10):

    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')

    kmeans.fit(X_red)

    SSE.append(kmeans.inertia_)



# converting the results into a dataframe and plotting them

frame = pd.DataFrame({'Cluster':range(1,10), 'SSE':SSE})

plt.figure(figsize=(12,6))

plt.plot(frame['Cluster'], frame['SSE'], marker='o')

plt.xlabel('Number of clusters')

plt.ylabel('Inertia')
N_CLUSTERS = 6



kmeans = KMeans(n_jobs = -1, n_clusters = N_CLUSTERS, init='k-means++')

kmeans.fit(X_red)

pred = kmeans.fit_predict(X_red)
df['new_topic_km'] = pred
print(df['new_topic_km'].value_counts())
def get_top_words(documents, top_n):

  '''

  function to get top tf-idf words and phrases

  '''

  vectoriser = TfidfVectorizer(ngram_range=(1, 2),

                               max_df=0.5)

  tfidf_matrix = vectoriser.fit_transform(documents)

  feature_names = vectoriser.get_feature_names()

  df_tfidf = pd.DataFrame()

  for doc in range(len(documents)):

    words = []

    scores = []

    feature_index = tfidf_matrix[doc,:].nonzero()[1]

    tfidf_scores = zip(feature_index, [tfidf_matrix[doc, x] for x in feature_index])

    for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:

      words.append(w)

      scores.append(s)

    df_temp = pd.DataFrame(data={'word':words, 'score':scores})

    df_temp = df_temp.sort_values('score',ascending=False).head(top_n)

    df_temp['topic'] = doc

    df_tfidf = df_tfidf.append(df_temp)

  return df_tfidf
topic_docs = []

# group text into topic-documents

for topic in range(N_CLUSTERS):

    topic_docs.append(' '.join(df[df['new_topic_km']==topic]['Article_clean'].values))
# apply function

df_tfidf = get_top_words(topic_docs, 10)
df_tfidf.loc[df_tfidf['topic'] == 0]
df_tfidf.loc[df_tfidf['topic'] == 1]
df_tfidf.loc[df_tfidf['topic'] == 2]
df_tfidf.loc[df_tfidf['topic'] == 3]
df_tfidf.loc[df_tfidf['topic'] == 4]

df_tfidf.loc[df_tfidf['topic'] == 5]