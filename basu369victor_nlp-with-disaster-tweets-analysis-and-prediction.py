# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import networkx as nx

import os

import re

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import KFold, StratifiedKFold

import matplotlib.pyplot as plt

from sklearn.decomposition import TruncatedSVD, NMF,LatentDirichletAllocation

import xgboost as xgb

from sklearn.metrics import roc_auc_score,accuracy_score,log_loss

from catboost import CatBoostClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.cluster import KMeans

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

import lightgbm as lgb

import seaborn as sns

from tqdm import tqdm as tqdm_base

from sklearn.model_selection import train_test_split

#from gensim.models.ldamodel import LdaModel

from sklearn import preprocessing

#from gensim.corpora import Dictionary

import umap

import seaborn as sns

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
BASE_PATH = "/kaggle/input/nlp-getting-started/"
train =pd.read_csv(BASE_PATH + "train.csv")

train.head()
test =pd.read_csv(BASE_PATH + "test.csv")

test.head()
train.dtypes
train['keyword'].value_counts(dropna=False, normalize=True).head()
train['location'].value_counts(dropna=False, normalize=True).head()
plt.hist(train['target'], label='train');

plt.legend();

plt.title('Distribution of target labels');
#train['location'].fillna('NULL', inplace=True)

#train['keyword'].fillna('NULL', inplace=True)
temp = train.dropna()

temp = temp.reset_index()
temp.shape
g = nx.DiGraph()

for x in range(500):

    g.add_edge(temp['location'][x],temp['keyword'][x], weight=x,capacity=5,length = 100) 

pos=nx.spring_layout(g,k=0.15,)

plt.figure(figsize =(20, 20)) 

nx.draw_networkx(g,pos,alpha=0.8,node_color='red',node_size=25,font_size=9, with_label = True) 
g = nx.DiGraph()

for x in range(500,1000):

    g.add_edge(train['location'][x], train['keyword'][x]) 

    

pos=nx.spring_layout(g,k=0.15,)

plt.figure(figsize =(20, 20)) 

nx.draw_networkx(g,pos,alpha=0.8,node_color='red',node_size=25,font_size=9, with_label = True) 
g = nx.DiGraph()

for x in range(1000,1500):

    g.add_edge(train['location'][x], train['keyword'][x]) 

    

pos=nx.spring_layout(g,k=0.15,)

plt.figure(figsize =(20, 20)) 

nx.draw_networkx(g,pos,alpha=0.8,node_color='red',node_size=25,font_size=9, with_label = True) 
g = nx.DiGraph()

for x in range(1500,2000):

    g.add_edge(train['location'][x], train['keyword'][x]) 

    

pos=nx.spring_layout(g,k=0.15,)

plt.figure(figsize =(20, 20)) 

nx.draw_networkx(g,pos,alpha=0.8,node_color='red',node_size=25,font_size=9, with_label = True) 
g = nx.DiGraph()

for x in range(2000,2500):

    g.add_edge(train['location'][x], train['keyword'][x]) 

    

pos=nx.spring_layout(g,k=0.15,)

plt.figure(figsize =(20, 20)) 

nx.draw_networkx(g,pos,alpha=0.8,node_color='red',node_size=25,font_size=9, with_label = True) 
g = nx.DiGraph()

for x in range(2500,3000):

    g.add_edge(train['location'][x], train['keyword'][x]) 

    

pos=nx.spring_layout(g,k=0.15,)

plt.figure(figsize =(20, 20)) 

nx.draw_networkx(g,pos,alpha=0.8,node_color='red',node_size=25,font_size=9, with_label = True) 
# Lets try out PorterStemmer first

stemmer_ = PorterStemmer()

print("The stemmed form of running is: {}".format(stemmer_.stem("running")))

print("The stemmed form of runs is: {}".format(stemmer_.stem("runs")))

print("The stemmed form of run is: {}".format(stemmer_.stem("run")))
# And then Lemmatization

lemm = WordNetLemmatizer()

print("I  case of Lemmatization, running is: {}".format(lemm.lemmatize("running")))

print("I  case of Lemmatization, runs is: {}".format(lemm.lemmatize("runs")))

print("I  case of Lemmatization, is: {}".format(lemm.lemmatize("run")))
def pre_Process_data(documents):

    '''

    For preprocessing we have regularized, transformed each upper case into lower case, tokenized,

    Normalized and remove stopwords. For normalization, we have used PorterStemmer. Porter stemmer transforms 

    a sentence from this "love loving loved" to this "love love love"

    

    '''

    STOPWORDS = set(stopwords.words('english'))

    stemmer = PorterStemmer()

    #lemm = WordNetLemmatizer()

    Tokenized_Doc=[]

    print("Pre-Processing the Data.........\n")

    for data in documents:

        review = re.sub('[^a-zA-Z]', ' ', data)

        url = re.compile(r'https?://\S+|www\.\S+')

        review = url.sub(r'',review)

        html=re.compile(r'<.*?>')

        review = html.sub(r'',review)

        emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

        review = emoji_pattern.sub(r'',review)

        gen_docs = [w.lower() for w in word_tokenize(review)] 

        tokens = [stemmer.stem(token) for token in gen_docs if not token in STOPWORDS]

        #tokens = [lemm.lemmatize(token) for token in gen_docs if not token in STOPWORDS]

        final_=' '.join(tokens)

        Tokenized_Doc.append(final_)

    return Tokenized_Doc
def Vectorization(processed_data):

    '''

    Vectorization is an important step in Natural Language Processing. We have

    Used Tf_Idf vectorization in this script. The n_gram range for vectorization 

    lies between 2 and 3, that means minimum and maximum number of words in 

    the sequence that would be vectorized is two and three respectively. There

    are other different types of vectorization algorithms also, which could be added to this 

    function as required.

    

    '''

    vectorizer = TfidfVectorizer(stop_words='english', 

                                    #max_features= 200000, # keep top 200000 terms 

                                    min_df = 3, ngram_range=(2,3),

                                    smooth_idf=True)

    X = vectorizer.fit_transform(processed_data)

    print("\n Shape of the document-term matrix")

    print(X.shape) # check shape of the document-term matrix

    return X, vectorizer
def topic_modeling(model,X):

    '''

    We have used three types of decomposition algorithm for unsupervised learning, anyone could 

    be selected with the help of the "model" parameter. Three of them are TruncatedSVD ,Latent

    Dirichlet Allocation and Matrix Factorization. This function is useful for comparing

    different model performances, by switching between different algorithms with the help of 

    the "model" parameter and also more algorithms could be easily added to this function.

    

    '''

    components = 900

    if model=='svd':

        print("\nTrying out Truncated SVD......")

        model_ = TruncatedSVD(n_components=components,n_iter=20)

        model_.fit(X)

    if model=='MF':

        print("\nTrying out Matrix Factorization......")

        model_ = NMF(n_components=components, random_state=1,solver='mu',

                      beta_loss='kullback-leibler', alpha=.1,max_iter=20,

                      l1_ratio=.5).fit(X)

        model_.fit(X)

    if model=='LDA':

        print("\nTrying out Latent Dirichlet Allocation......")

        #Tokenized_Doc=[doc.split() for doc in processed_data]

        #dictionary = Dictionary(Tokenized_Doc)

        #corpus = [dictionary.doc2bow(tokens) for tokens in Tokenized_Doc]

        #model_ = LdaModel(corpus, num_topics=components, id2word = dictionary)

        model_ = LatentDirichletAllocation(n_components=components,n_jobs=-1,

                                           max_iter=20,

                                           random_state=42,verbose=0

                                          )

        model_.fit(X)

    if model=='k-means':

        print("\nTrying out K-Means clustering......")

        true_k = 2

        model_ = KMeans(n_clusters=components, init='k-means++',max_iter=20, n_init=1)

        model_.fit(X)

        

    X = model_.transform(X)

    

    scl = preprocessing.StandardScaler()

    scl.fit(X)

    x_scl = scl.transform(X)



    return x_scl
def Visualize_clusters(X_topics, title):

    '''

    This function is used to visualize the clusters generated by our 

    model through unsupervised learning. We have used UMAP for better 

    visualization of clusters.

    

    '''

    #embedding = umap.UMAP(n_neighbors=30,

    #                        min_dist=0.0,

    #                        n_components=2,).fit_transform(X_topics)#20

    embedding = TSNE(n_components=2, 

                     verbose=1, random_state=0, angle=0.75).fit_transform(X_topics)

    

    plt.figure(figsize=(20,20))

    plt.gca().set_aspect('equal', 'datalim')

    #plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))

    plt.title(title,fontsize=16)

    plt.scatter(embedding[:, 0], embedding[:, 1], 

    c = train['target'], cmap='Spectral', alpha=0.7,

    s = 20, # size

    )

    plt.show()
processed_data = pre_Process_data(train['text'])
X, vectorizer = Vectorization(processed_data)
X_transform_1 = topic_modeling('svd',X)
Visualize_clusters(X_transform_1, "Clustering for Truncated SVD")
#X, vectorizer = Vectorization(processed_data)

X_transform_2 = topic_modeling('LDA',X)

Visualize_clusters(X_transform_2, "Clustering for LDA")
X_transform_3 = topic_modeling('MF',X)

Visualize_clusters(X_transform_3, "Clustering for MF")
X_transform_4 = topic_modeling('k-means',X)

Visualize_clusters(X_transform_4, "Clustering for K-Means")
y = train['target']

#xtrain, xvalid, ytrain, yvalid = train_test_split(X_transform_3, y, 

#                                                  stratify=y, 

#                                                  random_state=42, 

#                                                  shuffle=True)
def evaluate_performance(model):

    kf = StratifiedKFold(n_splits=3,shuffle=True,random_state=42)

    i=1

    model_list=[]

    for train_index,test_index in kf.split(X_transform_3, y):

        print('{} of KFold {}'.format(i,kf.n_splits))

        xtrain,xvalid = X_transform_3[train_index],X_transform_3[test_index]

        ytrain,yvalid = y[train_index],y[test_index]

        model.fit(xtrain, ytrain)

        predictions = model.predict(xvalid)

        print("Accuracy score: "+str(accuracy_score(predictions,yvalid)))

        print("ROC score: "+str(roc_auc_score(predictions,yvalid)))

        print("Log Loss: "+ str(log_loss(predictions,yvalid)))

        i=i+1

        model_list.append(model)

    return model_list
clf = lgb.LGBMClassifier(max_depth=12,

                             learning_rate=0.5,

                             n_estimators = 1000,

                             subsample=0.25,

                           )

model_1 = evaluate_performance(clf)
clf = xgb.XGBClassifier(max_depth=12, n_estimators=600, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.5)

model_2 = evaluate_performance(clf)
clf = LogisticRegression()

model_3 = evaluate_performance(clf)
clf = KNeighborsClassifier(n_neighbors=2)

model_4 = evaluate_performance(clf)
clf = ExtraTreesClassifier(n_estimators=600,max_depth=12)

model_5 = evaluate_performance(clf)
clf = CatBoostClassifier(max_depth=12, n_estimators=600, learning_rate=0.5,verbose=0)

model_6 = evaluate_performance(clf)
processed_data = pre_Process_data(test['text'])

X, vectorizer = Vectorization(processed_data)

X_transform_1 = topic_modeling('MF',X)
prob=model_5[2].predict(X_transform_1)
my_submission = pd.DataFrame({'id': test['id'], 'target': prob})

my_submission.to_csv('SubmissionVictor.csv', index=False)

my_submission.head()