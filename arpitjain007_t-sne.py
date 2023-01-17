# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import random

import re

import nltk

import time

import plotly.express as px

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from sklearn.feature_selection import SelectKBest , f_classif

from tqdm import tqdm as t

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from gensim.models import Word2Vec

from nltk.corpus import stopwords

def load_dataset(datapath, seed=123):

    

    imdb_data_path = os.path.join(datapath , 'aclImdb')

    

    train_texts=[]

    train_labels=[]

    test_texts=[]

    test_labels=[]

    

    # Loading Training Data

    print("[INFO] Loading training data....")

    for category in ['pos' , 'neg']:

        train_path = os.path.join(imdb_data_path , 'train' , category)

        for fname in sorted(os.listdir(train_path)):

            if fname.endswith('.txt'):

                with open(os.path.join(train_path , fname)) as f:

                    train_texts.append(f.read())

                train_labels.append(0 if category=='neg' else 1)

                

    # Loading Test Data

    print("[INFO] Loading test data....")

    for category in ['pos' , 'neg']:

        test_path = os.path.join(imdb_data_path , 'test' , category)

        for fname in sorted(os.listdir(test_path)):

            if fname.endswith('.txt'):

                with open(os.path.join(test_path , fname)) as f:

                    test_texts.append(f.read())

                test_labels.append(0 if category=='neg' else 1)

    

    print("[INFO] datasets loaded....")

    # Shuffle the training data and labels.

    random.seed(seed)

    random.shuffle(train_texts)

    random.seed(seed)

    random.shuffle(train_labels)

    random.seed(seed)

    random.shuffle(test_texts)

    random.seed(seed)

    random.shuffle(test_labels)

    

    return ((np.array(train_texts) , np.array(train_labels)) ,

            (np.array(test_texts) , np.array(test_labels)))
datapath = "../input/aclimdb/"

(X_train , y_train ) , (X_test , y_test) = load_dataset(datapath) 



print("The shape of training data:", X_train.shape)

print("The shape of training labels:", y_train.shape)

print("The shape of test data:" , X_test.shape)

print("The shape of test labels" , y_test.shape)
plt.figure(figsize=(15,6))

plt.hist([len(s) for s in X_train] , 50)

plt.xlabel("Length of sample")

plt.ylabel('Number of sample')

plt.grid()

plt.show()
kwargs ={ "ngram_range": (1,1)

        }

# instantiate the object 

count_vec = CountVectorizer(**kwargs)

vect_mat = count_vec.fit_transform(list(X_train))     # fit the method on the data

all_gram_names = count_vec.get_feature_names()        #Getting all the feature names

all_counts= vect_mat.sum(axis=0).tolist()[0]      # list of all the counts of n_gram
all_ngrams, all_counts = zip(*[(n, c) for c, n in sorted(zip(all_counts, all_gram_names), reverse=True)]) #unzipping 

ngrams = all_ngrams[:50]

counts = all_counts[:50]


idx = np.arange(50)

plt.figure(figsize=(15,10))

plt.bar(idx , counts , width=0.8)

plt.xlabel("N-grams")

plt.ylabel("Frequency of Words")

plt.xticks(idx , ngrams , rotation=45)

plt.grid()

plt.show()
def tsne_plot(x, y , c):

    kwargs ={

        "n_components": 3,

        'perplexity': 30,

        "learning_rate": 200

        }

    start = time.time()

    print("[INFO] starts...")

    tsne = TSNE(**kwargs)

    cat = ['avg' , 'weighted']

    if c not in cat:

        X_embeddings = tsne.fit_transform(x.toarray())

    else:    

        X_embeddings = tsne.fit_transform(np.array(x))

    data = np.hstack((X_embeddings , y.reshape(-1,1)))

    df = pd.DataFrame(data=data , columns=['Dim1' , 'Dim2','Dim3', 'Labels'])

    print("[INFO] ends....")

    color = { 0:"red" , 1:"blue"}

    fig =px.scatter_3d(df ,x='Dim1' , y='Dim2' ,z='Dim3', color='Labels')

    fig.show()

    print("Total time:", time.time()-start)
def selector(x,y ,n):

    print("[INFO] the shape of matrix before selection:", x.shape)

    select= SelectKBest(f_classif , k=1000)

    select.fit(x, y)

    X= select.transform(x)

    print("[INFO] Reducing the number of rows due to memory issues")

    X = X[:n,:] 

    print("[INFO] the shape of matrix after selection:", X.shape)

    return X
def textprocess(sentence):

    

    sentence = sentence.lower()

    html = re.compile("<.*?>")

    sent = re.sub(html , " " , sentence)

    filtered_list=[]

    for s in sent.split():

        clean = re.sub(r'[?|/|\'|"|!|.|,|)|(|#]' , " " , s)

        for word in clean.split():

            if word.isalpha():

                filtered_list.append(word)

            else:

                continue

    

    return filtered_list

                
def avgW2V(sentence):

    

    sent_vec = np.zeros(50)

    count=0

    for word in sent:

        try:

            wordvec = W2V.wv[word]

            sent_vec += wordvec

            count+=1

        except:

            continue

    sent_vec= sent_vec/count        

    return sent_vec
def Weighted(sentence):

    

    sent_vec = np.zeros(50)

    weighted_sum=0

    for word in sent:

        try:

            wordvec = W2V.wv[word]

            tfidff = final_tfidf[row , features.index(word)]

            sent_vec += (wordvec*tfidff)

            weighted_sum+= tfidff

        except:

            continue

    sent_vec= sent_vec/weighted_sum

    return sent_vec
countVec = CountVectorizer(ngram_range=(1,2))

final_count = countVec.fit_transform(list(X_train))
final_count = selector(final_count, y_train , 1500)
tsne_plot(final_count , y_train[:1500] , 'bagofwords' )
tfidf = TfidfVectorizer(ngram_range=(1,2))

final_tfidf = tfidf.fit_transform(X_train)

features = tfidf.get_feature_names()
final_tfidf = selector(final_tfidf , y_train , 1500)
tsne_plot(final_tfidf , y_train[:1500] , 'tfidf')
list_of_sent=[]

for sent in t(X_train):

    filtered_list = textprocess(sent)

    list_of_sent.append(filtered_list)
print(X_train[0])

print("\n", list_of_sent[0])
W2V = Word2Vec(list_of_sent , min_count=5 , size=50 , workers=4)
W2V.wv.most_similar("fat")
W2V.wv.most_similar('man')
sent_vectors =[]

for sent in t(list_of_sent):

    sent_vec= avgW2V(sent)

    sent_vectors.append(sent_vec)

#sent_vectors= np.array(sent_vectors)
tsne_plot(sent_vectors[:1500] , y_train[:1500] , 'avg')
row = 0

weighted_vectors =[]



for sent in t(list_of_sent[:10]):

    sent_vec = Weighted(sent)

    weighted_vectors.append(sent_vec)

    row+=1