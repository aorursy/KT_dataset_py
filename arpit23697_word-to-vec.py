import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups

from pprint import pprint



#using nltk

from nltk import word_tokenize

from nltk import download

from nltk.corpus import stopwords



#import gensim

from gensim.test.utils import common_texts, get_tmpfile

from gensim.models import Word2Vec



#for the tsne model

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

from sklearn import metrics
#download tokenizer and stopwords

#run this cell only once

download('punkt')

download('stopwords')

stopwords = stopwords.words('english')
#download the dataset

newsgroups_train = fetch_20newsgroups(subset='all' , 

                                     remove = ('headers' , 'footers' , 'quotes'))
def preprocess (text):

    text = text.lower()

    doc = word_tokenize(text)               #tokenize the text

    doc = [word for word in doc if word not in stopwords]              #removing stop words

    doc = [word for word in doc if word.isalpha()]                     #Containing only alphanumeric characters

    return doc
print(newsgroups_train.data[0])
corpus = [preprocess(text) for text in  newsgroups_train.data]
#removing the empty docs

def filter_docs (corpus , texts , labels):

    number_of_docs = len(corpus)

    if texts is not None:

        texts = [text for (text , doc) in zip(texts , corpus) if len(doc) != 0]

    

    labels = [i for (i , doc) in zip(labels , corpus) if len(doc) != 0]

    corpus = [doc for doc in corpus if (len(doc) != 0)]

    print("{} docs removed".format(number_of_docs - len(corpus)))

    return corpus , texts , labels    
corpus , texts , labels = filter_docs(corpus , newsgroups_train.data , newsgroups_train.target)
model = Word2Vec(corpus , min_count = 10 , size = 100)
#vocabulary size

len(model.wv.vocab)
model.wv['computer'].shape
model.wv.most_similar(positive = ['car'] , topn = 5)
model.wv.most_similar(positive = ['sports' , 'ball'] , negative = ['bat'] , topn = 5)
model.wv.most_similar(positive = ['girl' , 'father'] , negative = ['boy'] , topn = 5)
words_to_consider = ['baseball' , 'software' , 'police' , 'government' , 'circuit' , 'car']

tsne_data = {}

for i in range(len(words_to_consider)):

    tsne_data[words_to_consider[i]] =  [x for x , y in model.wv.most_similar(positive = [words_to_consider[i]] , topn = 20)]

    
#creating the whole dataset

tsne_data_plot = []

words_plot = []

for x,y in tsne_data.items():

    tsne_data_plot.append(model.wv[x])

    words_plot.append(x)

    words_plot.extend(y)

    for j in y:

        tsne_data_plot.append(model.wv[j])

len(tsne_data_plot)
#First applying PCA to reduce to 20 components and then use tsne to further reduce to 2 components

tsne = TSNE (n_components=2 ,random_state = 42)

pca = PCA(n_components=20)

pca.fit(tsne_data_plot)

data_pca = pca.transform(tsne_data_plot)

data_tsne = tsne.fit_transform(data_pca)
#plotting the graph

colors = ['red' , 'green' , 'blue' , 'black' , 'orange' , 'purple']

plt.figure(figsize = (10 , 10))

plt.xlim (data_tsne[: , 0].min() - 1, data_tsne[: , 0].max() + 1)

plt.ylim (data_tsne[: , 1].min() - 1 , data_tsne[: , 1].max() + 1)

for i in range(6):

    down = 21 * i

    up = 21 * (i+1)

    plt.scatter(data_tsne[down:up , 0] , data_tsne[down:up, 1] , c = colors[i] , label = words_to_consider[i] )

    for label , x, y in zip (words_plot[down:up] , data_tsne[down:up , 0] , data_tsne[down:up , 1]):

        plt.annotate(label , xy = (x , y) , xytext = (0 , 0) , textcoords = 'offset points')



plt.legend()

plt.xticks([])

plt.yticks([])

plt.title("Analysis")

plt.show()
def wordTodoc (embedding_type = 'min_max'):

    '''

    It will return an array  (number_of_docs * 100)  

    '''

    # data : contains the doc embeddings

    # target : contains the corresponding embedding

    embedding_size = 200

    if (embedding_type == 'avg'):

        embedding_size = 100

    elif (embedding_type == 'min_max'):

        embedding_size = 200

    elif (embedding_type == 'min_max_add'):

        embedding_size = 100

    elif (embedding_type == 'avg_std'):

        embedding_size =200

    elif (embedding_type == 'all'):

        embedding_size = 400

    

    

    data = np.zeros((0, embedding_size))

    target = []

    number_of_docs = len(corpus)

    

    

    for i in range(number_of_docs):

        

        #get the document

        doc = corpus[i]

        doc_all_embeddings = np.zeros((0,100))

        

        #collect all the embeddings

        for words in doc:

            if words not in model.wv.vocab:

                continue 

            doc_all_embeddings = np.vstack([ doc_all_embeddings, model.wv[words].reshape(1 , 100)            ]) 

    

        #if no word in doc remove it from data

        if (doc_all_embeddings.shape[0] == 0):

            continue

        

        #compute the doc embedding

        

        doc_embedding = (np.vstack([doc_all_embeddings.min(axis = 0).reshape(100 , 1) , doc_all_embeddings.max(axis = 0).reshape(100 , 1)])).reshape(1,200)

        if (embedding_type == 'avg'):

            doc_embedding = doc_all_embeddings.mean(axis = 0).reshape(1, 100)

        elif (embedding_type == 'min_max'):

            doc_embedding = (np.vstack([doc_all_embeddings.min(axis = 0).reshape(100 , 1) , doc_all_embeddings.max(axis = 0).reshape(100 , 1)])).reshape(1,200)

        elif (embedding_type == 'min_max_add'):

            doc_embedding = doc_all_embeddings.min(axis = 0).reshape(1 , 100) + doc_all_embeddings.max(axis = 0).reshape(1 , 100)

        elif (embedding_type == 'avg_std'):

            doc_embedding = (np.vstack([doc_all_embeddings.mean(axis = 0).reshape(100 , 1) , doc_all_embeddings.std(axis = 0).reshape(100 , 1)])).reshape(1,200)

        elif(embedding_type == 'all'):

            doc_embedding = (np.vstack([doc_all_embeddings.mean(axis = 0).reshape(100 , 1), 

                                        doc_all_embeddings.std(axis = 0).reshape(100 , 1) ,

                                        doc_all_embeddings.min(axis = 0).reshape(100 , 1),

                                        doc_all_embeddings.max(axis = 0).reshape(100 , 1),

                                       ])).reshape(1,400)

        

        

        #put the embeddings into the data and the target

        data = np.vstack([data , doc_embedding])

        target.append(labels[i])

    

    print("Shape of data {}".format(data.shape))

    print("Shape of labels {}".format(len(target)))

    return data , np.array(target)
data , target = wordTodoc(embedding_type='avg_std')
from sklearn.model_selection import train_test_split

X_train , X_val , y_train , y_val = train_test_split(data , target , test_size = 0.2 , random_state = 42)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 20 ,  max_depth = 10 )

rf.fit(X_train, y_train)
print("Accuracy on train data using Random forest : {}".format(rf.score(X_train , y_train)))

print("Accuracy on test data using Random forest : {}".format(rf.score(X_val , y_val)))
print(metrics.classification_report(y_val, rf.predict(X_val)))
from pandas import *

DataFrame(metrics.confusion_matrix(y_val, rf.predict(X_val)))
from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder



encoder = LabelEncoder()

encoder.fit(y_train)

encoded_Y = encoder.transform(y_train)



dummy_y = np_utils.to_categorical(encoded_Y)



# define the keras model

deep_model = Sequential()

deep_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))

deep_model.add(Dense(32, activation='relu'))

deep_model.add(Dense(32, activation='relu'))



deep_model.add(Dense(20, activation='softmax'))

# compile the keras model

deep_model.compile(loss='categorical_crossentropy' ,optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset

history = deep_model.fit(X_train, dummy_y, validation_split = 0.1 ,epochs=40, batch_size=32)

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'val'], loc='upper left')

plt.show()
# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'val'], loc='upper left')

plt.show()
y_pred = deep_model.predict_classes(X_val)


print(metrics.classification_report(y_val, y_pred))
DataFrame(metrics.confusion_matrix(y_val, y_pred ))
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(solver='sag',

                         multi_class='multinomial').fit(X_train, y_train)
print("Accuracy on train data using Logistic regression : {}".format(clf.score(X_train , y_train)))

print("Accuracy on test data using Logistic regression : {}".format(clf.score(X_val , y_val)))
print(metrics.classification_report(y_val, clf.predict(X_val)))
DataFrame(metrics.confusion_matrix(y_val, clf.predict(X_val)))