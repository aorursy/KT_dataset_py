import pandas as pd

import seaborn as sns

#df = pd.read_csv('../../dataset/training/train_lyrics_1000.csv', usecols=range(7))



df.head()# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings('ignore')
import pandas as pd

import seaborn as sns

df = pd.read_csv('../input/train_lyrics_1000.csv', usecols=range(7))

df1 =pd.read_csv('../input/valid_lyrics_200.csv')

print(df1.head())

df.head()

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn import decomposition, ensemble



import pandas, xgboost, numpy, textblob, string

from keras.preprocessing import text, sequence

from keras import layers, models, optimizers
train_x, valid_x, train_y, valid_y = df['lyrics'],df1['lyrics'],df['mood'],df1['mood']

encoder = preprocessing.LabelEncoder()

train_y = encoder.fit_transform(train_y)

valid_y = encoder.fit_transform(valid_y)
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')

count_vect.fit(df['lyrics'])



# transform the training and validation data using count vectorizer object

xtrain_count =  count_vect.transform(train_x)

xvalid_count =  count_vect.transform(valid_x)
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)

tfidf_vect.fit(df['lyrics'])

xtrain_tfidf =  tfidf_vect.transform(train_x)

xvalid_tfidf =  tfidf_vect.transform(valid_x)



# ngram level tf-idf 

tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

tfidf_vect_ngram.fit(df['lyrics'])

xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)

xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)



# characters level tf-idf

tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

tfidf_vect_ngram_chars.fit(df['lyrics'])

xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 

xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 
lda_model = decomposition.LatentDirichletAllocation(n_components=20, learning_method='online', max_iter=20)

X_topics = lda_model.fit_transform(xtrain_count)

topic_word = lda_model.components_ 

vocab = count_vect.get_feature_names()



# view the topic models

n_top_words = 10

topic_summaries = []

for i, topic_dist in enumerate(topic_word):

    topic_words = numpy.array(vocab)[numpy.argsort(topic_dist)][:-(n_top_words+1):-1]

    topic_summaries.append(' '.join(topic_words))
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):

    # fit the training dataset on the classifier

    classifier.fit(feature_vector_train, label)

    

    # predict the labels on validation dataset

    predictions = classifier.predict(feature_vector_valid)

    

    if is_neural_net:

        predictions = predictions.argmax(axis=-1)

    

    return metrics.accuracy_score(predictions, valid_y)
# Naive Bayes on Count Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)

print ("NB, Count Vectors: ", accuracy)



# Naive Bayes on Word Level TF IDF Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)

print ("NB, WordLevel TF-IDF: ", accuracy)



# Naive Bayes on Ngram Level TF IDF Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)

print ("NB, N-Gram Vectors: ", accuracy)



# Naive Bayes on Character Level TF IDF Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)

print ("NB, CharLevel Vectors: ", accuracy)
accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)

print ("LR, Count Vectors: ", accuracy)



# Linear Classifier on Word Level TF IDF Vectors

accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)

print ("LR, WordLevel TF-IDF: ", accuracy)



# Linear Classifier on Ngram Level TF IDF Vectors

accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)

print ("LR, N-Gram Vectors: ", accuracy)



# Linear Classifier on Character Level TF IDF Vectors

accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)

print ("LR, CharLevel Vectors: ", accuracy)
accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)

print ("SVM, N-Gram Vectors: ", accuracy)
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)

print ("RF, Count Vectors: ", accuracy)



# RF on Word Level TF IDF Vectors

accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)

print ("RF, WordLevel TF-IDF: ", accuracy)
# Extereme Gradient Boosting on Count Vectors

accuracy = train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y, xvalid_count.tocsc())

print ("Xgb, Count Vectors: ", accuracy)



# Extereme Gradient Boosting on Word Level TF IDF Vectors

accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xvalid_tfidf.tocsc())

print ("Xgb, WordLevel TF-IDF: ", accuracy)



# Extereme Gradient Boosting on Character Level TF IDF Vectors

accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram_chars.tocsc(), train_y, xvalid_tfidf_ngram_chars.tocsc())

print ("Xgb, CharLevel Vectors: ", accuracy)
def create_model_architecture(input_size):

    # create input layer 

    input_layer = layers.Input((input_size, ), sparse=True)

    

    # create hidden layer

    hidden_layer = layers.Dense(100, activation="relu")(input_layer)

    

    # create output layer

    output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)



    classifier = models.Model(inputs = input_layer, outputs = output_layer)

    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return classifier 



classifier = create_model_architecture(xtrain_tfidf_ngram.shape[1])

accuracy = train_model(classifier, xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, is_neural_net=True)

print ("NN, Ngram Level TF IDF Vectors",  accuracy)