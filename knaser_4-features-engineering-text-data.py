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
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string

from keras.preprocessing import text, sequence



from keras import layers, models, optimizers
# load the dataset

data = open('../input/corpus.txt').read()

labels, texts = [], []

for i, line in enumerate(data.split("\n")):

    content = line.split()

    labels.append(content[0])

    texts.append(" ".join(content[1:]))



# create a dataframe using texts and lables

trainDF = pandas.DataFrame()

trainDF['text'] = texts

trainDF['label'] = labels
# split the dataset into training and validation datasets 

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])



# label encode the target variable 

encoder = preprocessing.LabelEncoder()

train_y = encoder.fit_transform(train_y)

valid_y = encoder.fit_transform(valid_y)
# create a count vectorizer object 

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')

count_vect.fit(trainDF['text'])



# transform the training and validation data using count vectorizer object

xtrain_count =  count_vect.transform(train_x)

xvalid_count =  count_vect.transform(valid_x)
# word level tf-idf

#  Word Level TF-IDF : Matrix representing tf-idf scores of every term in different documents



tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)

tfidf_vect.fit(trainDF['text'])

xtrain_tfidf =  tfidf_vect.transform(train_x)

xvalid_tfidf =  tfidf_vect.transform(valid_x)

# N-gram Level TF-IDF : N-grams are the combination of N terms together. This Matrix representing tf-idf scores of N-grams



# ngram level tf-idf 

tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

tfidf_vect_ngram.fit(trainDF['text'])

xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)

xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# Character Level TF-IDF : Matrix representing tf-idf scores of character level n-grams in the corpus

# characters level tf-idf

tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

tfidf_vect_ngram_chars.fit(trainDF['text'])

xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 

xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 
# load the pre-trained word-embedding vectors 

embeddings_index = {}

for i, line in enumerate(open('../input/corpus.txt')):

    values = line.split()

    #embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')



# create a tokenizer 

token = text.Tokenizer()

token.fit_on_texts(trainDF['text'])

word_index = token.word_index



# convert text to sequence of tokens and pad them to ensure equal length vectors 

train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)

valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)



# create token-embedding mapping

embedding_matrix = numpy.zeros((len(word_index) + 1, 300))

for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
trainDF['char_count'] = trainDF['text'].apply(len)

trainDF['word_count'] = trainDF['text'].apply(lambda x: len(x.split()))

trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count']+1)

trainDF['punctuation_count'] = trainDF['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 

trainDF['title_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))

trainDF['upper_case_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
# train a LDA Model

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

print ('NB, Count Vectors:', accuracy)



# Naive Bayes on Word Level TF IDF Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)

print ('NB, WordLevel TF-IDF:', accuracy)



# Naive Bayes on Ngram Level TF IDF Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)

print ('NB, N-Gram Vectors: ', accuracy)



# Naive Bayes on Character Level TF IDF Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)

print ('NB, CharLevel Vectors: ', accuracy)

# Linear Classifier on Count Vectors

accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)

print ('LR, Count Vectors:  ', accuracy)



# Linear Classifier on Word Level TF IDF Vectors

accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)

print ('LR, WordLevel TF-IDF: ', accuracy)



# Linear Classifier on Ngram Level TF IDF Vectors

accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)

print ('LR, N-Gram Vectors: ', accuracy)



# Linear Classifier on Character Level TF IDF Vectors

accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)

print ('LR, CharLevel Vectors:', accuracy)

# SVM on Ngram Level TF IDF Vectors

accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)

print ('SVM, N-Gram Vectors:', accuracy)

# RF on Count Vectors

accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)

print ('RF, Count Vectors: ', accuracy)



# RF on Word Level TF IDF Vectors

accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)

print ('RF, WordLevel TF-IDF:', accuracy)

# Extereme Gradient Boosting on Count Vectors

accuracy = train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y, xvalid_count.tocsc())

print ('Xgb, Count Vectors:', accuracy)



# Extereme Gradient Boosting on Word Level TF IDF Vectors

accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xvalid_tfidf.tocsc())

print ('Xgb, WordLevel TF-IDF:', accuracy)



# Extereme Gradient Boosting on Character Level TF IDF Vectors

accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram_chars.tocsc(), train_y, xvalid_tfidf_ngram_chars.tocsc())

print ('Xgb, CharLevel Vectors:', accuracy)

def create_cnn():

    # Add an Input Layer

    input_layer = layers.Input((70, ))



    # Add the word embedding Layer

    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)

    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)



    # Add the convolutional Layer

    conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)



    # Add the pooling Layer

    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)



    # Add the output Layers

    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)

    output_layer1 = layers.Dropout(0.25)(output_layer1)

    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)



    # Compile the model

    model = models.Model(inputs=input_layer, outputs=output_layer2)

    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    

    return model



classifier = create_cnn()

accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)

print ("CNN, Word Embeddings",  accuracy)