# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Libraries for data set preparation feature engineering model training

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn import decomposition, ensemble



import pandas, xgboost, numpy, textblob, string

from keras.preprocessing import text, sequence

from keras import layers, models, optimizers
#load the dataset

data = open("../input/corpus").read()

labels, texts = [],  []



for i, line in enumerate(data.split("\n")):

    content = line.split()

    labels.append(content[0])

    texts.append(" ".join(content[1:]))
#create a dataframe using texts and lables

trainDF = pandas.DataFrame()

trainDF['text'] = texts

trainDF['label'] = labels
#split the dataset into training and validation datasets

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])
#label encode the target variable

encoder = preprocessing.LabelEncoder()

train_y = encoder.fit_transform(train_y)

valid_y = encoder.fit_transform(valid_y)
#create a count vectorizer object

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')

count_vect.fit(trainDF['text'])
#transform the training and validation data using count vectorizer object

xtrain_count = count_vect.transform(train_x)

xvalid_count = count_vect.transform(valid_x)
#word level tf-idf

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)

tfidf_vect.fit(trainDF['text'])

xtrain_tfidf = tfidf_vect.transform(train_x)

xvalid_tfidf = tfidf_vect.transform(valid_x)



# ngram level tf-idf 

tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

tfidf_vect_ngram.fit(trainDF['text'])

xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)

xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)



# characters level tf-idf

tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

tfidf_vect_ngram_chars.fit(trainDF['text'])

xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 

xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 
trainDF['char_count'] = trainDF['text'].apply(len)

trainDF['word_count'] = trainDF['text'].apply(lambda x: len(x.split()))

trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count']+1)

trainDF['punctuation_count'] = trainDF['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 

trainDF['title_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))

trainDF['upper_case_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
pos_family = {

    'noun' : ['NN','NNS','NNP','NNPS'],

    'pron' : ['PRP','PRP$','WP','WP$'],

    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],

    'adj' :  ['JJ','JJR','JJS'],

    'adv' : ['RB','RBR','RBS','WRB']

}



# function to check and get the part of speech tag count of a words in a given sentence

def check_pos_tag(x, flag):

    cnt = 0

    try:

        wiki = textblob.TextBlob(x)

        for tup in wiki.tags:

            ppo = list(tup)[1]

            if ppo in pos_family[flag]:

                cnt += 1

    except:

        pass

    return cnt



trainDF['noun_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'noun'))

trainDF['verb_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'verb'))

trainDF['adj_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adj'))

trainDF['adv_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adv'))

trainDF['pron_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'pron'))
#train a LDA Model

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

print ("NB, Count Vectors: "), accuracy



# Naive Bayes on Word Level TF IDF Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)

print ("NB, WordLevel TF-IDF: "), accuracy



# Naive Bayes on Ngram Level TF IDF Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)

print ("NB, N-Gram Vectors: "), accuracy



# Naive Bayes on Character Level TF IDF Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)

print ("NB, CharLevel Vectors: "), accuracy