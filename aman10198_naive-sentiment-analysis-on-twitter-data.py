# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #plotting graphs

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import re

import nltk

from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords, brown

from sklearn.metrics import recall_score, f1_score, accuracy_score, confusion_matrix, precision_score
dataset = pd.read_csv('../input/sentiment-analysis-data/training_data.csv')

dataset.drop(columns = 'Unnamed: 0', axis = 1, inplace = True)
extra_data = pd.read_csv('../input/data-for-train/X_train2.csv')

extra_data.drop(columns = 'Unnamed: 0', axis = 1, inplace = True)

dataset = pd.concat([dataset, extra_data], axis = 0, ignore_index= True, verify_integrity=True)
dataset.head()
value_freq = dataset['label'].value_counts()

print(value_freq)
aux = plt.bar(x = [1,2,0], height = value_freq, )

plt.show()
# removing stopword from each row of the dataset

def preprocess(text):

    text = re.sub('[^a-zA-Z]', ' ', text)

    text = text.lower()

    text = text.split()

    ps = PorterStemmer()

    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]

    text = ' '.join(text)

    return text
corpus = []

for i in range(dataset.shape[0]):

    corpus.append(preprocess(dataset['text'][i]))
#file = open('../input/bag_of_words.txt')

#bag_of_words = file.read().split(' ')

#file.close()
#creating a bag of words of 1000 most frequently occuring word and then converting data to array on its basis.

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

tf = TfidfVectorizer(ngram_range = (1,2), stop_words= ['http', 'appl','aapl', 'co', 'tweet', 'rt','amp', 'iphon'], max_features = 1500)
# fitting and creating the transformation of corpus to array

tf.fit(set(corpus))

X = tf.transform(corpus).toarray()

y = dataset['label'].values
bag_of_words = tf.get_feature_names()
pos_tweet = X[y == 2]

neu_tweet = X[y == 1]

neg_tweet = X[y == 0]
#creating word cloud for positive, neutral and negtive tweets

words_count_positive = {}

words_count_neutral = {}

words_count_negative = {}



i = 0

while i <X.shape[1]:

    words_count_positive[bag_of_words[i]] = np.sum(pos_tweet[:,i])

    words_count_neutral[bag_of_words[i]] = np.sum(neu_tweet[:,i])

    words_count_negative[bag_of_words[i]] = np.sum(neg_tweet[:,i])

    

    i = i+1
from wordcloud import WordCloud

wc_pos = WordCloud(background_color= 'white').generate_from_frequencies(words_count_positive)

wc_neu = WordCloud(background_color= 'white').generate_from_frequencies(words_count_neutral)

wc_neg = WordCloud(background_color= 'white').generate_from_frequencies(words_count_negative)



plt.figure(figsize = (40,50))



plt.subplot('131')

plt.title('Positive',fontsize = 30)

plt.imshow(wc_pos, interpolation="bicubic")

plt.axis("off")



plt.subplot('132')

plt.title('Neutral',fontsize = 30)

plt.imshow(wc_neu, interpolation="bicubic")

plt.axis("off")



plt.subplot('133')

plt.title('Negative',fontsize = 30)

plt.imshow(wc_neg, interpolation="bicubic")

plt.axis("off")



plt.show()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.15, random_state = 32)
def model_info_train(classifier, data_X, data_Y, name):

    # training set

    print("****training_set****")

    print("Model is <<"+name+">>")

    data_Y_pred = classifier.predict(data_X)

    print("accuracy:",accuracy_score(data_Y, data_Y_pred))

    print("f1 score:", f1_score(data_Y,data_Y_pred, average = "macro")) # unweighted mean,i.e. does not take label imbalance in account

    print("recall:", recall_score(data_Y,data_Y_pred, average= "macro"))

    print("precision:", precision_score(data_Y, data_Y_pred, average = "macro"))

    

    # plotting

    plt.figure(figsize = (10,4))

    ax1 = plt.subplot('121', )

    plt.title("X_train")

    train_freq = {}

    train_freq['positive'] = np.sum((data_Y == 2)*1)

    train_freq['neutral'] = np.sum((data_Y == 1)*1)

    train_freq['negative'] = np.sum((data_Y == 0)*1)

    

    lists = sorted(train_freq.items()) # sorted by key, return a list of tuples

    x, y = zip(*lists)

    plt.bar(x, y)

    

    plt.subplot('122', sharex = ax1, sharey = ax1)

    plt.title("X_train predict")

    train_pred_freq = {}

    train_pred_freq['positive'] = np.sum((data_Y_pred == 2)*1)

    train_pred_freq['neutral'] = np.sum((data_Y_pred == 1)*1)

    train_pred_freq['negative'] = np.sum((data_Y_pred == 0)*1)

    

    lists = sorted(train_pred_freq.items()) # sorted by key, return a list of tuples

    x, y = zip(*lists)

    plt.bar(x, y)

    

    
def model_info_validation(classifier, data_X, data_Y, name):

    # validate set

    print("****validate_set****")

    print("Model is <<"+name+">>")

    data_Y_pred = classifier.predict(data_X)

    print("accuracy:",accuracy_score(data_Y, data_Y_pred))

    print("f1 score:", f1_score(data_Y,data_Y_pred, average = "macro")) # unweighted mean,i.e. does not take label imbalance in account

    print("recall:", recall_score(data_Y,data_Y_pred, average= "macro"))

    print("precision:", precision_score(data_Y, data_Y_pred, average = "macro"))

    

    # plotting

    plt.figure(figsize = (10,4))

    ax1 = plt.subplot('121')

    plt.title("X_validate")

    validate_freq = {}

    validate_freq['positive'] = np.sum((data_Y == 2)*1)

    validate_freq['neutral'] = np.sum((data_Y == 1)*1)

    validate_freq['negative'] = np.sum((data_Y == 0)*1)

   

    lists = sorted(validate_freq.items()) # sorted by key, return a list of tuples

    x, y = zip(*lists)

    plt.bar(x, y)

    

    plt.subplot('122', sharex = ax1, sharey = ax1)

    plt.title("X_validate predict")

    validate_pred_freq = {}

    validate_pred_freq['positive'] = np.sum((data_Y_pred == 2)*1)

    validate_pred_freq['neutral'] = np.sum((data_Y_pred == 1)*1)

    validate_pred_freq['negative'] = np.sum((data_Y_pred == 0)*1)

    

    lists = sorted(validate_pred_freq.items()) # sorted by key, return a list of tuples

    x, y = zip(*lists)

    plt.bar(x, y)
def model_info_test(classifier, data_X, data_Y, name):

    # test set

    print("****test_set****")

    print("Model is <<"+name+">>")

    data_Y_pred = classifier.predict(data_X)

    print("accuracy:",accuracy_score(data_Y, data_Y_pred))

    print("f1 score:", f1_score(data_Y,data_Y_pred, average = "macro")) # unweighted mean,i.e. does not take label imbalance in account

    print("recall:", recall_score(data_Y,data_Y_pred, average= "macro"))

    print("precision:", precision_score(data_Y, data_Y_pred, average = "macro"))

    

    # plotting

    plt.figure(figsize = (10,4))

    ax1 = plt.subplot('121')

    plt.title("X_test")

    test_freq = {}

    test_freq['positive'] = np.sum((data_Y == 2)*1)

    test_freq['neutral'] = np.sum((data_Y == 1)*1)

    test_freq['negative'] = np.sum((data_Y == 0)*1)

    lists = sorted(test_freq.items()) # sorted by key, return a list of tuples

    x, y = zip(*lists)

    plt.bar(x, y)

    

    plt.subplot('122', sharex = ax1, sharey = ax1)

    plt.title("X_test predict")

    test_pred_freq = {}

    test_pred_freq['positive'] = np.sum((data_Y_pred == 2)*1)

    test_pred_freq['neutral'] = np.sum((data_Y_pred == 1)*1)

    test_pred_freq['negative'] = np.sum((data_Y_pred == 0)*1)

    lists = sorted(test_pred_freq.items()) # sorted by key, return a list of tuples

    x, y = zip(*lists)

    plt.bar(x, y)
from sklearn.naive_bayes import MultinomialNB

classifier_naive_bayes = MultinomialNB(fit_prior = False)

classifier_naive_bayes.fit(X_train,y_train)
model_info_train(classifier_naive_bayes, X_train, y_train, 'Naive Bayes Classifier')

model_info_validation(classifier_naive_bayes, X_test, y_test, 'Naive Bayes Classifier')
from sklearn.linear_model import LogisticRegression

lg_classifier = LogisticRegression(random_state = 0, max_iter = 10000, solver = 'saga', multi_class = 'multinomial', warm_start = True, n_jobs = -1)

lg_classifier.fit(X_train, y_train)
model_info_train(lg_classifier, X_train, y_train, 'Logistic Regression')
model_info_validation(lg_classifier, X_test, y_test, 'Logistic Regression')
from sklearn.svm import SVC

svc_classifier = SVC(kernel = 'linear', random_state = 32, coef0 = 0, gamma = 'auto')

svc_classifier.fit(X_train,y_train)
model_info_train(svc_classifier, X_train, y_train, 'Support Vector Machine')
model_info_validation(svc_classifier, X_test, y_test, 'Support Vector Machine')
from sklearn.tree import DecisionTreeClassifier

dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state  = 32)

dt_classifier.fit(X_train,y_train)
model_info_train(dt_classifier, X_train, y_train, 'Decision Tree Classification')
model_info_validation(dt_classifier, X_test, y_test, 'Decision Tree Classification')
from sklearn.ensemble import RandomForestClassifier

rdt_classifier = RandomForestClassifier(n_estimators=10, criterion='entropy',

                                   n_jobs = -1, random_state = 32)

rdt_classifier.fit(X_train,y_train)
model_info_train(rdt_classifier, X_train, y_train, 'Random Forest Classification')
model_info_validation(rdt_classifier, X_test, y_test, 'Random Forest Classification')
from sklearn.linear_model import SGDClassifier

sgd_classifier = SGDClassifier(max_iter = 10000, n_jobs = -1, random_state = 32, tol = 0.00001,early_stopping = True)

sgd_classifier.fit(X_train,y_train)
model_info_train(sgd_classifier, X_train, y_train, 'Stochastic Gradient Descent Classification')
model_info_validation(sgd_classifier, X_test, y_test, 'Stochastic Gradient Descent Classification')
dataset_test = pd.read_csv("../input/sentiment-analysis-data/commercial.csv")

dataset_test.head()
dataset_test['label'] = 0

for i in range(dataset_test.shape[0]):

    if dataset_test['sentiment'][i] == 'neutral':

        dataset_test['label'][i] = 1

    elif dataset_test['sentiment'][i] == 'positive':

        dataset_test['label'][i] = 2
corpus_test = []

for i in range(dataset_test.shape[0]):

    text = re.sub('[^a-zA-Z]', ' ', dataset_test['text'][i])

    text = text.lower()

    text = text.split()

    ps = PorterStemmer()

    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]

    text = ' '.join(text)

    corpus_test.append(text)



X_comm_test = tf.transform(corpus_test).toarray()

y_comm_test = dataset_test['label'].values
# naive bayes classifier

model_info_test(classifier_naive_bayes, X_comm_test, y_comm_test, 'Naive Bayes classifier')
# logistic Regression

model_info_test(lg_classifier, X_comm_test, y_comm_test, 'logistic classifier')
# SVM classifier

model_info_test(svc_classifier, X_comm_test, y_comm_test, 'SVM classifier')
# Decision Tree classifier

model_info_test(dt_classifier, X_comm_test, y_comm_test, 'Decision Tree classifier')
# Random Forest Classifier

model_info_test(rdt_classifier, X_comm_test, y_comm_test, 'Random Forest classifier')
# Stochastic Gradient Classifier

model_info_test(sgd_classifier, X_comm_test, y_comm_test, 'Stochastic Gradient classifier')
from keras.models import Sequential

from keras.layers import Dense, Dropout
Y_train = pd.get_dummies(y_train).values
# kernel_initializerialising the ANN

classifier = Sequential()

# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 1024,kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))



# Adding the hidden layer

classifier.add(Dense(units = 1024, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dropout(0.2))

classifier.add(Dense(units = 1024, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dropout(0.2))

classifier.add(Dense(units = 512, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dropout(0.2))

classifier.add(Dense(units = 512, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dropout(0.2))

classifier.add(Dense(units = 512, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dropout(0.2))

classifier.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dropout(0.2))

classifier.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dropout(0.2))

classifier.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dropout(0.2))

classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dropout(0.2))

classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))



# Adding the output layer

classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'softmax'))



classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set

classifier.fit(X_train, Y_train, batch_size = 16,epochs = 100, verbose = 5)
print("***Training Set***")

y_train_pred = classifier.predict_classes(X_train)

print("accuracy score :",accuracy_score(y_train, y_train_pred))

print("precision score :",precision_score(y_train, y_train_pred, average = 'macro'))

print("recall score :",recall_score(y_train, y_train_pred, average = 'macro'))

print("f1 score :",f1_score(y_train, y_train_pred, average = 'macro'))



print("***Test Set***")

y_test_pred = classifier.predict_classes(X_test)

print("accuracy score :",accuracy_score(y_test, y_test_pred))

print("precision score :",precision_score(y_test, y_test_pred, average = 'macro'))

print("recall score :",recall_score(y_test, y_test_pred, average = 'macro'))

print("f1 score :",f1_score(y_test, y_test_pred, average = 'macro'))
# plotting

plt.figure(figsize = (10,4))

ax1 = plt.subplot('121')

plt.title("X_train")

validate_freq = {}

validate_freq['positive'] = np.sum((y_train == 2)*1)

validate_freq['neutral'] = np.sum((y_train == 1)*1)

validate_freq['negative'] = np.sum((y_train == 0)*1)



lists = sorted(validate_freq.items()) # sorted by key, return a list of tuples

x, y = zip(*lists)

plt.bar(x, y)



plt.subplot('122', sharex = ax1, sharey = ax1)

plt.title("X_train predict")

validate_pred_freq = {}

validate_pred_freq['positive'] = np.sum((y_train_pred == 2)*1)

validate_pred_freq['neutral'] = np.sum((y_train_pred == 1)*1)

validate_pred_freq['negative'] = np.sum((y_train_pred == 0)*1)



lists = sorted(validate_pred_freq.items()) # sorted by key, return a list of tuples

x, y = zip(*lists)

plt.bar(x, y)



plt.show()
# On commercial dataset

print("***Commercial Test Set***")

y_comm_pred = classifier.predict_classes(X_comm_test)

print("accuracy score :",accuracy_score(y_comm_test, y_comm_pred))

print("precision score :",precision_score(y_comm_test, y_comm_pred, average = 'macro'))

print("recall score :",recall_score(y_comm_test, y_comm_pred, average = 'macro'))

print("f1 score :",f1_score(y_comm_test, y_comm_pred, average = 'macro'))