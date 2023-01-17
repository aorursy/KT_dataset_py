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
# Importing necessary paackages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import re

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.stem.wordnet import WordNetLemmatizer

import string



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.pipeline import Pipeline

import xgboost as xgb

seed = 4353
data = pd.read_csv('/kaggle/input/amazon-music-reviews/Musical_instruments_reviews.csv')

data.head()
data.columns = data.columns.str.lower()

data.columns
data.isnull().sum()
print('The train dataset contans {} rows and {} columns'.format(data.shape[0], data.shape[1]))
sns.countplot(data.overall)

plt.xlabel('Overall ratings')
# replacing numerical values with categorical values to reduce the classes to sentiments



data['sentiment'] = data.overall.replace({

    1:'negative',

    2:'negative',

    3:'neutral',

    4:'positive',

    5:'positive'

})
#Creating Train and Test datasets with only the product reviews (complete Review = reviewtext + summary)



X_data = data['reviewtext'] + ' ' + data['summary']

y_data = data['sentiment']
# changing the datatype from object to string



X_data = X_data.astype(str)
# creating new dataframe



X_data_df = pd.DataFrame(data=X_data)

X_data_df.columns = ['review']

X_data_df.head()
# creating functions for text processing



string.punctuation

def final(X_data_full):

    

    # function for removing punctuations

    def remove_punct(X_data_func):

        string1 = X_data_func.lower()

        translation_table = dict.fromkeys(map(ord, string.punctuation),' ')

        string2 = string1.translate(translation_table)

        return string2

    

    X_data_full_clear_punct = []

    for i in range(len(X_data_full)):

        test_data = remove_punct(X_data_full[i])

        X_data_full_clear_punct.append(test_data)

        

    # function to remove stopwords

    def remove_stopwords(X_data_func):

        pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')

        string2 = pattern.sub(' ', X_data_func)

        return string2

    

    X_data_full_clear_stopwords = []

    for i in range(len(X_data_full)):

        test_data = remove_stopwords(X_data_full[i])

        X_data_full_clear_stopwords.append(test_data)

        

    # function for tokenizing

    def tokenize_words(X_data_func):

        words = nltk.word_tokenize(X_data_func)

        return words

    

    X_data_full_tokenized_words = []

    for i in range(len(X_data_full)):

        test_data = tokenize_words(X_data_full[i])

        X_data_full_tokenized_words.append(test_data)

        

    # function for lemmatizing

    lemmatizer = WordNetLemmatizer()

    def lemmatize_words(X_data_func):

        words = lemmatizer.lemmatize(X_data_func)

        return words

    

    X_data_full_lemmatized_words = []

    for i in range(len(X_data_full)):

        test_data = lemmatize_words(X_data_full[i])

        X_data_full_lemmatized_words.append(test_data)

        

    # creating the bag of words model

    cv = CountVectorizer(max_features=1000)

    X_data_full_vector = cv.fit_transform(X_data_full_lemmatized_words).toarray()

    

    

    tfidf = TfidfTransformer()

    X_data_full_tfidf = tfidf.fit_transform(X_data_full_vector).toarray()

    

    return X_data_full_tfidf

    
# running the function



data_X = final(X_data)

X_train, X_test, y_train, y_test = train_test_split(data_X, y_data, test_size=0.25, random_state= seed)
# Instatiation, fitting and prediction



MNB = MultinomialNB()

MNB.fit(X_train, y_train)

predictions = MNB.predict(X_test)
# Model evaluation



print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))



MNB_f1 = round(f1_score(y_test, predictions, average='weighted'), 3)

MNB_accuracy = round((accuracy_score(y_test, predictions)*100),2)



print("Accuracy : " , MNB_accuracy , " %")

print("f1_score : " , MNB_f1)
# Instatiation, fitting and predictions



xgb_ = xgb.XGBClassifier(

 learning_rate =0.1,

 n_estimators=1000,

 max_depth=5,

 min_child_weight=1,

 gamma=0,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'multi:softmax',

 nthread=4,

 scale_pos_weight=1,

 seed= seed)



xgb_.fit(X_train, y_train)

predictions = xgb_.predict(X_test)
# Model evaluation

print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))



xgb_f1 = round(f1_score(y_test, predictions, average= 'weighted'), 3)

xgb_accuracy = round((accuracy_score(y_test, predictions) * 100), 2)



print("Accuracy : " , xgb_accuracy , " %")

print("f1_score : " , xgb_f1)
# Instatiation, fitting and predictions



from sklearn.ensemble import RandomForestClassifier



rfc=RandomForestClassifier(n_estimators= 10, random_state= seed)

rfc.fit(X_train, y_train)

predictions = rfc.predict(X_test)
# Model evaluation



print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))



rfc_f1 = round(f1_score(y_test, predictions, average= 'weighted'), 3)

rfc_accuracy = round((accuracy_score(y_test, predictions) * 100), 2)



print("Accuracy : " , rfc_accuracy , " %")

print("f1_score : " , rfc_f1)
# Instatiation and fitting



from sklearn.svm import SVC

from sklearn.model_selection import KFold

svc = SVC(random_state=seed)
# using KFold cross validation technique

kf=  KFold(n_splits=5, random_state=seed)



# Hyperparametric tuning using grid search

param_grid = [{'kernel':['rbf'],

              'gamma':[1e-3, 1e-4],

              'C':[1, 10, 100, 1000]},

             {'kernel':['linear'],

             'C':[1, 10, 100, 1000]}]



grid = GridSearchCV(estimator=svc, param_grid=param_grid, scoring='accuracy', cv=kf)

grid.fit(X_train, y_train)



print('Estimator: ', grid.best_estimator_)

print('Best params : \n', grid.best_params_)

print('Output Classes: ', grid.classes_)

print('Training Accuracy: ', grid.best_score_)
# predictions



predictions = grid.predict(X_test)



print(classification_report(y_test, predictions))



svc_f1 = round(f1_score(y_test, predictions, average='weighted'), 3)

svc_accuracy = round((accuracy_score(y_test, predictions)*100), 2)



print("Accuracy : " , svc_accuracy , " %")

print("f1_score : " , svc_f1)
from keras import Sequential

from keras.layers import Embedding, LSTM, Dense, Dropout



embedding_size=32

max_words=5000



model = Sequential()

model.add(Embedding(max_words, embedding_size, input_length=X_train.shape[1]))

model.add(LSTM(100))

model.add(Dense(3,activation='softmax'))



print(model.summary())
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
# converting categorical variables in y_train to numerical variables

y_train_dummies = pd.get_dummies(y_train).values

print('Shape of Label tensor: ', y_train_dummies.shape)
#train the model

model.fit(X_train, y_train_dummies, epochs=5, batch_size=32)

#model.sav('MusicalInstrumentReviews.h5')
model.save('MusicalInstrumentReviews.h5')
# converting categorical variables in y_train to numerical variables

y_test_dummies = pd.get_dummies(y_test).values

print('Shape of Label tensor: ', y_test_dummies.shape)
# model evaluation

from keras.models import load_model



model = load_model('MusicalInstrumentReviews.h5')

scores = model.evaluate(X_test, y_test_dummies)



LSTM_accuracy = scores[1]*100



print('Test accuracy: ', scores[1]*100, '%')
#Comapring the accuracy for various models



model = ['MNB', 'Random Forest', 'XGBoost', 'SVM', 'LSTM']

acc = [MNB_accuracy, rfc_accuracy, xgb_accuracy, svc_accuracy, LSTM_accuracy]



sns.set_style("whitegrid")

plt.figure(figsize=(10,5))

plt.yticks(np.arange(0,100,10))

plt.ylabel("Test Accuracy %")

plt.xlabel("Machine Learning Model")

sns.barplot(x= model, y= acc)

plt.show()
# Comparing the f1-score for various models

model = ['MNB', 'Random Forest', 'XGBoost', 'SVM']

f1_score = [MNB_f1, rfc_f1, xgb_f1, svc_f1]



sns.set_style("whitegrid")

plt.figure(figsize=(10,8))

plt.yticks(np.linspace(0,1,21))

plt.ylabel("f1-score")

plt.xlabel("Machine Learning Model")

sns.barplot(x= model,  y= f1_score)

plt.show()