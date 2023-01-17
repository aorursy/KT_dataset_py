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



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import re

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.stem.wordnet import WordNetLemmatizer

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import string



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.pipeline import Pipeline

import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier



from keras import Sequential

from keras.layers import Embedding, LSTM, Dense, Dropout

from keras.models import load_model





seed = 4353
true = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')

fake = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')
true.sample(5)
fake.head()
# Introducing new column in both dataframes



true['impression']=1

fake['impression']=0
# Concatenating them using pandas concatenate to form a single dataframe



data_raw = pd.concat([true, fake], axis=0)

data_raw.sample(10)
# Combining title and text to obtain a single string

# dropping title and



data_raw['fulltext'] = data_raw.title + ' ' + data_raw.text

data_raw.drop(['title','text'], axis=1, inplace=True)
# Extracting a new dataframe using features fulltext and impression

data = data_raw[['fulltext', 'impression']]

data = data.reset_index()

data.drop(['index'], axis=1, inplace=True)
# Check for missing values



data.isnull().sum()
print('The dataset contans {} rows and {} columns'.format(data.shape[0], data.shape[1]))
# Word extraction from true and fake texts



true_text = data[data.impression==1]['fulltext']

fake_text = data[data.impression==0]['fulltext']

fake_text = fake_text.reset_index().drop(['index'], axis=1)
# Function to extract major words from true and fake news



def wordcloud_words(X_data_full):

    

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

        

    return X_data_full_lemmatized_words
true_words = wordcloud_words(true_text)

fake_words = wordcloud_words(fake_text.fulltext)
def plot_wordcloud(text):

    wordcloud = WordCloud(background_color = 'black',

                         max_words = 3000,

                         width=1600,

                         height=800).generate(text)

    plt.clf()

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis('off')

    plt.show()
plt.figure(figsize=(20,18))

plot_wordcloud(' '.join(true_words))
plt.figure(figsize=(20,18))

plot_wordcloud(' '.join(fake_words))
# Data preparation



X_data = data['fulltext']

y_data = data.impression

X_data = X_data.astype(str)
# Function to retrieve processed words



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
# Setting the function with parameters



data_X = final(X_data)
# Preparing training and testing data using train_test_split



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
# Instatiation, fitting and prediction



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
# Comapring the accuracy for various models



model = ['MNB', 'Random Forest']

acc = [MNB_accuracy, rfc_accuracy]



sns.set_style("whitegrid")

plt.figure(figsize=(10,5))

plt.yticks(np.arange(0,100,10))

plt.ylabel("Test Accuracy %")

plt.xlabel("Machine Learning Model")

sns.barplot(x= model, y= acc)

plt.show()
# Comparing the f1-score for various models



model = ['MNB', 'Random Forest']

f1_score = [MNB_f1, rfc_f1]



sns.set_style("whitegrid")

plt.figure(figsize=(10,8))

plt.yticks(np.linspace(0,1,21))

plt.ylabel("f1-score")

plt.xlabel("Machine Learning Model")

sns.barplot(x= model,  y= f1_score)

plt.show()