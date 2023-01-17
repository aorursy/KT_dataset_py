import warnings

warnings.simplefilter('ignore')
import numpy as np

import pandas as pd

import pylab as plt

import matplotlib.pyplot as plt



from sklearn.linear_model import LogisticRegression



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import cohen_kappa_score



from sklearn.feature_extraction.text import TfidfVectorizer

import pickle



from imblearn.over_sampling import SMOTE

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords



import io

import requests

import nltk

nltk.download('punkt')

nltk.download('stopwords')
%%time

url = 'https://raw.githubusercontent.com/harshildarji/ML-Practise/master/IMDB%20Sentiment%20Classification%20using%20TF-IDF/datasets/data.csv'

_data = requests.get(url).content

data = pd.read_csv(io.StringIO(_data.decode('utf-8')))
data.head()
data.isnull().sum()
data = data.dropna().reset_index(drop=True)
data.isnull().sum()
X_train, X_test, y_train, y_test = train_test_split(data['review'], data['label'], test_size = .4, shuffle = False)
len(X_train), len(y_train), len(X_test), len(y_test)
sentiments = data['label'].value_counts()

print('Sentiments in entire dataset:\n Positive: {}\n Negative: {}'.format(sentiments[1], sentiments[0]))
def get_sentiments(d, _d):

    positive = (d==1).sum()

    negative = (d==0).sum()

    print('Sentiments in {}:\n Positive: {}\n Negative: {}'.format(_d, positive, negative))
get_sentiments(y_train, 'Train data')

get_sentiments(y_test, 'Test data')
def tokenize(text):

    return [word for word in word_tokenize(text.lower()) if word not in stopwords.words('english')]
def choose_vectorizer(option):

    if option == 'generate':

        vectorizer = TfidfVectorizer(tokenizer = tokenize)

    elif option == 'load':

        vectorizer = TfidfVectorizer(vocabulary = pickle.load(open('vocabulary.pkl', 'rb')))

    

    return vectorizer
%%time

options = ['generate', 'load']



# 0 to generate, 1 to load (choose wisely, your life depends on it!)

option = options[0] 



vectorizer = choose_vectorizer(option)

vectorized_train_data = vectorizer.fit_transform(X_train)

vectorized_test_data = vectorizer.transform(X_test)

    

if option == 'generate':

    pickle.dump(vectorizer.vocabulary_, open('vocabulary.pkl', 'wb'))
%%time

sm = SMOTE(random_state=42, ratio=1.0)

X_train, y_train = sm.fit_sample(vectorized_train_data, y_train)
clf = LogisticRegression()
%%time

clf.fit(X_train, y_train)
%%time

kf = KFold(n_splits=10, random_state = 42, shuffle = True)

scores = cross_val_score(clf, X_train, y_train, cv = kf)
print('Cross-validation scores:', scores)

print('Cross-validation accuracy: {:.4f} (+/- {:.4f})'.format(scores.mean(), scores.std() * 2))
predictions = clf.predict(vectorized_test_data)



validation = dict()



validation['accuracy'] = accuracy_score(y_test, predictions)

validation['precision'] = precision_score(y_test, predictions, average='macro')

validation['recall'] = recall_score(y_test, predictions, average='macro')

validation['f1'] = f1_score(y_test, predictions, average='macro')
print('Validation results:\n', '-' * 12)

for v in validation:

    print('{}: {:.5f}'.format(v.title(), validation[v]))
p = predictions.tolist()

ck = cohen_kappa_score(y_test, p)

print('C-K Score: {:.5f}'.format(ck))
example_reviews = [

    'An honest, engaging, and surprisingly funny look back at one of modern television\'s greatest achievements.',

    'Excellent movie! Inspiring and very entertaining for all especially youth and anyone inspired by today\'s modern age of tech entrepreneurship!',

    'Honestly even the trailer made me uncomfortable.',

    'I never write movie reviews, but this one was such a stinker, I feel I owe it to everyone to at least provide a warning.',

    'This movie was a good movie by standard and a lil beyond standard. It was written very well, The acting was great, each characters performance was clever and the comedic timing was spot on. The story line is very real and relatable. Enjoyable for adults and completely appropriate for pre-teens up to 20. Go support, my family loved it.'

]
example_preds = clf.predict(vectorizer.transform(example_reviews))

print(' '.join(str(int(p)) for p in example_preds))