# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os, re

from collections import Counter

import nltk

from nltk.stem import PorterStemmer, WordNetLemmatizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
data = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv', encoding='latin')

data.head()
data.shape
data.isna().sum()
data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

data.columns = ['class', 'emails']
data['class'].value_counts()
data['emails'][data['class']== 'spam']
spam_mails = ''

for s in data['emails'][data['class']== 'spam']:

    spam_mails += s

spam_mails_1 = re.sub(r'[^a-zA-Z]', ' ', spam_mails).lower()

words = re.findall(r'\w+', spam_mails_1)

words = [word for word in words if word not in set(nltk.corpus.stopwords.words('english'))]

Counter(words).most_common(30)
length = []

for i in range(data.shape[0]):

    length.append(len(data['emails'][i]))
data.insert(2, 'len', pd.Series(length, name='length'))

data
spam_data = data[data['class']== 'spam']

ham_data = data[data['class']== 'ham']

plt.figure(figsize=(8, 6))

sns.distplot(spam_data['len'], label='spam')

sns.distplot(ham_data['len'], label='ham')

plt.legend()
x_data = data['emails']

y_data = data['class']
def clean_data(x):

    st = PorterStemmer()

    wnl = WordNetLemmatizer()

    cleaned_sent = []

    for i in range(len(x)):

        sent = re.sub(r'[^a-zA-Z]', ' ', x[i])

        sent = sent.lower().split()

        sent = [wnl.lemmatize(word) for word in sent if word not in set(nltk.corpus.stopwords.words('english'))]

        sent = [st.stem(w) for w in sent]

        sent = ' '.join(sent)

        cleaned_sent.append(sent)

    for i in range(len(cleaned_sent)):

                   x[i] = cleaned_sent[i]

                       

    return x
x_data = clean_data(x_data)
encoder = LabelEncoder()

y_data = encoder.fit_transform(y_data)
# converting emails to vectors

cv = CountVectorizer(analyzer='word', max_features=5000, token_pattern=r'\w+')

x_trans = cv.fit_transform(x_data)
x_train, x_test, y_train, y_test = train_test_split(x_trans, y_data, test_size=0.2, random_state=0)
# function to check models performence

def model(model, x_train, x_test, y_train, y_test):

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print(confusion_matrix(y_true=y_test, y_pred=y_pred))

    print(classification_report(y_true=y_test, y_pred=y_pred))

    print(f'accuracy_score :{accuracy_score(y_true=y_test, y_pred=y_pred)}\nroc_auc_score :{roc_auc_score(y_test, y_pred)}')
model(RandomForestClassifier(random_state=0), x_train, x_test, y_train, y_test)
model(AdaBoostClassifier(random_state=0), x_train, x_test, y_train, y_test)
model(MultinomialNB(),x_train, x_test, y_train, y_test)
model(LogisticRegression(C=10, random_state=0), x_train, x_test, y_train, y_test)
model(SVC(C=100),x_train, x_test, y_train, y_test)
stacking_model = StackingClassifier(estimators=[('lgr', LogisticRegression(C=10, random_state=0)), ('rdf', RandomForestClassifier(random_state=0)), ('mnb',MultinomialNB())], final_estimator=LogisticRegression(random_state=0))
model(stacking_model,x_train, x_test, y_train, y_test)
models = ['RandomForestClassifier', 'AdaBoostClassifier', 'MultinomialNB', 'LogisticRegression', 'SVC', 'StackingClassifier']

accuracy = ['0.97578', '0.96053', '0.98565', '0.9856', '0.97578', '0.99192']

models = pd.DataFrame(models, columns=['model'])

models.insert( 1, 'accuracy', pd.Series(accuracy))

models