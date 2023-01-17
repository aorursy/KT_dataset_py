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
import matplotlib.pyplot as plt

import seaborn as sns

import nltk

import re

from gensim.models import Word2Vec

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from wordcloud import WordCloud,STOPWORDS



from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import LinearSVC

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.metrics import f1_score
train_data = pd.read_csv('/kaggle/input/topic-modeling-for-research-articles/train.csv')

test_data = pd.read_csv('/kaggle/input/topic-modeling-for-research-articles/test.csv')

train_data.head()

test_data.head()
print(train_data.shape)

print(test_data.shape)
x=train_data.iloc[:,3:].sum()

rowsums=train_data.iloc[:,2:].sum(axis=1)

no_label_count = 0

for sum in rowsums.items():

    if sum==0:

        no_label_count +=1



print("Total number of articles = ",len(train_data))

print("Total number of articles without label = ",no_label_count)

print("Total labels = ",x.sum())
print("Check for missing values in Train dataset")

print(train_data.isnull().sum().sum())

print("Check for missing values in Test dataset")

null_check=test_data.isnull().sum()

print(null_check)
x=train_data.iloc[:,3:].sum()

#plot

plt.figure(figsize=(12,12))

ax= sns.barplot(x.index, x.values, alpha=0.8)

plt.title("Class counts")

plt.ylabel('# of Occurrences', fontsize=12)

plt.xlabel('Label ', fontsize=12)



rects = ax.patches

labels = x.values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')



plt.show()

x=rowsums.value_counts()



#plot

plt.figure(figsize=(12,12))

ax = sns.barplot(x.index, x.values, alpha=0.8)

plt.title("Multiple tags per article")

plt.ylabel('# of Occurrences', fontsize=12)

plt.xlabel('# of Labels ', fontsize=12)



#adding the text labels

rects = ax.patches

labels = x.values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')



plt.show()

train_data['Text']=train_data['TITLE']+' '+train_data['ABSTRACT']
train_data.drop(columns=['TITLE','ABSTRACT'], inplace=True)

train_data.head(10)
plt.figure(figsize=(12,12))

#text = description_category.description.values

cloud = WordCloud(stopwords=STOPWORDS, background_color='black', collocations=False, width=2500, height=1800).generate(" ".join(train_data['Text']))

plt.axis('off')

plt.imshow(cloud)
train_data['Text'][5]
#Remove Stopwords

stop_words = set(stopwords.words('english'))



# function to remove stopwords

def remove_stopwords(text):

    no_stopword_text = [w for w in text.split() if not w in stop_words]

    return ' '.join(no_stopword_text)



train_data['Text'] = train_data['Text'].apply(lambda x: remove_stopwords(x))
#Clean Text

def clean_text(text):

    text = text.lower()

    text = re.sub("[^a-zA-Z]"," ",text) 

    text = ' '.join(text.split()) 

    return text



train_data['Text'] = train_data['Text'].apply(lambda x:clean_text(x))
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

def stemming(sentence):

    stemSentence = ""

    for word in sentence.split():

        stem = stemmer.stem(word)

        stemSentence += stem

        stemSentence += " "

    stemSentence = stemSentence.strip()

    return stemSentence



train_data['Text'] = train_data['Text'].apply(stemming)
train_data['Text'][5]
categories=['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance'] 

train_data[categories].head()
#split the data

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_data['Text'], train_data[categories], test_size=0.2, random_state=40, shuffle=True)

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)



# Define a pipeline combining a text feature extractor with multi lable classifier

NB_pipeline = Pipeline([

                ('tfidf', TfidfVectorizer(stop_words=stop_words)),

                ('clf', OneVsRestClassifier(MultinomialNB(

                    fit_prior=True, class_prior=None))),

            ])



NB_pipeline.fit(x_train, y_train)

prediction = NB_pipeline.predict(x_test)

print('Test accuracy is ', accuracy_score(y_test, prediction))

print('F1 score is ', f1_score(y_test, prediction, average="micro"))
SVC_pipeline = Pipeline([

                ('tfidf', TfidfVectorizer(stop_words=stop_words)),

                ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),

            ])

SVC_pipeline.fit(x_train, y_train)

prediction = SVC_pipeline.predict(x_test)

print('Test accuracy is ',accuracy_score(y_test, prediction))

print('F1 score is ',f1_score(y_test, prediction, average="micro"))
LogReg_pipeline = Pipeline([

                ('tfidf', TfidfVectorizer(stop_words=stop_words)),

                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),

            ])



LogReg_pipeline.fit(x_train, y_train)

prediction = LogReg_pipeline.predict(x_test)

print('Test accuracy is ',accuracy_score(y_test, prediction))

print('F1 score is ',f1_score(y_test, prediction, average="micro"))
# using binary relevance

from skmultilearn.problem_transform import BinaryRelevance

from sklearn.naive_bayes import GaussianNB



# initialize binary relevance multi-label classifier

# with a gaussian naive bayes base classifier

pipeline = Pipeline([

                ('tfidf', TfidfVectorizer(stop_words=stop_words)),

                ('clf', BinaryRelevance(GaussianNB())),

            ])

#classifier = BinaryRelevance(GaussianNB())



# train

pipeline.fit(x_train, y_train)



# predict

predictions = pipeline.predict(x_test)





from sklearn.metrics import accuracy_score

print('Accuracy = ', accuracy_score(y_test,predictions))

print('F1 score is ',f1_score(y_test, prediction, average="micro"))
# using classifier chains

from skmultilearn.problem_transform import ClassifierChain

from sklearn.naive_bayes import GaussianNB



# initialize classifier chains multi-label classifier

# with a gaussian naive bayes base classifier

pipeline = Pipeline([

                ('tfidf', TfidfVectorizer(stop_words=stop_words)),

                ('clf', ClassifierChain(LogisticRegression(solver='sag'))),

            ])

# ClassifierChain(GaussianNB())



# train

pipeline.fit(x_train, y_train)



# predict

predictions = pipeline.predict(x_test)



print('Accuracy = ', accuracy_score(y_test,predictions))

print('F1 score is ',f1_score(y_test, prediction, average="micro"))
# using Label Powerset

from skmultilearn.problem_transform import LabelPowerset

# initialize label powerset multi-label classifier

pipeline = Pipeline([

                ('tfidf', TfidfVectorizer(stop_words=stop_words)),

                ('clf',  LabelPowerset(LogisticRegression())),

            ])

#classifier = LabelPowerset(LogisticRegression())

# train

pipeline.fit(x_train, y_train)

# predict

predictions = pipeline.predict(x_test)

# accuracy

print("Accuracy = ",accuracy_score(y_test,predictions))

print('F1 score is ',f1_score(y_test, prediction, average="micro"))