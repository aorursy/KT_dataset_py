# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import string

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize

import re

from nltk.stem import PorterStemmer

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/spam-text-message-classification/SPAM text message 20170820 - Data.csv')

print(data.shape)

data.head()
data.isna().sum()
data['Category']= data['Category'].map(lambda x:x.lower())

data = data.replace({'Category':{'ham':1, 'spam':0}})

data.head()
print('Unique classes : \n', data['Category'].unique())

print('Classes ratio : \n', data['Category'].value_counts(normalize=True))
data['Message_length'] = data['Message'].apply(len)

print(data.shape)

data.head()
data['Message_length'].describe()
sns.set(rc={'figure.figsize':(11,5)})

sns.distplot(data['Message_length'] ,hist=True, bins=100)
data_zero = data[data['Category'] ==0 ]

data_one = data[data['Category'] ==1 ]
sns.distplot(data_zero['Message_length'] ,hist=False, bins=100)

sns.distplot(data_one['Message_length'] ,hist=False, bins=100)
def punctuationremoval(dataset):

    clean_list = [char for char in dataset if char not in string.punctuation]

    clean_data = ''.join(clean_list)

    return clean_data
data['Message'] = data['Message'].apply(punctuationremoval)

data['Message'].head()
stop_words = stopwords.words('english')

print(stop_words)
def stopword_removal(dataset):

    tokenization = word_tokenize(dataset)

    clean_data = [word.lower() for word in tokenization if word not in stop_words]

    return clean_data

    
data['Message'] = data['Message'].apply(stopword_removal)

data['Message'].head()
def number_removal(dataset):

    clean_list = []

    for i in dataset:

        if not re.search('\d',i):

            clean_list.append(i)

    clean_data = ' '.join(clean_list) 

    return clean_data
data['Message'] = data['Message'].apply(number_removal)

data['Message'].head()
porter = PorterStemmer()
data['Message'] = data['Message'].apply(lambda x:x.split())

data['Message'].head()
def stem_update(dataset):

    stem_list = []

    for word in dataset:

        word = porter.stem(word)

        stem_list.append(word)

    stem_data = ' '.join(stem_list)

    return stem_data
data['Message'] = data['Message'].apply(stem_update)

data['Message'].head()
from sklearn.feature_extraction.text import CountVectorizer

bow_data = CountVectorizer().fit(data['Message'])

print(len(bow_data.vocabulary_))
message_20 = data['Message'][20]

print(message_20)
message_20 = bow_data.transform([message_20])

print(message_20)

print(message_20.shape)
print(bow_data.get_feature_names()[5139])
message_bow = bow_data.transform(data['Message'])

print(message_bow.shape[1])

print(message_bow.nnz)
sparisty = ( 100*message_bow.nnz / (message_bow.shape[0]*message_bow.shape[1]))

print(sparisty)
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transform = TfidfTransformer().fit(message_bow)

tfidf_trans = tfidf_transform.transform(message_bow)

print(tfidf_trans)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators =10, criterion = 'entropy', random_state = 0)

classifier.fit(tfidf_trans,data['Category'])
print(classifier.predict(message_bow)[2])

print(data.Category[2])
all_predictions = classifier.predict(message_bow)

print(all_predictions)
from sklearn.metrics import classification_report

print(classification_report(data['Category'],all_predictions))
from sklearn import metrics 

print(metrics.accuracy_score(data['Category'],all_predictions))
from sklearn.model_selection import train_test_split

train_message, test_message, train_label, test_label = train_test_split(data['Message'], data['Category'], test_size=0.3)

from sklearn.pipeline import Pipeline

pipeline = Pipeline([

    ('bow', CountVectorizer()),

    ('tfidf', TfidfTransformer()),

    ('classifier',RandomForestClassifier()),]

    

)
pipeline.fit(train_message, train_label)
prediction = pipeline.predict(test_message)
from sklearn.metrics import confusion_matrix,classification_report

cm = confusion_matrix(test_label,prediction)

class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cm), annot=True, cmap="BuPu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
print(classification_report(prediction,test_label))
from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(prediction,test_label))