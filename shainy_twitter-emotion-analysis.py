# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import nltk

#nltk.download()

#nltk.download('punkt')

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve

from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import RegexpTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer





from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset=pd.read_csv("//kaggle/input/twitter-reviews-for-emotion-analysis/data.csv")

display(dataset.head())
dataset.describe(include='all')

dataset['length'] = dataset['Tweets'].apply(len)

dataset.head()

graph = sns.FacetGrid(data=dataset,col='Feeling')

graph.map(plt.hist,'length',bins=50,color='Purple')
val = dataset.groupby('Feeling').mean()

val

val.corr()
dataset.Feeling.value_counts()

Sentiment_val=dataset.groupby('Feeling').count()

plt.bar(Sentiment_val.index.values, Sentiment_val['Tweets'])

plt.xlabel('Review Sentiments')

plt.ylabel('Number of Review')

plt.show()
token = RegexpTokenizer(r'[a-zA-Z0-9]+')

cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)

text_counts= cv.fit_transform(dataset['Tweets'])

tf=TfidfVectorizer()

text_tf= tf.fit_transform(dataset['Tweets'])

x=text_tf

y=dataset['Feeling']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=1)
dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)

preddt = dt.predict(x_test)

print("Confusion Matrix for Decision Tree:")

print(confusion_matrix(y_test,preddt))
score = round(accuracy_score(y_test,preddt)*100,2)

print("Score:",score)
print("Classification Report:")

print(classification_report(y_test,preddt))