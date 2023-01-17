import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from textblob import TextBlob

from sklearn.model_selection import train_test_split 

from textblob.classifiers import NaiveBayesClassifier
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
df = pd.read_csv("/kaggle/input/amazon-alexa-reviews/amazon_alexa.tsv",delimiter = '\t', quoting = 3)

df.head()
df.shape
df.info()
df.columns
data = df.drop('date', axis = 1)

data.head()
data['feedback'] = np.where(data['feedback'].isin([1]), 'pos', 'neg')

data.head(3)
color = plt.cm.copper(np.linspace(0, 1, 15))

data['variation'].value_counts().plot.bar(color = color, figsize = (10, 6))

plt.title('Distribution of Variations in Alexa', fontsize = 15)

plt.xlabel('variations')

plt.ylabel('count')

plt.show()
sns.distplot(data['rating'])
sns.countplot(x='rating', data=data)
data['verified_reviews'][500]
text_object = TextBlob(data['verified_reviews'][500])

print(text_object.sentiment)
def polarity(review):

    return TextBlob(review).sentiment.polarity



data['Sentiment_Polarity'] = data['verified_reviews'].apply(polarity)

data.head()
sns.distplot(data['Sentiment_Polarity'])
sns.barplot(x='rating', y='Sentiment_Polarity', data=data)
most_negative = data[data.Sentiment_Polarity == -1].verified_reviews.head()

print(most_negative)
data['verified_reviews'][661]
most_positive = data[data.Sentiment_Polarity == 1].verified_reviews.head()

print(most_positive)
data['verified_reviews'][173]
x = data['verified_reviews']

y = data['feedback']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.3)
train = [x for x in zip(x_train,y_train)]

test = [x for x in zip(x_test, y_test)]
clf = NaiveBayesClassifier(train)
print(clf.accuracy(test))
clf.show_informative_features(10)