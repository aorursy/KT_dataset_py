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
import matplotlib.pyplot as plt



import seaborn as sns



from collections import Counter



from sklearn.model_selection import train_test_split

import re



import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import BaggingClassifier



from sklearn.metrics import accuracy_score

from sklearn.metrics import accuracy_score, confusion_matrix
data = pd.read_csv("../input/spam.csv", encoding = 'latin1')

data.head()
# unavailable

data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)
data = data.rename(columns ={"v1":"target", "v2":"text"})

data.head()
data.target.value_counts()
data.groupby('target').describe()
sns.countplot(x = "target", data = data)

data.loc[:, 'target'].value_counts()

plt.title('Distribution of Spam and Ham')
# graph by length

ham =data[data['target'] == 'ham']['text'].str.len()

sns.distplot(ham, label='Ham')

spam = data[data['target'] == 'spam']['text'].str.len()

sns.distplot(spam, label='Spam')

plt.title('Distribution by Length')

plt.legend()
# graph by digits

ham1 = data[data['target'] == 'ham']['text'].str.replace(r'\D+', '').str.len()

sns.distplot(ham1, label='Ham')

spam1 = data[data['target'] == 'spam']['text'].str.replace(r'\D+', '').str.len()

sns.distplot(spam1, label='Spam')

plt.title('Distribution by Digits')

plt.legend()
#graph by non-digits.



ham2 = data[data['target'] == 'ham']['text'].str.replace(r'\w+', '').str.len()

sns.distplot(ham2, label='Ham')

spam2 = data[data['target'] == 'spam']['text'].str.replace(r'\w+', '').str.len()

sns.distplot(spam2, label='Spam')

plt.title('Distribution of Non-Digits')

plt.legend()
count1 = Counter(" ".join(data[data['target']=='ham']["text"]).split()).most_common(30)

data1 = pd.DataFrame.from_dict(count1)

data1 = data1.rename(columns={0: "words of ham", 1 : "count"})

count2 = Counter(" ".join(data[data['target']=='spam']["text"]).split()).most_common(30)

data2 = pd.DataFrame.from_dict(count2)

data2 = data2.rename(columns={0: "words of spam", 1 : "count_"})
# top 30 ham

data1.plot.bar(legend = False, color = 'purple',figsize = (20,15))

y_pos = np.arange(len(data1["words of ham"]))

plt.xticks(y_pos, data1["words of ham"])

plt.title('Top 30 words of ham')

plt.xlabel('words')

plt.ylabel('number')

plt.show()
# top 30 spam

data2.plot.bar(legend = False, color = 'green', figsize = (20,17))

y_pos = np.arange(len(data2["words of spam"]))

plt.xticks(y_pos, data2["words of spam"])

plt.title('Top 30 words of spam')

plt.xlabel('words')

plt.ylabel('number')

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data['text'], data['target'], test_size = 0.3, random_state = 37)

print ("X_train: ", len(X_train))

print("X_test: ", len(X_test))

print("y_train: ", len(y_train))

print("y_test: ", len(y_test))
corpus = []

for i in range(0, 5572):

    review = re.sub('[^a-zA-Z]', ' ', data['text'][i])

    review = review.lower()

    review = review.split()

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    review = ' '.join(review)

    corpus.append(review)
cv = CountVectorizer(max_features = 1500)

cv.fit(X_train)
X_train_cv = cv.transform(X_train)

X_train_cv
X_test_cv = cv.transform(X_test)

X_test_cv
# Naive Bayes

mnb = MultinomialNB(alpha = 0.5)

mnb.fit(X_train_cv,y_train)



y_mnb = mnb.predict(X_test_cv)
print('Naive Bayes Accuracy: ', accuracy_score( y_mnb , y_test))

print('Naive Bayes confusion_matrix: \n', confusion_matrix(y_mnb, y_test))
# KNN classification

knc = KNeighborsClassifier(n_neighbors=100)

knc.fit(X_train_cv,y_train)



y_knc = knc.predict(X_test_cv)
print('KNeighbors Accuracy_score: ',accuracy_score(y_test,y_knc))

print('KNeighbors confusion_matrix: \n', confusion_matrix(y_test, y_knc)) 
#code ends