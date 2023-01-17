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

filename = "/kaggle/input/stockmarket-sentiment-dataset/stock_data.csv"

df = pd.read_csv(filename)

df.head()
df
sorted_data=df.sort_values('Sentiment', axis=0, ascending=False, inplace=False, kind='quicksort', na_position='last')

sorted_data
sorted_data.describe()
# printing some random reviews

sent_0 = sorted_data['Text'].values[0]

print(sent_0)

print("="*50)



sent_1000 = sorted_data['Text'].values[1000]

print(sent_1000)

print("="*50)



sent_1500 = sorted_data['Text'].values[1500]

print(sent_1500)

print("="*50)



sent_4900 = sorted_data['Text'].values[4900]

print(sent_4900)

print("="*50)
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

vectorizer = TfidfVectorizer( )

X = vectorizer.fit_transform(sorted_data['Text'].values)

y= sorted_data['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, stratify = y, random_state = 5)
import plotly.express as px

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

counter=[]

for i in range(1,100):

    estimator = KNeighborsClassifier(n_neighbors=i)

    estimator.fit(X_train, y_train)

    counter.append(accuracy_score(y_test,estimator.predict(X_test)))

fig = px.line(x=np.arange(1,100), y=counter)

fig.show()
neigh = KNeighborsClassifier(n_neighbors=10)

neigh.fit(X_train, y_train)

y_pred1 = neigh.predict(X_train)

y_pred2 = neigh.predict(X_test)
y_new_pred = neigh.predict(X_test)

print("The predicted sentiment  is: \n", y_new_pred)

y_new_pred[69]
r_sq = neigh.score(X.toarray(), y)

print('coefficient of determination:', r_sq)
from sklearn.metrics import accuracy_score

y_new_pred = neigh.predict(X_test)

accuracy = accuracy_score(y_test, y_new_pred)

print("Accuracy is: %.4f\n" % accuracy)
from sklearn.model_selection import cross_val_score

k_list = list(range(1,50,2))

# creating list of cv scores

cv_scores = []



# perform 10-fold cross validation

for k in k_list:

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')

    cv_scores.append(scores.mean())
import matplotlib.pyplot as plt

import seaborn as sns

MSE = [1 - x for x in cv_scores]



plt.figure()

plt.figure(figsize=(15,10))

plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')

plt.xlabel('Number of Neighbors K', fontsize=15)

plt.ylabel('Misclassification Error', fontsize=15)

sns.set_style("whitegrid")

plt.plot(k_list, MSE)



plt.show()
k_list[MSE.index(min(MSE))]
neigh = KNeighborsClassifier(n_neighbors=9)

neigh.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

y_new_pred = neigh.predict(X_test)

accuracy = accuracy_score(y_test, y_new_pred)

print("Accuracy is: %.4f\n" % accuracy)
y_new_pred = neigh.predict(X_test)

print("The predicted class is: \n", y_new_pred)

y_new_pred[69]
r_sq = neigh.score(X.toarray(), y)

print('coefficient of determination:', r_sq)
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.datasets import make_classification

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings("ignore")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)



print(X_train.shape, y_train.shape)



print(X_test.shape, y_test.shape)

# Create and training a Gaussian Naive Bayes classifier model

from sklearn import naive_bayes

naive = naive_bayes.GaussianNB()

naive.fit(X_train.toarray(), y_train) 

y_new_pred = naive.predict(X_test.toarray())

print("The predicted class is: \n", y_new_pred)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_new_pred)

print("Accuracy is: %.4f\n" % accuracy)
y_new_pred = naive.predict(X_test.toarray())

print("The predicted class is: \n", y_new_pred)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_new_pred)

print("Accuracy is: %.4f\n" % accuracy)
metrics.confusion_matrix(y_test,y_new_pred)
r_sq = naive.score(X.toarray(), y)

print('coefficient of determination:', r_sq)