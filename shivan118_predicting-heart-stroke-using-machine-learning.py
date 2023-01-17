# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = '/kaggle/input/heart-disease-uci/heart.csv'
#load the dataset

data = pd.read_csv(path)

data.head()
data.tail()
data.columns
data.describe()
data.describe().T
data.info()
data['age'].value_counts()
plt.figure(figsize=(15,12))

sns.countplot(x=data['age'])
data['sex'].value_counts()
plt.figure(figsize=(10,8))

sns.countplot(x=data['sex'])
data['cp'].value_counts()
plt.figure(figsize=(10,8))

sns.countplot(x=data['cp'])
data['trestbps'].value_counts()
plt.figure(figsize=(20,8))

sns.countplot(x=data['trestbps'])
data['fbs'].value_counts()
# plt.figure(figsize=(4,8))

sns.countplot(x=data['fbs'])
data['restecg'].value_counts()
sns.countplot(x=data['restecg'])
plt.figure(figsize=(20,8))

sns.countplot(x=data['thalach'])


data['thal'].value_counts()
plt.figure(figsize=(12,8))

sns.countplot(x=data['thal'])
import matplotlib.pyplot as plt

import seaborn as sns

#get correlations of each features in dataset

corrmat = data.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
data.hist(figsize=(20,10))
sns.set_style('whitegrid')

sns.countplot(x='target',data=data,palette='RdBu_r')
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()

columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

data[columns_to_scale] = standardScaler.fit_transform(data[columns_to_scale])
y = data['target']

X = data.drop(['target'], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(X, y,  test_size=0.20)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

knn_scores = []

for k in range(1,21):

    knn_classifier = KNeighborsClassifier(n_neighbors = k)

    score=cross_val_score(knn_classifier,X,y,cv=10)

    knn_scores.append(score.mean())
plt.figure(figsize=(20,8))

plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')

for i in range(1,21):

    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))

plt.xticks([i for i in range(1, 21)])

plt.xlabel('Number of Neighbors (K)')

plt.ylabel('Scores')

plt.title('K Neighbors Classifier scores for different K values')
knn_classifier = KNeighborsClassifier(n_neighbors = 12)

score=cross_val_score(knn_classifier,X,y,cv=10)
score.mean()
from sklearn.ensemble import RandomForestClassifier

randomforest_classifier= RandomForestClassifier(n_estimators=10)



score=cross_val_score(randomforest_classifier,X,y,cv=10)
score.mean()
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=0)

score = cross_val_score(tree, X, y, cv=40)
score.mean()
from sklearn.metrics import confusion_matrix,classification_report

from sklearn.svm import SVC
clf = SVC()

clf.fit(x_train, y_train)

y_pred=clf.predict(x_test)

confusion_matrix(y_pred,y_test)

print(classification_report(y_pred,y_test))
