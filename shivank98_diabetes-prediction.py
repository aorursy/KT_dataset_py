import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import itertools

%matplotlib inline

plt.style.use('fivethirtyeight')





from subprocess import check_output

data = pd.read_csv('../input/diabetes.csv')
data.isnull().sum()
data.shape
sns.countplot(x = 'Outcome', data = data)

plt.show()
data.head(10)
features = data.columns[:8]

print(features)

plt.subplots(figsize=(16,12))

l = len(features)

for i,j in itertools.zip_longest(features, range(l)):

    plt.subplot((l/2),3,j+1)

    plt.subplots_adjust(wspace=0.2, hspace=0.5)

    data[i].hist(bins=20, edgecolor='green')

    plt.title(i)

    

plt.show()    
sns.pairplot(data = data, hue='Outcome', diag_kind='kde')

plt.show()

from sklearn import svm

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.cross_validation import KFold

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

import warnings

warnings.filterwarnings('ignore')
outcome = data['Outcome']

features = data[features]



train, test = train_test_split(data, test_size=0.25, random_state=0,stratify=data['Outcome'])



train_X = train[train.columns[:8]]

test_X = test[test.columns[:8]]



train_y = train['Outcome']



test_y = test[test.columns[8]]
print(test_X.shape, test_y.shape)
model=svm.SVC(kernel='linear')

model.fit(train_X,train_y)

prediction=model.predict(test_X)

from sklearn.metrics import accuracy_score
accuracy_score(prediction,test_y)

model = LogisticRegression()

model.fit(train_X,train_y)

prediction = model.predict(test_X)
accuracy_score(prediction, test_y)
sns.heatmap(data[data.columns[:8]].corr(),annot=True,cmap='RdYlGn')

fig = plt.gcf()

fig.set_size_inches(12,8)

plt.show()