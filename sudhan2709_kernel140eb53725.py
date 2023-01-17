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

data=pd.read_csv('../input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv')

data
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style="white", color_codes=True)

sns.set(font_scale=1.5)



# import libraries for model validation

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



# import libraries for metrics and reporting

from sklearn.metrics import confusion_matrix

df = data
df.shape
df.head()
df['banking_crisis'] = df['banking_crisis'].astype('category')

df['banking_crisis'] = df['banking_crisis'].cat.codes

df
x = df.iloc[:,4:13]

x
y = df.iloc[:,13].values

y
len(df)

df.head()

df.isnull().any()

df.isnull().sum()
df.isnull().sum()

df.columns
import seaborn as sns

sns.set(style="white", color_codes=True)

sns.set(font_scale=1.5)
len(df)
# Splitting Training and Test Dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42)
print (x_train.shape)

print (x_test.shape)

print (y_train.shape)

print (y_test.shape)
from sklearn.linear_model import LogisticRegression



# fit the model to the training data

model = LogisticRegression()

model.fit(x_train, y_train)



print (model.intercept_)

print (model.coef_)

print (x_train.columns)
display (x_test[:10])

print ()

display (model.predict_proba(x_test)[:10]) # prob

print ()

display (model.predict(x_test)[:10])
from sklearn.metrics import accuracy_score

print ("Logistic testing accuracy is %2.2f" % accuracy_score(y_test,model.predict(x_test)))
print ("Logistic training accuracy is %2.2f" % accuracy_score(y_train,model.predict(x_train)))
from sklearn.tree import DecisionTreeClassifier

classifier =DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors= 5,

                                 metric = 'minkowski', p=2)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', random_state = 0)

classifier.fit(x_train, y_train)
cm = confusion_matrix(y_test, y_pred)

cm
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(x_train, y_train)
cm = confusion_matrix(y_test, y_pred)

cm
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10,

                                   criterion = 'entropy', random_state = 0)
cm = confusion_matrix(y_test, y_pred)

cm
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
from sklearn.decomposition import PCA

#Principal component Analysis

pca = PCA(n_components=None)

x_train_n = pca.fit_transform(x_train)

x_test_n = pca.fit_transform(x_test)
x_train
pd.DataFrame(x_train_n)
from sklearn.decomposition import PCA

#Principal component Analysis

pca = PCA(n_components=2)

x_train_2 = pca.fit_transform(x_train)

x_test_2 = pca.fit_transform(x_test)
explained_variance = pca.explained_variance_ratio_
explained_variance
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(x_train_2, y_train)
y_pred = logmodel.predict(x_test_2)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)
x_test_2