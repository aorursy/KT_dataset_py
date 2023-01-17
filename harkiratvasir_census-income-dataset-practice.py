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

import numpy as np
dataset = pd.read_csv('/kaggle/input/adult-census-income/adult.csv',na_values='?')
dataset.head()
dataset.isnull().sum()
dataset.shape
import matplotlib.pyplot as plt

import seaborn as sns
sns.countplot(dataset['income'])
dataset['income'].value_counts()/dataset['income'].count()*100
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp.fit(dataset[['occupation','workclass','native.country']])
dataset[['occupation','workclass','native.country']] = imp.transform(dataset[['occupation','workclass','native.country']])
dataset.isnull().sum()
dataset.head()
numerical_features = dataset.select_dtypes(include=['int64', 'float64'])

numerical_features
numerical_features.hist(figsize = (20,16),bins = 30)
categorical_features = dataset.select_dtypes('object')
categorical_features.columns
fig = plt.figure(figsize = (20,100))

for i,col in enumerate(categorical_features):

    print(i,col)

    ax1 = fig.add_subplot(9, 1, i+1)

    sns.countplot(dataset[col],hue = dataset['income'])

    plt.xticks(rotation = 90)

    plt.legend(loc = 'best')
X = dataset.drop('income',axis =1)

y = dataset.income
y.value_counts()
X.drop('education.num',axis = 1,inplace  = True)
X = pd.get_dummies(X)
X.shape
from imblearn.over_sampling import SMOTE

smote = SMOTE()

X,y = smote.fit_resample(X,y)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_pred,y_test)*100
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_pred,y_test)*100
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 100)

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_pred,y_test)*100
importances = classifier.feature_importances_

imp = pd.DataFrame(importances)
col = list(X.columns)
imp['Col'] = col
values = imp.sort_values(by = [0,'Col'],ascending = False)
values.head(10)