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
dataframe = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')

dataframe.head()
plt.figure(figsize=(12,10))

sns.heatmap(dataframe.iloc[:,1:32].corr())


X = dataframe.iloc[: ,2:32].values

y = dataframe.iloc[: ,1].values
print(X)
print(y)
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")

imputer.fit(X)

X = imputer.transform(X)

X
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(y)

y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
print(X_train)
print(X_test)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import AdaBoostClassifier

classifier = LogisticRegression()

classifier_boost = AdaBoostClassifier(n_estimators = 10, base_estimator = classifier, learning_rate = 1)

boost = classifier_boost.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

print(accuracy_score(boost.predict(X_train),y_train))
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

cm = confusion_matrix(boost.predict(X_test),y_test)

print("Confusion Matrix: \n",cm,"\n")

print("Score = ",accuracy_score(y_test,boost.predict(X_test)),"\n")

print(classification_report(y_test,boost.predict(X_test)))
import seaborn as sns

sns.heatmap(cm, annot=True)