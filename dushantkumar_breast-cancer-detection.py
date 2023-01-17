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
import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt
data= pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
data.shape
data.head()
data.describe()
data.isna().sum()
data=data.dropna(axis=1)
data.shape
data['diagnosis'].value_counts()

#b-don't have cancer

#m- have cancer
sns.countplot(data['diagnosis'])
data.dtypes
from sklearn.preprocessing import LabelEncoder

labelencoder_y= LabelEncoder()

data.iloc[:,1]=labelencoder_y.fit_transform(data.iloc[:,1].values)

data.iloc[:,1]
sns.pairplot(data.iloc[:,1:7],hue="diagnosis")
plt.figure(figsize=(10,10))

sns.heatmap(data.iloc[:,1:12].corr(),annot=True,fmt='.0%')
X= data.iloc[:,2:31].values

Y= data.iloc[:,1].values
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler



Sc= StandardScaler()

X_train= Sc.fit_transform(X_train)

X_test= Sc.fit_transform(X_test)
#Logistic Regression

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



logreg = LogisticRegression(random_state=0)

logreg.fit(X_train, Y_train)

y_pred = logreg.predict(X_test)

acc_logreg = round(accuracy_score(y_pred, Y_test) * 100, 2)

print(acc_logreg)
#Decision Tree

from sklearn.tree import DecisionTreeClassifier



decisiontree = DecisionTreeClassifier(random_state=0,criterion='entropy')

decisiontree.fit(X_train, Y_train)

y_pred = decisiontree.predict(X_test)

acc_decisiontree = round(accuracy_score(y_pred, Y_test) * 100, 2)

print(acc_decisiontree)
# Random Forest

from sklearn.ensemble import RandomForestClassifier



randomforest = RandomForestClassifier(n_estimators=10,random_state=0)

randomforest.fit(X_train, Y_train)

y_pred = randomforest.predict(X_test)

acc_randomforest = round(accuracy_score(y_pred, Y_test) * 100, 2)

print(acc_randomforest)
models = pd.DataFrame({

    'Model': [ 'Logistic Regression', 

              'Random Forest','Decision Tree'],

    'Score': [ acc_logreg, 

              acc_randomforest, acc_decisiontree]})

models.sort_values(by='Score', ascending=False)
from sklearn.metrics import confusion_matrix

cm= confusion_matrix(Y_test,randomforest.predict(X_test))

cm
#predict the prediction of random forest



predict=randomforest.predict(X_test)

print(predict)

print()

print()

print(Y_test)
#******* End ************