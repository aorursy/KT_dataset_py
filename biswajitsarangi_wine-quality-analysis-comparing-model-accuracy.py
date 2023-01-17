# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv("/kaggle/input/wine-quality/winequalityN.csv")
dataset.head()  # looking into initial 5 rows of dataset
dataset.tail()  # looking into last 5 rows of dataset
dataset.describe()
dataset.isnull().any()  
dataset['fixed acidity'].fillna(dataset['fixed acidity'].mean(),inplace = True)

dataset['volatile acidity'].fillna(dataset['volatile acidity'].mean(),inplace = True)

dataset['citric acid'].fillna(dataset['citric acid'].mean(),inplace = True)

dataset['residual sugar'].fillna(dataset['residual sugar'].mean(),inplace = True)

dataset['chlorides'].fillna(dataset['chlorides'].mean(),inplace = True)

dataset['pH'].fillna(dataset['pH'].mean(),inplace = True)

dataset['sulphates'].fillna(dataset['sulphates'].mean(),inplace = True)
dataset.isnull().any()  # Now we have removed all missing or null values and replaced them by the mean.
dataset.corr()
import seaborn as sn



corrmat = dataset.corr()

sn.heatmap(corrmat,annot = True)
dataset.quality.hist(bins=10)
# Now we will se the variations of each factor against the overall quality,



columns = list(dataset.columns)

columns.remove('type')

columns.remove('quality')





for i in columns:

  fig = plt.figure(figsize = (10,6))

  sn.barplot(x = 'quality', y = i, data = dataset)
dataset = dataset.sample(frac=1).reset_index(drop=True)
dataset.head()
dataset.tail()
for i in range(len(dataset['quality'])):

    if dataset['quality'][i] <= 6.5:

        dataset['quality'][i] = 0

    else:

        dataset['quality'][i] = 1

        

# We classify the good and bad wine with, using a certain threshold.
lb = LabelEncoder()



dataset['type'] = lb.fit_transform(dataset['type'])
x = dataset.iloc[:,0:12].values

y = dataset.iloc[:,12:].values
x # we will scale these values
y
sc = StandardScaler()
x = sc.fit_transform(x)

x
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)
x_train.shape
x_test.shape
y_train.shape
y_test.shape
x_train = sc.fit_transform(x_train)

x_test = sc.fit_transform(x_test)
rfc = RandomForestClassifier(n_estimators=400)

rfc.fit(x_train,y_train.ravel())
y_prediction = rfc.predict(x_test)
print(accuracy_score(y_test,y_prediction))
kn = KNeighborsClassifier(n_neighbors=2)

kn.fit(x_train,y_train.ravel())
y_prediction = kn.predict(x_test)
print(accuracy_score(y_test,y_prediction))
s = SVC()

s.fit(x_train,y_train.ravel())
y_prediction = s.predict(x_test)

print(accuracy_score(y_test,y_prediction))
lr = LogisticRegression()

lr.fit(x_train,y_train.ravel())
y_prediction = lr.predict(x_test)

print(accuracy_score(y_test,y_prediction))