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
dataset = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
dataset.head()
dataset.isnull().any()
import seaborn as sns

import matplotlib.pyplot as plt

for feature in dataset.columns:

    if feature == 'quality':

        break

    sns.boxplot('quality',feature, data = dataset)

    plt.figure()

    

        

    
corr = dataset.corr()



fig, ax = plt.subplots(figsize=(10, 8))



sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f")









plt.show()
dataset.drop(['pH','chlorides','free sulfur dioxide', 'residual sugar'], axis = 1, inplace = True)
dataset
sns.countplot(x = 'quality', data = dataset)
bins = (2, 6.5, 8)

labels = ['bad', 'good']

dataset['quality'] = pd.cut(x = dataset['quality'], bins = bins, labels = labels)
sns.countplot(x = 'quality', data = dataset)
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

dataset['quality'] = labelencoder.fit_transform(dataset['quality'])
X = dataset.drop('quality', axis = 1)

y = dataset['quality']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 30)
print("Shape of X_train: ",X_train.shape)

print("Shape of X_test: ", X_test.shape)

print("Shape of y_train: ",y_train.shape)

print("Shape of y_test",y_test.shape)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

from sklearn.svm import SVC

classifier = SVC()

from sklearn.model_selection import GridSearchCV

parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},

              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

grid_search = GridSearchCV(estimator = classifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10,

                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)
accuracy = grid_search.best_score_

accuracy
grid_search.best_params_
classifier = SVC(kernel = 'rbf', gamma=0.7)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test,y_pred)

accuracy
