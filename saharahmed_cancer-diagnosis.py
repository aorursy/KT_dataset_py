# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn import datasets

data= datasets.load_breast_cancer()
df = pd.DataFrame(np.c_[data['data'], data['target']],columns = np.append(data['feature_names'], ['target']))

df.head()
df.columns
df.describe()
df.info()

import seaborn as sns

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(df.corr(), annot=True)
X = data['data']

y = data['target']
from sklearn import preprocessing

# standardize the data attributes

standardized_X = preprocessing.scale(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.linear_model import LogisticRegression

logisticRegr = LogisticRegression()

logisticRegr.fit(X_train, y_train)

logisticRegr.predict(X_test)
predictions = logisticRegr.predict(X_test)

score = logisticRegr.score(X_test, y_test)

print(score)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

ac = accuracy_score(y_test,logisticRegr.predict(X_test))

print('Accuracy is: ',ac)

cm = confusion_matrix(y_test,logisticRegr.predict(X_test))

sns.heatmap(cm,annot=True,fmt="d")