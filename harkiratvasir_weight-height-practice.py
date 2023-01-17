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
dataset = pd.read_csv('/kaggle/input/weight-height/weight-height.csv')
dataset.head()
dataset.info()
dataset.describe()
dataset.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns
sns.distplot(dataset['Height'])

sns.distplot(dataset['Weight'])
#We can see the distribution is farily normal
sns.scatterplot(x = dataset['Weight'],y = dataset['Height'])

#Height and weight are strongly correlated
dataset.pivot_table(index = dataset['Gender'],values = 'Weight',aggfunc = np.mean).plot(kind = 'bar')
dataset.pivot_table(index = dataset['Gender'],values = 'Height',aggfunc = np.mean).plot(kind = 'bar')
sns.countplot(dataset['Gender'])
sns.heatmap(dataset.corr(),annot = True)
dataset.columns
X = dataset.drop('Gender',axis = 1)

y = dataset.Gender
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

y = labelencoder.fit_transform(y)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)*100