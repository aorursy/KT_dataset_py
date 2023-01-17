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
data = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')
#first five rows

data.head()
data.shape
# checking for null values

data.isnull().sum()
data.describe()
data.info()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.pairplot(data)
data.species.value_counts()
#importing libraries to train model

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler,LabelEncoder
data = data.apply(LabelEncoder().fit_transform)
X = data.iloc[:,0:4]

y = data.iloc[:,-1]
y = data['species']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)
print(len(X_train))

print(len(X_test))
std =StandardScaler()

X_train = std.fit_transform(X_train)

X_test = std.fit_transform(X_test)
clf = LogisticRegression()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
y_pred
data = pd.DataFrame({'Actual':y_test,'Expected':y_pred})
data