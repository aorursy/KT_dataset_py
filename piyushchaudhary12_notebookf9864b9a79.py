# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/car-data/CarPrice_Assignment.csv")

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

display(data)

data.describe()
CompanyName = data['CarName'].apply(lambda x : x.split(' ')[0])

data.insert(3,"CompanyName",CompanyName)

data.drop(['CarName'],axis=1,inplace=True)

data.head()
data.CompanyName = data.CompanyName.str.lower()



def replace_name(a,b):

    data.CompanyName.replace(a,b,inplace=True)



replace_name('maxda','mazda')

replace_name('porcshce','porsche')

replace_name('toyouta','toyota')

replace_name('vokswagen','volkswagen')

replace_name('vw','volkswagen')

data.CompanyName.unique()
X=data.iloc[:,1:-1].values

y=data.iloc[:,-1].values

np.set_printoptions(threshold=np.inf)



from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()

X[:, 1] = le.fit_transform(X[:, 1])

X[:, 2] = le.fit_transform(X[:, 2])

X[:, 3] = le.fit_transform(X[:, 3])

X[:, 4] = le.fit_transform(X[:, 4])

X[:, 5] = le.fit_transform(X[:, 5])

X[:, 6] = le.fit_transform(X[:, 6])

X[:, 7] = le.fit_transform(X[:, 7])

X[:, 13] = le.fit_transform(X[:, 13])

X[:, 14] = le.fit_transform(X[:, 14])

X[:, 16] = le.fit_transform(X[:, 16])



print(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =  train_test_split(X, y, random_state = 0)

from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor(random_state = 0)

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

y_pred
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)
X_test
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))