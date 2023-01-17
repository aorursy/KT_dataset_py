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
df = pd.read_csv('../input/regression-with-neural-networking/concrete_data.csv')

df.head()
df.info()
print('Number of missing values in dataset:',df.isnull().sum().sum())
corrMatrix = df[df.columns[0:]].corr()['Strength'][:-1]

corrMatrix = corrMatrix.to_frame()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
plt.figure(1,figsize =(20,6))

sns.set(style="whitegrid")

sns.barplot(x = corrMatrix.index,y = corrMatrix['Strength'],data = corrMatrix)

plt.title('Correlation of Strength to other features')

plt.ylabel('Correlation with Strength')

plt.xlabel('Features')

plt.show()
fig = plt.figure(figsize = (20,20))

ax = fig.gca()

df.hist(ax=ax)

plt.show()
X = df.iloc[:,:-1]

y = df.iloc[:,-1]
from sklearn.model_selection import train_test_split as tts

X_train,X_test,y_train,y_test = tts(X,y,test_size = 0.2,random_state = 7)

print(X_train.shape)
from sklearn.preprocessing import PolynomialFeatures as pf

poly = pf(2)

X_train_poly = poly.fit_transform(X_train)

X_test_poly = poly.fit_transform(X_test)
from sklearn.linear_model import LinearRegression as lr

lm = lr()

lm.fit(X_train,y_train)
lm_poly = lr()

lm_poly.fit(X_train_poly,y_train)
y_pred = lm.predict(X_test)

y_pred_poly = lm_poly.predict(X_test_poly)
import math

from sklearn.metrics import mean_squared_error as mse

#math.sqrt(mse(y_test, y_pred))

math.sqrt(mse(y_test,y_pred_poly))