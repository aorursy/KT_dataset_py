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

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

plt.style.use('bmh')
df=pd.read_excel('../input/same-as/Onion_2019.xlsx')

df.head(6)
df.shape
plt.figure(figsize=(16,8))

plt.title('CROP PREDICTION')

plt.xlabel('Days')

plt.ylabel('average price')

plt.plot(df['Modal Price(Rs./Quintal)'])

plt.show()
df=df[['Modal Price(Rs./Quintal)']]

df.head(4)
future_days=25

df['Prediction']=df[['Modal Price(Rs./Quintal)']].shift(-future_days)

df.head(4)
df.tail(4)
X=np.array(df.drop(['Prediction'],1))[ : -future_days]

print(X)
y=np.array(df['Prediction'])[ : -future_days]

print(y)
x_train,x_test, y_train,y_test=train_test_split(X,y,test_size=0.25)
tree=DecisionTreeRegressor().fit(x_train,y_train)

lr=LinearRegression().fit(x_train,y_train)
x_future=df.drop(['Prediction'],1)[ : -future_days]

x_future=x_future.tail(future_days)

x_future=np.array(x_future)

x_future
tree_prediction=tree.predict(x_future)

print(tree_prediction)

print()



lr_prediction=lr.predict(x_future)

print(lr_prediction)
predictions = tree_prediction

valid=df[X.shape[0]: ]

valid['Predictions']=predictions

plt.figure(figsize=(16,8))

plt.title('model')

plt.xlabel('days')

plt.ylabel('average price')

plt.plot(df['Modal Price(Rs./Quintal)'])

plt.plot(valid[['Modal Price(Rs./Quintal)','Predictions']])

plt.legend(['Orig','Val','Pred'])

plt.show()
predictions = lr_prediction

valid=df[X.shape[0]: ]

valid['Predictions']=predictions

plt.figure(figsize=(16,8))

plt.title('model')

plt.xlabel('days')

plt.ylabel('average price')

plt.plot(df['Modal Price(Rs./Quintal)'])

plt.plot(valid[['Modal Price(Rs./Quintal)','Predictions']])

plt.legend(['Orig','Val','Pred'])

plt.show()