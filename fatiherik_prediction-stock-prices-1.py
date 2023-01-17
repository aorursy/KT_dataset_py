# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
plt.style.use('bmh')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
AAPL=pd.read_csv('/kaggle/input/appl-stock/AAPL.csv')
AAPL
plt.figure(figsize=(19,8))
plt.plot(AAPL['Close'],label='AAPL',linewidth=2)
plt.xticks(rotation=45)
plt.title('Apple Close Price (252 days)')
plt.xlabel('Days')
plt.ylabel('Close Price ($)')
plt.legend(loc='upper left')
plt.show()
df=AAPL[['Close']]
df.head()
future_days=25
df['Prediction']=df[['Close']].shift(-future_days)
df.head()
df.tail()
X=np.array(df.drop(['Prediction'],1))[:-future_days]
X
y=np.array(df['Prediction'])[:-future_days]
y
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
tree=DecisionTreeRegressor().fit(x_train,y_train)
lr=LinearRegression().fit(x_train,y_train)
x_future=df.drop(['Prediction'],1)[:-future_days]
x_future=x_future.tail(future_days)
x_future=np.array(x_future)
x_future
tree_prediction=tree.predict(x_future)
tree_prediction
lr_prediction=lr.predict(x_future)
lr_prediction
predictions=tree_prediction
valid=df[X.shape[0]:]
valid['Predictions']=predictions
valid
plt.figure(figsize=(19,8))
plt.plot(df['Close'],linewidth=2)
plt.plot(valid[['Close','Predictions']],linewidth=2)
plt.xticks(rotation=45)
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close Price ($)')
plt.legend(['Orig','Val','Pred'],loc='upper left')
plt.show()
predictions=lr_prediction
valid=df[X.shape[0]:]
valid['Predictions']=predictions

plt.figure(figsize=(19,8))
plt.plot(df['Close'],linewidth=2)
plt.plot(valid[['Close','Predictions']],linewidth=2)
plt.xticks(rotation=45)
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close Price ($)')
plt.legend(['Orig','Val','Pred'],loc='upper left')
plt.show()
