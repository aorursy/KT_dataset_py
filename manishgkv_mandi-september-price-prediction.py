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
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

import folium

from folium import plugins



plt.rcParams['figure.figsize']=12,12



import warnings

warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics
dataset1=pd.read_excel('../input/mandi-data-apple-september/mandi data apple price september.xlsx')

dataset2=pd.read_excel('../input/mandi-data-onion-price-september/mandi data onion price prediction.xlsx')
dataset1.drop('Arrival Date', axis=1, inplace=True)

dataset2.drop('Arrival Date', axis=1, inplace=True)
dataset1
dataset2
dataset1.describe()
dataset2.describe()
x=dataset1[['Arrivals (Tonnes)','Minimum Price(Rs./Quintal)','Maximum Price(Rs./Quintal)']].values

y=dataset1['Modal Price(Rs./Quintal)'].values
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=0)
regressor=LinearRegression()
regressor.fit(x_train,y_train)
print(regressor.coef_)
print(regressor.intercept_)
predicted=regressor.predict(x_test)
print(predicted)
dframe= pd.DataFrame({'Actual': y_test.flatten(),'Predicted':predicted.flatten()})
dframe.head(25)
import math
print('mean absolute error', metrics.mean_absolute_error(y_test,predicted))

print('mean squared error', metrics.mean_squared_error(y_test,predicted))

print('root mean squared error', math.sqrt(metrics.mean_absolute_error(y_test,predicted)))
graph=dframe.head(30)
graph.plot(kind='bar')
x=dataset2[['Arrivals (Tonnes)','Minimum Price(Rs./Quintal)','Maximum Price(Rs./Quintal)']].values

y=dataset2['Modal Price(Rs./Quintal)'].values
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=0)
regressor=LinearRegression()
regressor.fit(x_train,y_train)
print(regressor.coef_)
print(regressor.intercept_)
predicted=regressor.predict(x_test)
print(predicted)
dframe= pd.DataFrame({'Actual': y_test.flatten(),'Predicted':predicted.flatten()})
dframe.head(25)
print('mean absolute error', metrics.mean_absolute_error(y_test,predicted))

print('mean squared error', metrics.mean_squared_error(y_test,predicted))

print('root mean squared error', math.sqrt(metrics.mean_absolute_error(y_test,predicted)))
graph=dframe.head(30)
graph.plot(kind='bar')
import numpy as np

import pandas as pd

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

plt.style.use('bmh')
df=pd.read_excel('../input/mandi-data-apple-september/mandi data apple price september.xlsx')

df.head(6)
df2=pd.read_excel('../input/mandi-data-onion-price-september/mandi data onion price prediction.xlsx')

df2.head(6)
df.shape
df2.shape
plt.figure(figsize=(16,8))

plt.title('CROP PREDICTION')

plt.xlabel('Days')

plt.ylabel('average price')

plt.plot(df['Modal Price(Rs./Quintal)'])

plt.show()
plt.figure(figsize=(16,8))

plt.title('CROP PREDICTION')

plt.xlabel('Days')

plt.ylabel('average price')

plt.plot(df2['Modal Price(Rs./Quintal)'])

plt.show()
df=df[['Modal Price(Rs./Quintal)']]

df.head(4)
df2=df2[['Modal Price(Rs./Quintal)']]

df2.head(4)
future_days=25

df['Prediction']=df[['Modal Price(Rs./Quintal)']].shift(-future_days)

df.head(4)
future_days=25

df2['Prediction']=df2[['Modal Price(Rs./Quintal)']].shift(-future_days)

df2.head(4)
X=np.array(df.drop(['Prediction'],1))[ : -future_days]

print(X)
X2=np.array(df2.drop(['Prediction'],1))[ : -future_days]

print(X2)
y=np.array(df['Prediction'])[ : -future_days]

print(y)
y2=np.array(df2['Prediction'])[ : -future_days]

print(y2)
x_train,x_test, y_train,y_test=train_test_split(X,y,test_size=0.25)
x2_train,x2_test, y2_train,y2_test=train_test_split(X2,y2,test_size=0.25)
tree=DecisionTreeRegressor().fit(x_train,y_train)

lr=LinearRegression().fit(x_train,y_train)
tree=DecisionTreeRegressor().fit(x2_train,y2_train)

lr=LinearRegression().fit(x2_train,y2_train)
x_future=df.drop(['Prediction'],1)[ : -future_days]

x_future=x_future.tail(future_days)

x_future=np.array(x_future)

x_future
x2_future=df2.drop(['Prediction'],1)[ : -future_days]

x2_future=x2_future.tail(future_days)

x2_future=np.array(x2_future)

x2_future
tree_prediction=tree.predict(x_future)

print(tree_prediction)

print()



lr_prediction=lr.predict(x_future)

print(lr_prediction)
tree_prediction=tree.predict(x2_future)

print(tree_prediction)

print()



lr_prediction=lr.predict(x2_future)

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
predictions = tree_prediction

valid=df2[X2.shape[0]: ]

valid['Predictions']=predictions

plt.figure(figsize=(16,8))

plt.title('model')

plt.xlabel('days')

plt.ylabel('average price')

plt.plot(df2['Modal Price(Rs./Quintal)'])

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
predictions = lr_prediction

valid=df2[X2.shape[0]: ]

valid['Predictions']=predictions

plt.figure(figsize=(16,8))

plt.title('model')

plt.xlabel('days')

plt.ylabel('average price')

plt.plot(df2['Modal Price(Rs./Quintal)'])

plt.plot(valid[['Modal Price(Rs./Quintal)','Predictions']])

plt.legend(['Orig','Val','Pred'])

plt.show()