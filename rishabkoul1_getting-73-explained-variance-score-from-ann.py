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
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
delhi_housing = pd.read_csv('/kaggle/input/delhi-house-price-prediction/MagicBricks.csv')
delhi_housing.head()
delhi_housing.info()
delhi_housing.describe()
delhi_housing.columns
sns.pairplot(delhi_housing)
sns.distplot(delhi_housing['Price'])
sns.heatmap(delhi_housing.corr(),annot=True)
sns.heatmap(delhi_housing.isnull(),yticklabels=False,cbar=False,cmap='viridis')
delhi_housing.drop('Per_Sqft',axis=1,inplace=True)
sns.heatmap(delhi_housing.isnull(),yticklabels=False,cbar=False,cmap='viridis')
delhi_housing['Parking'].mean()
delhi_housing['Parking'].unique()
def average(parking):
    if pd.isnull(parking):
        return 2
    else:
        return parking
delhi_housing['Parking'] = delhi_housing['Parking'].apply(average)
delhi_housing['Parking'].unique()
sns.heatmap(delhi_housing.isnull(),yticklabels=False,cbar=False,cmap='viridis')
delhi_housing.dropna(inplace=True)
sns.heatmap(delhi_housing.isnull(),yticklabels=False,cbar=False,cmap='viridis')
delhi_housing.info()
delhi_housing['Furnishing'].unique()
furnished = pd.get_dummies(delhi_housing['Furnishing'],drop_first=True)
delhi_housing['Status'].unique()
status = pd.get_dummies(delhi_housing['Status'],drop_first=True)
status
delhi_housing['Transaction'].unique()
transaction = pd.get_dummies(delhi_housing['Transaction'],drop_first=True)
delhi_housing['Type'].unique()
types = pd.get_dummies(delhi_housing['Type'],drop_first=True)
locality=pd.get_dummies(delhi_housing['Locality'],drop_first=True)
locality
delhi_housing.drop(['Furnishing','Status','Transaction','Type','Locality'],axis=1,inplace=True)
delhi_housing = pd.concat([delhi_housing,furnished,status,transaction,types,locality ],axis=1)
delhi_housing.head()
delhi_housing.columns
X = delhi_housing.loc[:, delhi_housing.columns != 'Price']
y = delhi_housing['Price']
X
y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=50);
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train
X_train= scaler.fit_transform(X_train)
X_train
X_test = scaler.transform(X_test)
X_train.shape
X_test.shape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
model = Sequential()

model.add(Dense(371,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(185,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(93,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(46,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')
model.fit(x=X_train,y=y_train.values,
          validation_data=(X_test,y_test.values),
          epochs=10000,callbacks=[early_stop])
losses = pd.DataFrame(model.history.history)
losses.plot()
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
predictions = model.predict(X_test)
mean_absolute_error(y_test,predictions)
np.sqrt(mean_squared_error(y_test,predictions))
explained_variance_score(y_test,predictions)
# Our predictions
plt.scatter(y_test,predictions)

# Perfect predictions
plt.plot(y_test,y_test,'r')
single_house = delhi_housing.drop('Price',axis=1).iloc[0]
single_house = scaler.transform(single_house.values.reshape(-1, 371))
delhi_housing['Price'][0]
model.predict(single_house)
