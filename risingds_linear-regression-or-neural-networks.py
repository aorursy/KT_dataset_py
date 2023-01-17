# import usual stuff

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# loading data

from sklearn.datasets import load_boston

boston = load_boston()
boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_df.head()
boston_df['MEDV'] = boston.target
boston_df.head()
boston_df.info()
boston_df.describe()
sns.boxplot('CHAS','MEDV',data=boston_df)
sns.pairplot(boston_df)
sns.scatterplot('LSTAT','MEDV',data = boston_df)
X = boston_df.drop('MEDV', axis=1)

y= boston_df['MEDV']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
# lets first use linear regression

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)
sns.scatterplot(y_test,y_pred)
sns.distplot(y_test-y_pred)
coef = model.coef_

inter = model.intercept_

coefficients = pd.DataFrame(coef,X.columns,columns=['Coefficients'])

coefficients
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print('MAE: ',mean_absolute_error(y_test,y_pred))

print('RMSE: ',np.sqrt(mean_squared_error(y_test,y_pred)))

print('R2: ',r2_score(y_test,y_pred))
# let's try lasso regression with L1 Penalty

from sklearn.linear_model import Lasso

lassomod = Lasso()

lassomod.fit(X_train,y_train)

yls_pred = lassomod.predict(X_test)
sns.scatterplot(y_test,yls_pred)
sns.distplot(y_test-yls_pred)
coefls = lassomod.coef_

interls = lassomod.intercept_

coefficients_ls = pd.DataFrame(coefls,X.columns,columns=['Coefficients'])

coefficients_ls
print('MAE: ',mean_absolute_error(y_test,yls_pred))

print('RMSE: ',np.sqrt(mean_squared_error(y_test,yls_pred)))

print('R2: ',r2_score(y_test,yls_pred))
# let's try Ridge Regression

from sklearn.linear_model import Ridge

modelridge = Ridge()

modelridge.fit(X_train,y_train)

yridge_pred = modelridge.predict(X_test)
sns.scatterplot(y_test, yridge_pred)
sns.distplot(y_test-yridge_pred)
coefridge = modelridge.coef_

interridge = modelridge.intercept_

coefficients_ridge = pd.DataFrame(coefridge,X.columns,columns=['Coefficients'])

coefficients_ridge
print('MAE: ',mean_absolute_error(y_test,yridge_pred))

print('RMSE: ',np.sqrt(mean_squared_error(y_test,yridge_pred)))

print('R2: ',r2_score(y_test,yridge_pred))
# now let's try neural networks for prediction and see how much we imporve

# for getting number of nodes in a single layer, thumb rule is to have as many as your features

X_train.shape
#scaling for input into tf model

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled_train = scaler.fit_transform(X_train)

X_scaled_test = scaler.transform(X_test)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation

from tensorflow.keras.optimizers import Adam
tfmodel = Sequential()



tfmodel.add(Dense(13,activation='relu'))

tfmodel.add(Dense(13,activation='relu'))

tfmodel.add(Dense(13,activation='relu'))

tfmodel.add(Dense(13,activation='relu'))



tfmodel.add(Dense(1))



tfmodel.compile(optimizer='adam', loss='mse')

tfmodel.fit(x=X_scaled_train,y=y_train.values,

          validation_data=(X_scaled_test,y_test.values),

          batch_size=128,epochs=400)
lossdf = pd.DataFrame(tfmodel.history.history)
lossdf.plot()
y_pred = tfmodel.predict(X_scaled_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



print('MAE:', mean_absolute_error(y_test,y_pred))

print('RMSE:', np.sqrt(mean_squared_error(y_test,y_pred)))

print('R2:', r2_score(y_test,y_pred))