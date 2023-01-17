import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import xgboost as xgb

from sklearn.linear_model import LinearRegression

from sklearn import metrics
df = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
df.head()
df.info()
df.shape
df.describe()
print(df.dtypes.unique())
new_df=pd.DataFrame({'name':df.dtypes})
new_df[new_df['name']=='object']
df.drop(['date'], axis=1, inplace=True)
100*(df.isnull().sum())/(df.shape[0])
df.corr()['price'].sort_values(ascending=False).drop('price')
cor=df.corr()['price'].sort_values(ascending=False).drop('price')
plt.figure(figsize=(8,6))

plt.bar(x=list(cor.index), height=list(cor.values), color='teal')

plt.xticks(rotation=90)

plt.xlabel('Feature', fontsize=12)

plt.ylabel('Correlation', fontsize=12)

plt.title('Correlation of Features with Price', fontsize=15)

plt.show()
X=df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',

       'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above',

       'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long',

       'sqft_living15', 'sqft_lot15']]

y=df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
xgbr= xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,

                           colsample_bytree=1, max_depth=7)
xgbr.fit(X_train,y_train)
prediction_xgbr=xgbr.predict(X_test)
print('RMSE_XGBoost Regression=', np.sqrt(metrics.mean_squared_error(y_test,prediction_xgbr)))

print('R2 Score_XGBoost Regression=',metrics.r2_score(y_test,prediction_xgbr))
plt.figure(figsize=(8,6))

plt.scatter(x=y_test, y=prediction_xgbr, color='dodgerblue')

plt.plot(y_test,y_test, color='deeppink')

plt.xlabel('Actual Sensitivity',fontsize=12)

plt.ylabel('Predicted Sensitivity',fontsize=12)

plt.title('XGBoost Regression (R2 Score=0.89)',fontsize=14)

plt.show()
lr=LinearRegression()

lr.fit(X_train,y_train)
predictions_lr=lr.predict(X_test)
print('RMSE_Linear Regression=', np.sqrt(metrics.mean_squared_error(y_test,predictions_lr)))

print('R2 Score_Linear Regression=',metrics.r2_score(y_test,predictions_lr))
plt.figure(figsize=(8,6))

plt.scatter(x=y_test, y=predictions_lr, color='dodgerblue')

plt.plot(y_test,y_test, color='deeppink')

plt.xlabel('Actual Sensitivity',fontsize=12)

plt.ylabel('Predicted Sensitivity',fontsize=12)

plt.title('Linear Regression (R2 Score=0.71)',fontsize=14)

plt.show()