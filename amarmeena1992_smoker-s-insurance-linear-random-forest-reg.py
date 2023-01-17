import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import os

from warnings import filterwarnings

filterwarnings('ignore')

%matplotlib inline
df = pd.read_csv('../input/insurance.csv')

df.head()
df.describe()
df.info()
df_clean = df.replace(to_replace={'yes':1, 'no':0})
df_clean.sex.unique()
df_clean.region.unique()
#defining my own palette for smokers and non-smokers to appear as red and green respectively.

pal = ['#FF0000', #Red

       '#006400', #Green

      ]
plt.figure(figsize=(10,8))

sns.heatmap(df_clean.corr(), annot=True)
sns.pairplot(df_clean, hue='smoker', palette=pal)
sns.scatterplot(x='charges', y='age', data=df_clean, hue='smoker', palette=pal)
sns.boxplot(x='smoker', y='charges', data=df_clean, palette=pal, order=[1, 0])
sns.scatterplot(x='charges', y='age', data=df_clean[(df_clean['smoker']==1)], color="Red", hue='bmi', palette='Blues')

plt.title('Smoker\'s "Age vs Charges"')
df_clean['BMI below limit'] = df_clean['bmi'].apply(lambda x: 1 if x<=30 else 0)

sns.scatterplot(x='charges', y='age', data=df_clean[(df_clean['smoker']==1)], color="Red", hue='BMI below limit', palette='Blues')

plt.title('Smoker\'s "Age vs Charges"')
from sklearn.model_selection import train_test_split

y= df_clean[df_clean['smoker']==1]['charges']

X= df_clean[df_clean['smoker']==1][['age', 'bmi', 'children']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, y_train)

predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
coefficients = pd.DataFrame(lm.coef_,X.columns)

coefficients.columns = ['Coefficient']

coefficients
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, classification_report



print('MAE:', mean_absolute_error(y_test, predictions))

print('MSE:', mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(mean_squared_error(y_test, predictions)))

print('R2 test_data', r2_score(y_test, predictions))
sns.distplot((y_test-predictions),bins=15)
from sklearn.ensemble import RandomForestRegressor



rfr = RandomForestRegressor(n_estimators =100, criterion = 'mse',random_state = 42,n_jobs = -1)

rfr.fit(X_train,y_train)

rfr_pred_train = rfr.predict(X_train)

rfr_pred_test = rfr.predict(X_test)





print('MSE train_data: ', round((mean_squared_error(y_train,rfr_pred_train)), 1))

print('MSE test_data: ', round(mean_squared_error(y_test,rfr_pred_test), 1))

print('R2 train_data: ',round(r2_score(y_train,rfr_pred_train), 2))

print('R2 test_data: ', round(r2_score(y_test,rfr_pred_test), 2))

sns.distplot((y_test - rfr_pred_test),bins=30)