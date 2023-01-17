import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('darkgrid')
df_train = pd.read_csv('../input/train.csv')
df_train.shape
df_train.head()
df_train.info()
df_train.describe()
df_train.columns
sns.distplot(df_train['SalePrice'])
df_train.corr()
f,ax = plt.subplots(figsize=(20,20))
sns.heatmap(df_train.corr(), annot = True,linewidths=.2, fmt='.1f', ax=ax)


df_train['SalePrice'].describe()

sns.pairplot(df_train,x_vars=['TotalBsmtSF','GrLivArea','GarageArea','OverallQual','FullBath'],
             y_vars=['SalePrice'],kind='reg')
df_train.columns
X = df_train[['TotalBsmtSF','GrLivArea','GarageArea','OverallQual','FullBath']]
y = df_train['SalePrice']
#from sklearn.cross_validation import train_test_split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=35)
from sklearn import metrics
#Mean Absolute Error 
print('MAE:', metrics.mean_absolute_error(y_test, predictions))

#Mean Squared Error
print('MSE:', metrics.mean_squared_error(y_test, predictions))

#Root Mean Squared Error
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
