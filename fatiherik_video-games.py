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
        df = pd.read_csv(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
df.head()
df.columns
df.shape
df.isna().any()
df.isnull().sum()
df.info()
df.describe()
df['Year'].unique()
df['Platform'].unique()
df['Genre'].unique()
df['Publisher'].unique()
df['Publisher'].value_counts()
df['Year'].value_counts()
print(df['Year'].median())
print(df['Year'].mean())
print(df['Year'].mode())
df['Year']=df['Year'].fillna(df['Year'].median())
df['Year'].isnull().sum()
df['Year'].value_counts()
df['Publisher']=df['Publisher'].replace(np.nan, df['Publisher'].mode()[0])
df['Publisher'].isnull().sum()
df.isnull().sum()
categorical_labels = ['Platform', 'Genre', 'Publisher']
numerical_lables = ['Global_Sales']
enc = LabelEncoder()
encoded_df = pd.DataFrame(columns=['Platform', 'Genre', 'Publisher', 'Global_Sales'])

for label in categorical_labels:
    temp_column = df[label]

    encoded_temp_col = enc.fit_transform(temp_column)

    encoded_df[label] = encoded_temp_col

for label in numerical_lables:
    encoded_df[label] = df[label].values

encoded_df.head()
encoded_df['Platform'].value_counts()
encoded_df['Publisher'].value_counts()
encoded_df['Genre'].value_counts()
#from sklearn import preprocessing
#norm_encoded = preprocessing.normalize(encoded_df)
#norm_encoded

# bu yontem kullanilabilir mi? kullanilabilirse bagimli degiskeni de normalize edecek miyiz? cunku asagida standartscaler yaparken 
#global sales sutununu scaler etmemis.
x = encoded_df.iloc[:, 0:3]
y = encoded_df.iloc[:,3:]

scalar = StandardScaler()
x = scalar.fit_transform(x)
x

#import statsmodels.api as sm
xm=sm.add_constant(x)
model = sm.OLS(y,xm).fit()
print_model = model.summary()
print(print_model)
#print(xm)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
lr = LinearRegression()
lr.fit(x_train,y_train)
y_predict = lr.predict(x_test)

print(lr.coef_)
print(lr.intercept_)
print(lr.predict([[26,10,359]]))
print(lr.predict([[1.21567658,  1.3482215 ,  0.3714072 ]]))
# predict yaparken scaler edilmis degerleri mi yazacagiz? scaler edilmemis degerleri mi?
mae=mean_absolute_error(y_predict, y_test)
mse=mean_squared_error(y_predict, y_test)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_predict)

print('mae is  {}'.format(mae))
print('mse is  {}'.format(mse))
print('rmse is  {}'.format(rmse))
print('r2 is  {}'.format(r2))
#import matplotlib.pyplot as plt
plt.scatter(y_test,y_predict)
linear_reg = LinearRegression()
y_pred = cross_val_predict(linear_reg, x_test, y_test, cv=10)

mae=mean_absolute_error(y_pred, y_test)
mse=mean_squared_error(y_pred, y_test)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)

print('mae is  {}'.format(mae))
print('mse is  {}'.format(mse))
print('rmse is  {}'.format(rmse))
print('r2 is  {}'.format(r2))

plt.scatter(y_test,y_pred)
from sklearn.preprocessing import PolynomialFeatures

for i in range(1,6):

    poly_features = PolynomialFeatures(degree=i)
    
    x_train_poly = poly_features.fit_transform(x_train)
    x_test_poly = poly_features.fit_transform(x_test)

    poly_model = LinearRegression()
    poly_model.fit(x_train_poly, y_train)
    
    #coef and intercept
    #print('for degree '+str(i)+':'+'coef: ' +str(poly_model.coef_)+' intercept: '+str(poly_model.intercept_))

    # RMSE and r2 score for train data
    y_train_pred = poly_model.predict(x_train_poly)
    rmse_train = np.sqrt(mean_squared_error(y_train,y_train_pred))
    r2_train = r2_score(y_train, y_train_pred)
    print('train data for degree '+str(i)+' rmse_train:' +str(rmse_train)+' r2_train: '+str(r2_train))

    # RMSE and r2 score for test data
    y_test_pred = poly_model.predict(x_test_poly)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_test = r2_score(y_test, y_test_pred)
    print('test data for degree '+str(i)+' rmse_train:' +str(rmse_test)+' r2_train: '+str(r2_test)+'\n')
    
    
    plt.plot(y_test,y_test_pred, color= "green",label = "poly")
    plt.legend()
    plt.show()
