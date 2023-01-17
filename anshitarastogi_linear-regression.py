import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import os

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
car_data=pd.read_csv('/kaggle/input/vehicle-dataset-from-cardekho/car data.csv')

car_data.head()
print('Shape of dataframe : {}'.format(car_data.shape))
car_data.info()
car_data.describe()
#Checking null values

print('Missing Values :\n{}'.format(car_data.isnull().sum()))
print('DType of features :\n{}'.format(car_data.dtypes))
print('Unique values and counts in object type features :\n')

print(car_data['Car_Name'].value_counts(),'\n')

print(car_data['Fuel_Type'].value_counts(),'\n')

print(car_data['Seller_Type'].value_counts(),'\n')

print(car_data['Transmission'].value_counts(),'\n')
car_data['age_of_car']=2020-car_data['Year']

car_data.head()
fig,ax=plt.subplots(figsize=(18,10))

ax=sns.countplot(x='Year',data=car_data)
fig=plt.figure(figsize=(20,10))

sns.barplot(x='Fuel_Type',y='Selling_Price',data=car_data,ci=None)

plt.title('Variation in selling price of cars due to different fuel type')
fig=plt.figure(figsize=(20,10))

sns.lmplot(x='Kms_Driven',y='Selling_Price',data=car_data,hue='Fuel_Type')

plt.title('Variation in selling price of cars due to distance travelled by car')
fig=plt.figure(figsize=(20,10))

sns.barplot(x='Transmission',y='Selling_Price',data=car_data,hue='Owner',ci=None)

plt.title('Variation in selling price of cars due to the number of owners and transmission')
fig,ax=plt.subplots(1,2,figsize=(20,10))

sns.barplot(x='Seller_Type',y='Selling_Price',data=car_data,hue='Transmission',ax=ax[0],ci=None)

ax[0].set_title('Variation in selling price of cars due to the seller type and the type of car he sells.')

sns.barplot(x='Seller_Type',y='Selling_Price',data=car_data,hue='Owner',ax=ax[1],ci=None)

ax[1].set_title('Variation in selling price of cars due to the seller type and the number of owners the car earlier had.')
fig=plt.figure(figsize=(20,10))

sns.barplot(x='age_of_car',y='Selling_Price',data=car_data,ci=None)

plt.title('Variation in selling price of cars due to age of the car')
fig,ax=plt.subplots(2,2,figsize=(20,12))

sns.barplot(x='age_of_car',y='Selling_Price',data=car_data,hue='Seller_Type',ax=ax[0][0],ci=None)

ax[0][0].set_title('Variation in selling price of cars due to age of the car and seller type')

sns.barplot(x='age_of_car',y='Selling_Price',data=car_data,hue='Fuel_Type',ax=ax[0][1],ci=None)

ax[0][1].set_title('Variation in selling price of cars due to age of the car and fuel type')

sns.barplot(x='age_of_car',y='Selling_Price',data=car_data,hue='Transmission',ax=ax[1][0],ci=None)

ax[1][0].set_title('Variation in selling price of cars due to age of the car and type of car')

sns.barplot(x='age_of_car',y='Selling_Price',data=car_data,hue='Owner',ax=ax[1][1],ci=None)

ax[1][1].set_title('Variation in selling price of cars due to age of the car and number of owners')
sns.lmplot(x='Present_Price',y='Selling_Price',data=car_data,hue='Fuel_Type')
from sklearn.preprocessing import LabelEncoder



le=LabelEncoder()

car_data['fuel_type']=le.fit_transform(car_data['Fuel_Type'])

car_data['seller_type']=le.fit_transform(car_data['Seller_Type'])

car_data['transmission']=le.fit_transform(car_data['Transmission'])

car_data.head()
car_data.drop(['Car_Name','Fuel_Type','Seller_Type','Transmission'],axis=1,inplace=True)

car_data.head()
#Since we have to predict the selling proce of the cars, the target variable is Selling_Price

features=['Year','Present_Price','Kms_Driven','Owner','age_of_car','fuel_type','seller_type','transmission']

X=car_data.loc[:,features]

y=car_data.loc[:,'Selling_Price']
from sklearn.model_selection import train_test_split



X_train, X_test,y_train,y_test = train_test_split(X, y, test_size = 0.30, random_state = 1)
print('Shape of X_train : {} and y_train : {}'.format(X_train.shape,y_train.shape))

print('Shape of X_test : {} and y_test : {}'.format(X_test.shape,y_test.shape))
y_test=y_test.reset_index()

y_test
from sklearn.linear_model import LinearRegression



linreg = LinearRegression()

linreg.fit(X_train,y_train)
y_pred=linreg.predict(X_test)

df_linear=pd.DataFrame(y_pred)

df_linear.head()
from sklearn.linear_model import Ridge

ridge=Ridge(alpha=0.1)

ridge.fit(X_train,y_train)
ridge_predict=ridge.predict(X_test)

df_ridge=pd.DataFrame(ridge_predict)

df_ridge.head()
from sklearn.linear_model import Lasso

lasso=Lasso(alpha=0.1)

lasso.fit(X_train,y_train)
lasso_predict=lasso.predict(X_test)

df_lasso=pd.DataFrame(lasso_predict)

df_lasso.head()
fig,ax=plt.subplots(figsize=(20,10))

plt.scatter(y_test['Selling_Price'],df_ridge,marker='^',s=50,color='r',label='Ridge')

plt.scatter(y_test['Selling_Price'],df_linear,marker='o',s=50,alpha=0.3,color='b',label='Linear')

plt.scatter(y_test['Selling_Price'],df_lasso,marker='*',s=50,alpha=0.3,color='g',label='Lasso')

plt.xlabel('Actual Y')

plt.ylabel('Predicted Y')

plt.legend()

plt.show()
from sklearn.metrics import r2_score

r2={}

r2['Linear_Regression']=r2_score(y_test['Selling_Price'],df_linear)

r2['Ridge_Regression']=r2_score(y_test['Selling_Price'],df_ridge)

r2['Lasso_Regression']=r2_score(y_test['Selling_Price'],df_lasso)

r2
my_submission = pd.DataFrame({'Regression Technique': r2.keys(), 'R2 Score': r2.values()})

my_submission.to_csv('submission.csv')