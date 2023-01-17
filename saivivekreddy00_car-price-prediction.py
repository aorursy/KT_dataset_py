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
import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("/kaggle/input/car-price/CarPrice_Assignment.csv")
df.head()
#shape of the data

df.shape
#info the dataframe

df.info()
#describe the data

df.describe()
# checking for duplicates

df.duplicated(subset = ['car_ID']).sum()
# checking for null values

df.isnull().sum()
# pairplot for symboling vs price

sns.pairplot(y_vars = 'symboling', x_vars = 'price' ,data = df)
#Column CarName

df['CarName'].value_counts()
#spliting carname and car company

df['car_company'] = df['CarName'].apply(lambda x:x.split(' ')[0])
#rechecking

df.head()
#deleting the original column

df = df.drop(['CarName'], axis =1)
#cheking car company 

df['car_company']
# replacing names for duplicate values

df['car_company'].replace('toyouta', 'toyota',inplace=True)

df['car_company'].replace('Nissan', 'nissan',inplace=True)

df['car_company'].replace('maxda', 'mazda',inplace=True)

df['car_company'].replace('vokswagen', 'volkswagen',inplace=True)

df['car_company'].replace('vw', 'volkswagen',inplace=True)

df['car_company'].replace('porcshce', 'porsche',inplace=True)
#rechecking the data:

df['car_company'].value_counts()
#doornumber - Number of doors in a car

df['doornumber'].value_counts()
# changing format by using functions concept

def number_(x):

    return x.map({'four':4, 'two': 2})

    

df['doornumber'] = df[['doornumber']].apply(number_)
#rechecking

df['doornumber'].value_counts()
#cylindernumber- cylinder placed in the car

df['cylindernumber'].value_counts()
# changing format by using functions

def convert_number(x):

    return x.map({'two':2, 'three':3, 'four':4,'five':5, 'six':6,'eight':8,'twelve':12})



df['cylindernumber'] = df[['cylindernumber']].apply(convert_number)
# splitting data into numeric values

cars_numeric = df.select_dtypes(include =['int64','float64'])

cars_numeric.head()
plt.figure(figsize = (30,30))

sns.pairplot(cars_numeric)

plt.show()
# checking co-relation of variables

plt.figure(figsize = (20,20))

sns.heatmap(df.corr(), annot = True ,cmap = 'YlGnBu')

plt.show()
#spliting data into categorical values

categorical_cols = df.select_dtypes(include = ['object'])

categorical_cols.head()
# boxplot for checking outliers

plt.figure(figsize = (20,12))

plt.subplot(3,3,1)

sns.boxplot(x = 'fueltype', y = 'price', data = df)

plt.subplot(3,3,2)

sns.boxplot(x = 'aspiration', y = 'price', data = df)

plt.subplot(3,3,3)

sns.boxplot(x = 'carbody', y = 'price', data = df)

plt.subplot(3,3,4)

sns.boxplot(x = 'drivewheel', y = 'price', data = df)

plt.subplot(3,3,5)

sns.boxplot(x = 'enginelocation', y = 'price', data = df)

plt.subplot(3,3,6)

sns.boxplot(x = 'enginetype', y = 'price', data = df)

plt.subplot(3,3,7)

sns.boxplot(x = 'fuelsystem', y = 'price', data = df)
plt.figure(figsize = (20,12))

sns.boxplot(x = 'car_company', y = 'price', data = df)
#creating dummies

cars_dummies = pd.get_dummies(categorical_cols, drop_first = True)

cars_dummies.head()
# merging dummys with the original data set

car_df  = pd.concat([df, cars_dummies], axis =1)
car_df = car_df.drop(['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation',

       'enginetype', 'fuelsystem', 'car_company'], axis =1)
car_df.info()
# importing libraries

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

from sklearn.feature_selection import RFE

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error
# spliting data

df_train, df_test = train_test_split(car_df, train_size = 0.7, test_size = 0.3, random_state = 100)
df_train.shape
df_test.shape
cars_numeric.columns
col_list = ['symboling', 'doornumber', 'wheelbase', 'carlength', 'carwidth','carheight', 'curbweight', 'cylindernumber', 'enginesize', 'boreratio',

            'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'price']
scaler = StandardScaler()
df_train[col_list] = scaler.fit_transform(df_train[col_list])
df_train.describe()
y_train = df_train.pop('price')

X_train = df_train
lr = LinearRegression()

lr.fit(X_train,y_train)



# Subsetting training data for 15 selected columns

rfe = RFE(lr,15)

rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
cols = X_train.columns[rfe.support_]

cols
#model 1

X1 = X_train[cols]

X1_sm = sm.add_constant(X1)



lr_1 = sm.OLS(y_train,X1_sm).fit()
lr_1.params
print(lr_1.summary())
#VIF

vif = pd.DataFrame()

vif['Features'] = X1.columns

vif['VIF'] = [variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = 'VIF', ascending = False)

vif


lr2 = LinearRegression()



rfe2 = RFE(lr2,10)

rfe2.fit(X_train,y_train)
list(zip(X_train.columns,rfe2.support_,rfe2.ranking_))
supported_cols = X_train.columns[rfe2.support_]

supported_cols 
#model 2

X2 = X_train[supported_cols]

X2_sm = sm.add_constant(X2)



model_2 = sm.OLS(y_train,X2_sm).fit()
model_2.params
print(model_2.summary())
#VIF

vif = pd.DataFrame()

vif['Features'] = X2.columns

vif['VIF'] = [variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = 'VIF', ascending = False)

vif
#model 3

X3 = X2.drop(['car_company_subaru'], axis =1)

X3_sm = sm.add_constant(X3)



Model_3 = sm.OLS(y_train,X3_sm).fit()
Model_3.params
print(Model_3.summary())
#VIF

vif = pd.DataFrame()

vif['Features'] = X3.columns

vif['VIF'] = [variance_inflation_factor(X3.values, i) for i in range(X3.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = 'VIF', ascending = False)

vif
#model 4

X4 = X3.drop(['enginetype_ohcf'], axis =1)

X4_sm = sm.add_constant(X4)



Model_4 = sm.OLS(y_train,X4_sm).fit()
Model_4.params
print(Model_4.summary())
#VIF

vif = pd.DataFrame()

vif['Features'] = X4.columns

vif['VIF'] = [variance_inflation_factor(X4.values, i) for i in range(X4.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = 'VIF', ascending = False)

vif
#model 5

X5 = X4.drop(['car_company_peugeot'], axis =1)

X5_sm = sm.add_constant(X5)



Model_5 = sm.OLS(y_train,X5_sm).fit()
Model_5.params
print(Model_5.summary())
#VIF

vif = pd.DataFrame()

vif['Features'] = X5.columns

vif['VIF'] = [variance_inflation_factor(X5.values, i) for i in range(X5.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = 'VIF', ascending = False)

vif
#model 6

X6 = X5.drop(['enginetype_l'], axis =1)

X6_sm = sm.add_constant(X6)



Model_6 = sm.OLS(y_train,X6_sm).fit()
Model_6.params
print(Model_6.summary())
#VIF

vif = pd.DataFrame()

vif['Features'] = X6.columns

vif['VIF'] = [variance_inflation_factor(X6.values, i) for i in range(X6.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = 'VIF', ascending = False)

vif
y_train_pred = Model_6.predict(X6_sm)

y_train_pred.head()
Residual = y_train- y_train_pred
sns.distplot(Residual, bins =15)
df_test[col_list] = scaler.transform(df_test[col_list])
y_test = df_test.pop('price')

X_test = df_test
final_cols = X6.columns
X_test_model6= X_test[final_cols]

X_test_model6.head()
X_test_sm = sm.add_constant(X_test_model6)
y_pred = Model_6.predict(X_test_sm)
y_pred.head()
plt.scatter(y_test, y_pred)

plt.xlabel('y_test')

plt.ylabel('y_pred')
r_squ = r2_score(y_test,y_pred)

r_squ