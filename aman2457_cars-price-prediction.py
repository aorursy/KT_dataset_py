# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



sns.set(rc={'figure.figsize':(11.7,8.27)})



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import chardet

with open('../input/used-cars-database-50000-data-points/autos.csv', 'rb') as rawdata:

    result = chardet.detect(rawdata.read(100000))

result
#reading the csv file and dropping a col named 'noOfPictures' , also encoding parameter is provided 

cars_data = (pd.read_csv('../input/used-cars-database-50000-data-points/autos.csv',encoding='Windows-1252')).drop('nrOfPictures',axis=1)
#making copy of dataframe 

cars_data2  = cars_data.copy()

cars_data3 = cars_data.copy()
#structure of the data

cars_data2.info()
cars_data2.columns
cars_data2.columns= ['dateCrawled', 'name', 'seller', 'offerType', 'price', 'abtest',

       'vehicleType', 'yearOfRegistration', 'gearbox', 'powerPS', 'model',

       'kilometer', 'monthOfRegistration', 'fuelType', 'brand',

       'notRepairedDamage', 'dateCreated', 'postalCode', 'lastSeen']
#creating a function to convert $1,000 into 1000

def convert_price(x):

    x = x.replace('$','')

    x = x.replace(',','')

    

    return x



cars_data2['price'] = cars_data2['price'].apply(convert_price)

cars_data2['price'],cars_data2['kilometer']
#creating a function to convert $1,000 into 1000

def convert_km(x):

    x = x.replace('km','')

    x = x.replace(',','')

    

    return x



cars_data2['kilometer'] = cars_data2['kilometer'].apply(convert_km)

cars_data2['price'],cars_data2['kilometer']
cars_data2.dtypes
#converting the price and kilometer into int

cars_data2['price'] = cars_data2['price'].astype('int64')

cars_data2['kilometer'] = cars_data2['kilometer'].astype('int64')
cars_data2.dtypes


#setting the float format of summary dataframe

pd.set_option('display.float_format',lambda x: '%.2f'%x)

cars_data2.describe()
#dropping some unwanted columns

col = ['name','dateCrawled','dateCreated','postalCode','lastSeen']

cars_data2 = cars_data2.drop(columns=col,axis=1)
#dropping duplicate entries by keeping the first

cars_data2.drop_duplicates(keep='first',inplace=True)
cars_data2.shape
#Data Cleaning



#check the number of null values in all columns

cars_data2.isna().sum()
#variable yearOfRegistration

countYearwise = cars_data2['yearOfRegistration'].value_counts().sort_index()

countYearwise
sum(cars_data2['yearOfRegistration'] > 2020)
sum(cars_data2['yearOfRegistration']< 1950 )
sns.regplot(x='yearOfRegistration' ,y='price' , scatter=True , fit_reg= False , data=cars_data2)
#now checking for varible price

countPrice = cars_data2['price'].value_counts().sort_index()

countPrice
sns.distplot(cars_data2['price'])
cars_data2['price'].describe()
sns.boxplot(y=cars_data2['price'])
sum(cars_data2['price']>150000)
sum(cars_data2['price'] < 100)
#checking for varible powerPS

countPowerPS = cars_data2['powerPS'].value_counts().sort_index()

countPowerPS
sns.distplot(cars_data2['powerPS'])
cars_data2['powerPS'].describe()
sns.boxplot(y=cars_data2['powerPS'])
sns.regplot(x='powerPS' , y='price' ,scatter=True , fit_reg= False , data=cars_data2)
sum(cars_data2['powerPS'] > 500)
sum(cars_data2['powerPS'] <10)
#setting the working range for columns(removing mass outliers)

cars_data2 = cars_data2[ (cars_data2.yearOfRegistration <= 2019) & (cars_data2.yearOfRegistration >=1950)

               & (cars_data2.price <= 150000) & (cars_data2.price >= 100)

                       & (cars_data2.powerPS >= 10) & (cars_data2.powerPS <= 500)]
#setting a new columns which consist of age cars in terms of month or years

cars_data2['monthOfRegistration']  /= 12
cars_data2['Age'] = (2020 - cars_data2['yearOfRegistration']) + cars_data2['monthOfRegistration']

cars_data2['Age'] = round(cars_data2['Age'],2)

cars_data2['Age'].describe()
#dropping the yearOfRegistration and monthOfRegistration

cars_data2 = cars_data2.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1)
cars_data2.shape
#visualizing Age



sns.distplot(cars_data2['Age'])
sns.boxplot(y=cars_data2['Age'])
#for price

sns.distplot(cars_data2['price'])
#for price

sns.boxplot(y=cars_data2['price'])
#powerPs

sns.distplot(cars_data2['powerPS'])
#powerPs

sns.boxplot(y=cars_data2['powerPS'])
#checking for impact of age on price

sns.regplot(x=cars_data2['Age'] , y='price' , scatter =True ,

               fit_reg=False , data=cars_data2)



#inference

#cars whose prices are higher are mostly newer cars as the cars get older the price gets decremnet also. 

#some old cars are also having higher price we can say that those are premium cars
#checking for impact of powerPS on price

sns.regplot(x=cars_data2['powerPS'] , y='price' , scatter =True ,

               fit_reg=False , data=cars_data2)

#inference

#cars whose powerPS are higher has higher Price.
#checking the categorical vars

cars_data2['seller'].value_counts()
pd.crosstab(cars_data2['seller'],columns='count',normalize=True)
sns.countplot(x='seller',data =cars_data2)
#offerType variable

cars_data2['offerType'].value_counts()
sns.countplot(x='offerType',data =cars_data2)
#abtest vars

cars_data2['abtest'].value_counts()
pd.crosstab(cars_data2['abtest'],columns='count',normalize=True)
sns.countplot(x='abtest',data =cars_data2)

#equally distributed
sns.boxplot(x='abtest',y='price',data=cars_data2)

#doesnt affect price much
cars_data2['vehicleType'].value_counts()
pd.crosstab(cars_data2['vehicleType'],columns='count',normalize=True)
sns.countplot(x='vehicleType',data=cars_data2)
sns.boxplot(x='vehicleType',y='price',data=cars_data2)

#vehicleTypes affect the price

#SUV,coupe,cabrio,bus,limousine
#exploring gearbox

cars_data2['gearbox'].value_counts()
pd.crosstab(cars_data2['gearbox'],columns='counts',normalize=True)
sns.countplot(x='gearbox',data=cars_data2)
sns.boxplot(x='gearbox',y='price',data=cars_data2)

#automatic gearbox affects price
#var model

cars_data2['model'].value_counts()
pd.crosstab(cars_data2['model'],columns='counts',normalize=True)
sns.countplot(x='model',data=cars_data2)
sns.boxplot(x='model',y='price',data=cars_data2)

#distributed over many models,can be considered
#kilometer var

cars_data2['kilometer'].value_counts().sort_index()

pd.crosstab(cars_data2['kilometer'],columns='counts',normalize=True)
sns.boxplot(x='kilometer',y='price',data=cars_data2)

#considered
cars_data2['kilometer'].describe()
sns.distplot(cars_data2['kilometer'],bins=8,kde=False)
sns.regplot(x='kilometer',y='price',scatter=True,fit_reg=False,data=cars_data2)
#fuelType

cars_data2['fuelType'].value_counts()
pd.crosstab(cars_data2['fuelType'],columns='count',normalize=True)
sns.countplot(x='fuelType',data=cars_data2)
sns.boxplot(x='fuelType',y='price',data=cars_data2)

#infer

#hybrid cars,diesel are having higher price

#affects price so included
#brand colums

cars_data2['brand'].value_counts()
pd.crosstab(cars_data2['brand'],columns='counts',normalize=True)
sns.countplot(x='brand',data=cars_data2)
sns.boxplot(x='brand',y='price',data=cars_data2)

#variable notRapaired

cars_data2['notRepairedDamage'].value_counts()
pd.crosstab(cars_data2['notRepairedDamage'],columns='counts',normalize=True)
sns.countplot(x='notRepairedDamage',data=cars_data2)
sns.boxplot(x='notRepairedDamage',y='price',data=cars_data2)
#removing insignificant cols

cols=['seller','abtest','offerType']

cars_data2 = cars_data2.drop(columns=cols,axis=1)
cars = cars_data2.copy()
cars.head()
cars['notRepairedDamage'].unique()
#converting some variable in columns

def convert_gearBox(x):

    if x == 'automatik':

        x = 'automatic'

    elif x == 'manuell':

        x = 'manual'

    else:

        x = x

    return x



cars['gearbox'] = cars['gearbox'].apply(convert_gearBox)





        
def convert_vehType(x):

    if x == 'kleinwagen':

        x = 'small car'

    elif x == 'andere':

        x = 'others'

    else:

        x = x

    return x



cars['vehicleType'] = cars['vehicleType'].apply(convert_vehType)
def convert_fuelType(x):

    if x == 'benzin':

        x = 'petrol'

    elif x == 'elektro':

        x = 'electric'

    elif x =='andere':

        x = 'others'

    else:

        x = x

    return x



cars['fuelType'] = cars['fuelType'].apply(convert_fuelType)
#converting some variable in columns

def convert_repairedStats(x):

    if x == 'nein':

        x = 'no'

    elif x == 'ja':

        x = 'yes'

    else:

        x = x

    return x



cars['notRepairedDamage'] = cars['notRepairedDamage'].apply(convert_repairedStats)





        
cars.shape
#correlation

#selecting numerical data

cars_select = cars.select_dtypes(exclude=[object])

correlation = cars_select.corr()

round(correlation,3)
cars_select.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]
#Model Builidng

#1. neglecting the missing rows

cars_noNa = cars.dropna(axis=0)
#converting categorical variables to dummy variables

cars_noNa = pd.get_dummies(cars_noNa,drop_first=True)

cars_noNa
#importing libraries

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
#separting the input and output features

x1 = cars_noNa.drop(['price'],axis='columns',inplace=False)

y1 = cars_noNa['price']
#plotting the variable price

prices = pd.DataFrame({"1.Before":y1 ,"2. After":np.log(y1)})
prices.hist()
#converting the price into log of price

y1 = np.log(y1)
#splitting the data into test and train

X_train , X_test , y_train , y_test = train_test_split(x1,y1,test_size=0.3 ,random_state = 3)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
#baseline model for omitted data



"""

I am making a base model by using test data mean value.

This is to set a benchmark and to compare with our regression model

"""
base_prediction = np.mean(y_test)

print(base_prediction)
#repeating same value till the length of test data

base_prediction = np.repeat(base_prediction,len(y_test))
#finding the root mean squared va;ue

rmse_val = np.sqrt(mean_squared_error(y_test , base_prediction))

print(rmse_val)
#linear regression model

#setting intercept as true

lgr = LinearRegression(fit_intercept=True)



#model

model_first = lgr.fit(X_train,y_train)
#predcition from model

cars_prediction_linmo = lgr.predict(X_test)
#computing mse and rmse

lin_mse1 = mean_squared_error(y_test , cars_prediction_linmo)

lin_rmse1 = np.sqrt(lin_mse1)

print(lin_rmse1)
#R squared value --> good in prediction of y

r2_lin_test1 = model_first.score(X_test,y_test)

r2_lin_train1 = model_first.score(X_train,y_train)

print(r2_lin_test1,r2_lin_train1)
result_lgr1 =  lgr.score(X_test,y_test)

print('Accuracy for Prediction(Omitted Data) using LinearRgression is {}'.format(result_lgr1*100))
#regression diagnostic - Residual Plot Analysis

residuals1 = y_test - cars_prediction_linmo

sns.regplot(x = cars_prediction_linmo , y= residuals1 ,scatter=True,

           fit_reg=False)

#model Parameters

#randomforest model

rf1 = RandomForestRegressor(n_estimators =100, max_features ='auto',

                               max_depth=100 ,min_samples_split =10,

                                   min_samples_leaf =4,random_state =1)
#model of random Forest

model_rf1 = rf1.fit(X_train,y_train)
#predicting from rf

cars_prediction_rf = rf1.predict(X_test)

#computing mse and rmse for random forest model

rf_mse1 = mean_squared_error(y_test,cars_prediction_rf)

rf_rmse1 = np.sqrt(rf_mse1)

print(rf_rmse1)
#calculating R squred Value

rf_test1 = model_rf1.score(X_test,y_test)

rf_train1 = model_rf1.score(X_train,y_train)

print(rf_test1,rf_train1)
#checking the accuracy



result_rf1 =  model_rf1.score(X_test,y_test)

print('Accuracy for Prediction(Omitted Data) using Random Forest is {}'.format(result_rf1*100))
print(cars_prediction_linmo)
output = pd.DataFrame({'Price': y_test, 'Predicted Price': cars_prediction_linmo})

output
#now building the model by imputing the missing datas

cars_imputed = cars.apply(lambda x:x.fillna(x.median())

                             if x.dtype == 'float'

                                 else x.fillna(x.value_counts().index[0]))
cars_imputed.isnull().sum()
cars_imputed.shape
#converting categorical variables into dummy variables

cars_imputed = pd.get_dummies(cars_imputed, drop_first=True)
#model builing

#separating the i/p and o/p var

x2 = cars_imputed.drop(['price'],axis='columns',inplace=False)

y2 = cars_imputed['price']
#plotting the variable price

prices = pd.DataFrame({"1.Before":y2 ,"2. After":np.log(y2)})

prices.hist()
#transforming the price variable into logarithmic value

y2 = np.log(y2)
#splitting the datapoints into test and train

X_train1 , X_test1 , y_train1 , y_test1 = train_test_split(x2,y2 , test_size=0.3 , random_state=3)

print(X_train1.shape,X_test1.shape,y_train1.shape,y_test1.shape)
#baseline model

base_prediction1 = np.mean(y_test1)

print(base_prediction1)
base_prediction1 = np.repeat(base_prediction1 , len(y_test1))
base_rmse_imputed = np.sqrt(mean_squared_error(y_test1,base_prediction1))

print(base_rmse_imputed)
#setting intercept as true

lgr2 = LinearRegression(fit_intercept =True)
#Model

model_lin2 = lgr2.fit(X_train1,y_train1)
cars_prediction_linmo2 = lgr2.predict(X_test1)
#computing rmse and mse error

lin_mse2 = mean_squared_error(y_test1,cars_prediction_linmo2)

lin_rmse2 = np.sqrt(lin_mse2)

print(lin_rmse2)
#R squared value --> good in prediction of y

r2_lin_test2 = model_lin2.score(X_test1,y_test1)

r2_lin_train2 = model_lin2.score(X_train1,y_train1)

print(r2_lin_test2,r2_lin_train2)
result_lgr2 =  lgr2.score(X_test1,y_test1)

print('Accuracy for Prediction(Imputed Data) Using Linear Regression is {}'.format(result_lgr2*100))
#Random Forest with imputed Data

#model Parameters

rf2 = RandomForestRegressor(n_estimators =100, max_features ='auto',

                               max_depth=100 ,min_samples_split =10,

                                   min_samples_leaf =4,random_state =1)
#model of random Forest

model_rf = rf2.fit(X_train1,y_train1)
#predicting by model on test set

cars_prediction_rf1 = rf2.predict(X_test1)
#computing mse and rmse for random forest model

rf_mse2 = mean_squared_error(y_test1,cars_prediction_rf1)

rf_rmse2 = np.sqrt(rf_mse2)

print(rf_rmse2)
#checking the accuracy

rf_test2 = model_rf.score(X_test1,y_test1)

rf_train2 = model_rf.score(X_train1,y_train1)

print(rf_test2,rf_train2)
result_rf2 =  model_rf.score(X_test1,y_test1)

print('Accuracy for Prediction(Omitted Data) using Random Forest is {}'.format(result_rf2*100))
output = pd.DataFrame({'Price': y_test1, 'Predicted Price': cars_prediction_rf1})

output
#Final Output

print('Metrics for model built under the condition where the missing datapoint were omitted')

print("R squared value for train from Linear Regression = {}".format(r2_lin_train1))

print("R squared value for test from Linear Regression = {}".format(r2_lin_test1))

print("R squared value for train from Random Forest = {}".format(rf_train1))

print("R squared value for test from Random Forest = {}".format(rf_test1))

print('Base RMSE value of model built from dataset whose missing datapoints were omitted {}'.format(rmse_val))

print('RMSE value for test from Linear Regression{}'.format(lin_rmse1))

print('RMSE value for test from Random Forest{}'.format(rf_rmse1))

print('Accuracy for Prediction(Omitted Data) using Linear Regression is {}'.format(result_lgr1*100))

print('Accuracy for Prediction(Omitted Data) using Random Forest is {}'.format(result_rf1*100))



print("\n\n")



print('Metrics for model built under the condition where the missing datapoint were imputed using median')

print("R squared value for train from Linear Regression = {}".format(r2_lin_train2))

print("R squared value for test from Linear Regression = {}".format(r2_lin_test2))

print("R squared value for train from Random Forest = {}".format(rf_train2))

print("R squared value for test from Random Forest = {}".format(rf_test2))

print('Base RMSE value of model built from dataset whose missing datapoints were imputed {}'.format(base_rmse_imputed))

print('RMSE value for test from Linear Regression{}'.format(lin_rmse2))

print('RMSE value for test from Random Forest{}'.format(rf_rmse2))

print('Accuracy for Prediction(Imputeded Data) using Linear Regression is {}'.format(result_lgr2*100))

print('Accuracy for Prediction(Imputed Data) using Random Forest is {}'.format(result_rf2*100))



print("\n\n")
