# Importing the packages for further processes

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# ignore warnings

import warnings

warnings.filterwarnings('ignore')





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





# Working with os module - os is a module in Python 3.

# Its main purpose is to interact with the operating system. 

# It provides functionalities to manipulate files and folders.



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
print('# File sizes')

for f in os.listdir('../input'):

    print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')
# Reading the file by using pandas library

path=r'../input/used_cars.csv'

used_cars = pd.read_csv(path, sep=',',encoding='Latin1')

used_cars#price, abtest, vehicleType, yearOfRegistration, gearbox, powerPs, model, kilometer, monthOfRegistration, fueltype, brand, notRepairedDamage, 
# Unnecessary columns -

#*dateCrawled, name, seller, offerType, dateCreated, *nrOfPictures, *postalCode, *lastSeen columns are removed
#pip install pandas-profiling

#import pandas_profiling as pp
#pp.ProfileReport(used_cars)

# There are 4 numeric and 15 categorical datatypes

# powerPS is highly skewed and  have more no of zeros

# vehicleType has more no of missing values
cars_new = used_cars.iloc[:,4:16]
cars_new.dtypes
# After removing unnecessary columns, dealing with missing values, outliers by replacing them with mode, lower and upper outlier values respectively

# yearOfRegistration

miss_y = cars_new.yearOfRegistration.isnull().sum()

print('no of missing value:',miss_y)

count_y = cars_new.yearOfRegistration.value_counts()

print('count:',count_y)

# replacing missing values with mode

mode_y = cars_new.yearOfRegistration.mode()

print('mode :',mode_y)





cars_new.yearOfRegistration.value_counts()

cars_new.yearOfRegistration.isnull().sum()

cars_new.yearOfRegistration.fillna(2000,inplace=True)

cars_new.yearOfRegistration.isnull().sum()
cars_new.yearOfRegistration = pd.to_numeric(cars_new.yearOfRegistration,errors='coerce')

# removing the observations which have yearOfRegistration below 1945 and above 2017

cars_new = cars_new[

  (cars_new["yearOfRegistration"].between(1945,2017, inclusive=True))

]

# model

count_m = cars_new.model.value_counts()

missing_m = cars_new.model.isnull().sum()

print("count: ",count_m,'missing values: ',missing_m)

cars_new.model.fillna('blank',inplace=True)

# there is no need of replacing outliers

# labelling the variable using label encoder

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

cars_new['model']=le.fit_transform(cars_new['model'].astype('str'))
cars_new.model
# brand

count_brand = cars_new.brand.value_counts()

missing_brand = cars_new.brand.isnull().sum()

print('count: ',count_brand,'missing values: ',missing_brand)

# replacing missingvalues with mode

cars_new.brand.fillna('blank',inplace=True)

cars_branddtype = cars_new.model.dtype

print("data type: ",cars_branddtype)

cars_new['brand']=le.fit_transform(cars_new['brand'].astype('str'))
cars_new.brand.value_counts()
cars_new
cars_new.info()
cars_new.describe()
#pp.ProfileReport(cars_new)
# First deal with numeric datatype, replacing missing values, outliers  

# powerPS, monthOfRegistration

cars_new.powerPS
cars_new.powerPS.value_counts().sort_index()
# First dealing with missing values, replacing them with median 

cars_new.powerPS.isnull().sum()

cars_new.powerPS.fillna(115,inplace=True)

# Rough cleansing should be done to remove the values below 100 and above 500

cars_new = cars_new[

  (cars_new["powerPS"].between(50, 500, inclusive=True))

]

# Secondly dealing with outliers, replacing them with lower dendrum values and higher dendrum values.

# lower dendrum value is 0, may be replacing with 0 is not a good idea, may be replacing with lower quartile is good

# replacing with lower quartile and will check first

qpo1 = cars_new.powerPS.quantile(0.25)

qpo2 = cars_new.powerPS.quantile(0.75)

iqrpo = qpo2 -qpo1

print("inter qaurtile range : ",iqrpo)

lowerpo = qpo1 - (1.5*iqrpo)

upperpo = qpo2 + (1.5*iqrpo)

print("lower outlier value: ",lowerpo,',',"upper outlier value: ",',',upperpo)

cars_new.powerPS.loc[cars_new.powerPS < lowerpo] = lowerpo

cars_new.powerPS.loc[cars_new.powerPS > upperpo] = upperpo

cars_new.powerPS.value_counts()
cars_new.monthOfRegistration.value_counts().sort_index()
mor_na = cars_new.monthOfRegistration.isnull().sum()

print("number of missing values is",mor_na)

mor_mode = cars_new.monthOfRegistration.mode()

print(" Mode of mor is",mor_mode)

cars_new.monthOfRegistration.fillna(1,inplace=True)

# As the mode is 0, but i replace values with 3



cars_new.monthOfRegistration.value_counts()

cars_new
# Next dealing with outliers, replacing with lower and upper values, but replacing with qauntiles due to negative values

qm1 = cars_new.monthOfRegistration.quantile(0.25)

qm2 = cars_new.monthOfRegistration.quantile(0.75)

print('qm1=',qm1,'qm2=',qm2)

# replacing outliers with lower and upper values

iqrm = qm2 - qm1

print('iqr :',iqrm)

lowerm = qm1 -(1.5*iqrm)

upperm = qm2 +(1.5*iqrm)

print("lower outlier value: ",lowerm,',',"upper outlier value: ",',',upperm)

cars_new.monthOfRegistration.loc[cars_new.monthOfRegistration < lowerm] = lowerm

cars_new.monthOfRegistration.loc[cars_new.monthOfRegistration > upperm] = upperm

# As there are only 12 months in a year but referred variable contains 13, so replacing 0 value with mode value

cars_new.monthOfRegistration.value_counts() # here after 0 the mode is 3

# After numerical, dealing with categorical

# abtest, vehicleType, yearOfRegistration, gearbox, kilometer, fueltype, notRepairedDamage

# abtest

# abtest is removed from the dataset as it increases the rmse value

'''

count_ab = cars_new.abtest.value_counts()

miss_ab = cars_new.abtest.isnull().sum()

print("sum of missing values:",miss_ab)

cars_new.abtest.fillna(value='blank',inplace=True)

# no outliers for this variable

count_ab

'''



del cars_new['abtest']

# vehicleType

count_v = cars_new.vehicleType.value_counts()

print(count_v)

miss_v = cars_new.vehicleType.isnull().sum()

print(miss_v)

# replacing missing values with its mode

mode_v = cars_new.vehicleType.mode()

print("mode is :",mode_v)

cars_new.vehicleType.fillna(value='blank',inplace = True)

# there is no need of replacing any outliers
cars_new.yearOfRegistration.mode()
cars_new
#gearbox

count_g = cars_new.gearbox.value_counts()

print('count:',count_g)

miss_g = cars_new.gearbox.isnull().sum()

print('missing values:',miss_g)

#replacing missing values with mode if possible

mode_g = cars_new.gearbox.mode()

print("mode:",mode_g)

cars_new.gearbox.fillna(value='blank',inplace=True)

cars_new.gearbox[cars_new.gearbox == '25-03-2016 00:00'] = 'manuell'
cars_new.kilometer.value_counts()
#kilometer

cars_new.kilometer.fillna(150000,inplace=True)

cars_new['kilometer'] = cars_new['kilometer'].astype(np.int64)



miss_k = cars_new.kilometer.isnull().sum()

count_k = cars_new.kilometer.value_counts()

print("missingvalues:",miss_k,"count:",count_k)

qk1 = cars_new.kilometer.quantile(0.25)

qk2 = cars_new.kilometer.quantile(0.75)

print("quantiles:",'qk1:',qk1,'qk2:',qk2)

iqrk = qk2-qk1

print('iqrk =',iqrk)

lowerk = qk1 - (1.5*iqrk)

upperk = qk2 + (1.5*iqrk)

print("lower:",lowerk,'upperk:',upperk)

#replacing outliers 

cars_new.kilometer[cars_new.kilometer < lowerk] = lowerk

cars_new.kilometer[cars_new.kilometer > upperk] = upperk



#fuelType

missing_f = cars_new.fuelType.isnull().sum()

print('missing values:',missing_f)

# replacing missing values with mode

cars_new.fuelType.fillna(value='blank',inplace=True)

# replacing outliers

#notRepairedDamage

count_n = cars_new.notRepairedDamage.value_counts()

missing_n = cars_new.notRepairedDamage.isnull().sum()

print('count',count_n,'missing :',missing_n)

cars_new.notRepairedDamage.fillna(value='blank',inplace=True)
# removing the observations which are less than 100 and more than 2,00,000

cars_new = cars_new[

    (cars_new["price"].between(100, 200000, inclusive=True))

]



# output variable price also have missing valuess



cars_new.price.isnull().sum()

cars_new.price.median()

cars_new.price.fillna(value='blank',inplace=True)
# conversion of float datatype to int

cars_new.powerPS = cars_new.powerPS.astype(int)

cars_new.monthOfRegistration = cars_new.monthOfRegistration.astype(int)

cars_new.price = pd.to_numeric(cars_new.price,errors ='coerce') 

#outlier replace may give less rmse value, replacing outliers in output varaible is not suggested

'''

# outlier replace

qpr1 = cars_new.price.quantile(0.25)

qpr2 = cars_new.price.quantile(0.75)

iqpr = qpr2 - qpr1

lower_pr = qpr1 - (1.5*iqpr)

higher_pr = qpr2 + (1.5*iqpr)

cars_new.price[cars_new.price < lower_pr] = lower_pr

cars_new.price[cars_new.price > higher_pr] = higher_pr

'''
cars_new.info()

X_drop = cars_new.copy()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

#cars_new['abtest']=le.fit_transform(cars_new['abtest'].astype('str'))

cars_new['vehicleType']=le.fit_transform(cars_new['vehicleType'].astype('str'))

cars_new['gearbox']=le.fit_transform(cars_new['gearbox'].astype('str'))

cars_new['fuelType']=le.fit_transform(cars_new['fuelType'].astype('str'))

cars_new['notRepairedDamage']=le.fit_transform(cars_new['notRepairedDamage'].astype('str'))





# converted object format to int type 

# given labeling by using label encoder
cars_new.dtypes
X_drop.dtypes
sns.distplot(X_drop.monthOfRegistration)

# More no of cars are registered in the months ranging 3-6
sns.distplot(X_drop.powerPS)
sns.countplot(X_drop.gearbox)

# Manuell geartype is very high when compared to automatik geartype
X_drop.dtypes

X_drop.kilometer = X_drop.kilometer.astype(float)
numerical_d = used_cars.select_dtypes(include='float64')

numerical_d
categorical_d = used_cars.select_dtypes(include ='object')

categorical_d

numerical_d.hist(bins=15,figsize=(15,10),layout=(3,2))

# More no of vehicles present are run for more than 140000 kms

# Most no of vehicles registered between 1995 and 2010
sns.relplot('price','vehicleType',hue='fuelType',data=X_drop)

# coupe and cabrio are the only vehicles present in prices above 150000 with fueltype benzene

# suv and bus  vehicles are mostly diesel type and valued less than 100000
sns.relplot('price','yearOfRegistration',hue='fuelType',data = X_drop)

# Most no of vehicles of diesel vehicletype are registered between 2000 to 2020 and are of price less than 100000
used_cars
# Applying multiple linear regression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
cars_new.columns
X = cars_new.iloc[:,1:]

X

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_n = scaler.fit_transform(X)

X_n

X_norm = pd.DataFrame(X_n,columns = X.columns)

X_norm
y = cars_new.iloc[:,0:1]

y
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size = 0.30)

from sklearn.linear_model import LinearRegression

slm = LinearRegression()

slm.fit(X_train,y_train)

y_pred = slm.predict(X_test)
test_set_rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))

test_set_rmse
cars_new.corr()
from sklearn.metrics import r2_score
lr = LinearRegression()

lr.fit(X_train, y_train)

LinearRegression()

pred_test_lr= lr.predict(X_test)

rmse_linear = np.sqrt(mean_squared_error(y_test,pred_test_lr))

r2_score_linear = r2_score(y_test, pred_test_lr)

print('rmse_linear:',rmse_linear)

print('r2_score_linear:',r2_score_linear)
from sklearn.linear_model import Ridge

rr = Ridge(alpha=0.01)

rr.fit(X_train, y_train) 

pred_test_rr= rr.predict(X_test)

rmse_ridge = np.sqrt(mean_squared_error(y_test,pred_test_rr))

r2_score_ridge = r2_score(y_test, pred_test_rr)

print('rmse_ridge:',rmse_ridge)

print('r2_score_ridge',r2_score_ridge)
from sklearn.linear_model import Lasso

model_lasso = Lasso(alpha=0.01)

model_lasso.fit(X_train, y_train) 

pred_test_lasso= model_lasso.predict(X_test)

rmse_lasso = np.sqrt(mean_squared_error(y_test,pred_test_lasso))

r2_score_lasso = r2_score(y_test, pred_test_lasso)

print('rmse_lasso: ',rmse_lasso )

print('r2_score_lasso: ',r2_score_lasso )
from sklearn.linear_model import ElasticNet

model_enet = ElasticNet(alpha = 0.01)

model_enet.fit(X_train, y_train) 

pred_test_enet= model_enet.predict(X_test)

rmse_elasticnet = np.sqrt(mean_squared_error(y_test,pred_test_enet))

r2_score_elasticnet = r2_score(y_test, pred_test_enet)

print('rmse_elasticnet: ',rmse_elasticnet )

print('r2_score_elasticnet: ',r2_score_elasticnet )



from sklearn.ensemble import RandomForestRegressor

model_random = RandomForestRegressor(random_state=42,n_estimators=100)

model_random.fit(X_train,y_train)

pred_test_random= model_random.predict(X_test)

rmse_random = np.sqrt(mean_squared_error(y_test,pred_test_random))

r2_score_random = r2_score(y_test, pred_test_random)

print('rmse_random: ',rmse_random)

print('r2_score_random:',r2_score_random)

from sklearn.tree import DecisionTreeRegressor   

decision_regressor = DecisionTreeRegressor(random_state = 42)  

decision_regressor.fit(X_train,y_train)

pred_test_decision = decision_regressor.predict(X_test)

rmse_decision = np.sqrt(mean_squared_error(y_test,pred_test_decision))

r2_score_decision = r2_score(y_test,pred_test_decision)

print('rmse_decision: ',rmse_decision)

print('r2_score_decision: ',r2_score_decision)

cars_1 = used_cars.copy()

cars_1
cars_1.groupby(['vehicleType'],sort=False).sum()

daf = cars_1.groupby(['vehicleType','fuelType'])
daf.first()
imp_var = pd.DataFrame(decision_regressor.feature_importances_,[X.columns])

imp_var
print('The rmse value of Linear Regression :  ',        rmse_linear)

print('The rmse value of Lasso Regression :  ',         rmse_lasso)

print('The rmse value of Ridge Regression:  ',          rmse_ridge)

print('The rmse value of Elastinet Regression :  ',     rmse_elasticnet)

print('The rmse value of Decision Tree Regression :  ', rmse_decision)

print('The rmse value of Random Forest Regression :  ', rmse_random)
print('The r2_score value of Linear Regression :  ',  r2_score_linear)

print('The r2_score value of Lasso Regression :  ',   r2_score_lasso)

print('The r2_score of Ridge Regression:  ',          r2_score_ridge)

print('The r2_score of Elastinet Regression :  ',     r2_score_elasticnet)

print('The r2_score of Decision Tree Regression :  ', r2_score_decision)

print('The r2_score of Random Forest Regression :  ', r2_score_random)
'''

#By replacing the ouliers in output

rmse values are like this



The rmse value of Linear Regression :   3075.300093344899

The rmse value of Lasso Regression :   3075.3001705132097

The rmse value of Ridge Regression:   3075.300092000035

The rmse value of Elastinet Regression :   3075.215366155245

The rmse value of Decision Tree Regression :   2055.8594935065275

The rmse value of Random Forest Regression :   1597.671817662537



r2_square values are like this



The r2_score value of Linear Regression :   0.6536285739798497

The r2_score value of Lasso Regression :   0.6536285565968969

The r2_score of Ridge Regression:   0.6536285742827941

The r2_score of Elastinet Regression :   0.6536476593844238

The r2_score of Decision Tree Regression :   0.8452060743681286

The r2_score of Random Forest Regression :   0.9065149415929749

'''