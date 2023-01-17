# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

#import tensorflow as tf

#from tensorflow import keras

#from tensorflow.keras import layers



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

dataset = pd.read_csv('../input/train.csv')

dataset.drop('Id',axis=1,inplace=True)
#tf.logging.set_verbosity(tf.logging.INFO)
dataset.info()
#Lets view the corr matrix between columns

plt.figure(figsize=(20,20))

sns.heatmap(dataset.corr(),annot=True,cmap='coolwarm')
datasetcorr = dataset.corr()

crit1 = datasetcorr['SalePrice'] > 0.5

crit2 = datasetcorr['SalePrice'] < -0.5

crit = crit1 | crit2

datasetcorr[crit]['SalePrice']
# Based on the above corr below continous columns are the important 

column_names = ['OverallQual','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd',

                'GarageCars','GarageArea','SalePrice']
contdataset = dataset[column_names]
contdataset.head()
#YearBuilt,YearRemodAdd - Lets Bucketize these columns later as they are years 

# Discrete columns : FullBath,TotRmsAbvGrd,GarageCars
catdataset = dataset.select_dtypes(include=np.object)
catdataset.loc[catdataset['MSZoning'] == 'C (all)','MSZoning'] = 'C' # Chaning the C (all) -> C as thats more comfirtable for any processing
# Removing NULL values in Categorical variables
# Prints and displays 

def chkcatcolsdet(ds):

    nullcatcols = []

    for col in ds.columns:

            print("%s count:%d, nullvalues: %s, values: %s "%(col,len(ds[col].unique()),ds[col].isnull().sum(),ds[col].unique()))

            if ds[col].isnull().sum(): nullcatcols.append(col) 

    return nullcatcols
nullcatcols1 = chkcatcolsdet(catdataset)
print(nullcatcols1)
def repmincntvals(ds,nullcatcols):

    for col in nullcatcols:

        columnval = pd.DataFrame(ds.groupby([col])[col].count()).transpose().min().index[0]

        ds.groupby(by=col)[col].count()

        ds.loc[ds[col].isnull(),col] = columnval

        ds.groupby(by=col)[col].count()
repmincntvals(catdataset,nullcatcols1)
chkcatcolsdet(catdataset) # no more Null columns
# Lets drop Alley,FireplaceQu,PoolQC,Fence,MiscFeature as most of the values are NULL

#Alley count:3, nullvalues: 1369, values: [nan 'Grvl' 'Pave'] 

#FireplaceQu count:6, nullvalues: 690, values: [nan 'TA' 'Gd' 'Fa' 'Ex' 'Po'] 

#PoolQC count:4, nullvalues: 1453, values: [nan 'Ex' 'Fa' 'Gd'] 

#Fence count:5, nullvalues: 1179, values: [nan 'MnPrv' 'GdWo' 'GdPrv' 'MnWw'] 

#MiscFeature count:5, nullvalues: 1406, values: [nan 'Shed' 'Gar2' 'Othr' 'TenC'] 

catdataset.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
len(catdataset.columns)
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

def labelencoderdataset(ds,catcols):

    labencdataset = pd.DataFrame()

    for i in range(0,len(catcols)):

        if (int(ds[str(catcols[i])].isna().sum())):

            next

        else:

            print ("label encoding column : %s"%str(catcols[i]))

            labencdataset[catcols[i]] = labelencoder.fit_transform(ds[catcols[i]])

    return labencdataset
catlabencdataset = labelencoderdataset(catdataset,catdataset.columns)
# Lets Scale the Continous data

contdataset1 = contdataset
contdataset.columns
# - Lets Bucketize these columns later as they are years 

# Discrete columns : OverallQual,FullBath,TotRmsAbvGrd,GarageCars,YearBuilt,YearRemodAdd

#Continous Variables : 'TotalBsmtSF', '1stFlrSF','GrLivArea','GarageArea'



contcontdataset = contdataset[['TotalBsmtSF', '1stFlrSF','GrLivArea','GarageArea']]

contdiscdataset = contdataset[['OverallQual','FullBath','TotRmsAbvGrd','GarageCars']]

contbuckdataset = contdataset[['YearBuilt','YearRemodAdd']]
contbuckdataset.min(),contbuckdataset.max()
year_ranges = ["[{0} - {1})".format(year, year + 10) for year in range(1870, 2010, 10)]



contbuckdataset['YearBuiltrange'] = pd.cut(contbuckdataset['YearBuilt'],labels=year_ranges,bins=len(year_ranges))

contbuckdataset['YearRemodAddrange'] = pd.cut(contbuckdataset['YearRemodAdd'],labels=year_ranges,bins=len(year_ranges))
contbuckdataset['YearBuiltrange'].value_counts().plot(kind='bar')
contbuckdataset['YearRemodAddrange'].value_counts().plot(kind='bar')
contbuckdataset.drop(['YearBuilt','YearRemodAdd'],axis=1,inplace=True)
contbuckdataset.info()
contbucklabdataset = labelencoderdataset(contbuckdataset,contbuckdataset.columns)
nullcatcols1 = chkcatcolsdet(contdiscdataset)
labeldataset = pd.DataFrame(contdataset['SalePrice'])
contdataset1.drop(['SalePrice'],axis=1,inplace=True)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

contcontdataset2 = pd.DataFrame(scaler.fit_transform(contcontdataset),columns=contcontdataset.columns)
finaldataset = pd.concat([contcontdataset2,contdiscdataset,contbucklabdataset,catlabencdataset,labeldataset],axis=1)
finaldataset.head()
# Lets Try Z-Score to detect outliers

from scipy import stats

import numpy as np

z = np.abs(stats.zscore(finaldataset))

print(z)
threshold = 10

len(np.where(z > 3.9)[0])
# Lets take backup of existing dataset - just in case and remove the above 100 outliers

finaldataset1 = finaldataset
finaldataset = finaldataset[(z < 3.9).all(axis=1)]
X = finaldataset.drop(['SalePrice'],axis=1)

Y = finaldataset['SalePrice'] # np.log(finaldataset['SalePrice']) - lets try the skewed data later

Y = np.log(finaldataset['SalePrice']) # Metric to calculate is log of the price
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.33, random_state=101)
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor
Decstreemodel = DecisionTreeRegressor()
Decstreemodel.fit(X_train,y_train)
from sklearn.metrics import mean_squared_error
y_pred_dectree = Decstreemodel.predict(X_test)
mean_squared_error(y_test,y_pred_dectree)**0.5
y_pred_dectree[:2],y_test[:2]
from sklearn.model_selection import RandomizedSearchCV

param_dist = {

 'n_estimators': [50,100,150,200,250,350,450,550,650,750,1000],

 'learning_rate' : [0.001,0.01,0.05,0.1,0.3,1],

 'loss' : ['linear', 'square', 'exponential']

 }

AdaBoostRgr = RandomizedSearchCV(AdaBoostRegressor(DecisionTreeRegressor(max_depth=50)),

 param_distributions = param_dist,

 cv=4,

 n_iter = 25,

 n_jobs=-1)

AdaBoostRgr.fit(X_train,y_train)
AdaBoostRgr.best_params_
AdaBoostRgr.best_params_['n_estimators']
# Create the dataset

AdaBoostfinalrgr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=40),

                          n_estimators=AdaBoostRgr.best_params_['n_estimators'], random_state=33,learning_rate=1,loss='square')
AdaBoostfinalrgr.fit(X_train,y_train)
y_pred_adaboost = AdaBoostfinalrgr.predict(X_test)
(y_pred_adaboost[:5],y_test[:5])
from sklearn.metrics import mean_squared_error,median_absolute_error
mean_squared_error(y_test,y_pred_adaboost)**0.5
from sklearn import ensemble
param_dist = {

 'n_estimators': [50,100,150,200,250,350,450,550,650,750,1000],

 'learning_rate' : [0.001,0.01,0.05,0.1,0.3,0.7,1],

 'loss' : ['ls', 'lad', 'huber', 'quantile'],

 }

GradBoostRgrs = RandomizedSearchCV(ensemble.GradientBoostingRegressor(max_depth=10),param_dist)
GradBoostRgrs.fit(X_train,y_train)
GradBoostRgrs.best_params_
gradboost = ensemble.GradientBoostingRegressor(max_depth=10,n_estimators=GradBoostRgrs.best_params_['n_estimators'],loss=GradBoostRgrs.best_params_['loss'],learning_rate=GradBoostRgrs.best_params_['learning_rate'])
gradboost.fit(X_train,y_train)
gradboost_pred = gradboost.predict(X_test)
mse_gradboost = mean_squared_error(y_test, gradboost_pred)

mse_adaboost = mean_squared_error(y_test,y_pred_adaboost)
mse_gradboost**0.5,mse_adaboost**0.5
from xgboost import XGBRegressor
param_dist = {

 'n_estimators': [100,200,300,400,500,600,700,800,900,1000],

 'learning_rate' : [0.001,0.01,0.02,0.05,0.1,0.3,0.5,0.7,1],

 }

XGBRgrs = RandomizedSearchCV(XGBRegressor(max_depth=10),param_dist)
XGBRgrs.fit(X_train,y_train)
XGBRgrs.best_params_
xgb = XGBRegressor(max_depth=10,n_estimators=XGBRgrs.best_params_['n_estimators'],learning_rate=XGBRgrs.best_params_['learning_rate'])
xgb.fit(X_train,y_train)
xgb_pred = xgb.predict(X_test)
mse_gradboost = mean_squared_error(y_test, gradboost_pred)

mse_adaboost = mean_squared_error(y_test,y_pred_adaboost)

mse_xgboost = mean_squared_error(y_test,xgb_pred)
print(mse_gradboost**0.5,mse_adaboost**0.5,mse_xgboost**0.5)
testdataset = pd.read_csv('../input/test.csv')
# Based on the above corr below continous columns are the important 

column_names = ['OverallQual','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd',

                'GarageCars','GarageArea']

resultid = testdataset['Id']

testdataset.drop(['Id'],axis=1,inplace=True)

conttestdataset = testdataset[column_names]

cattestdataset = testdataset.select_dtypes(include=np.object)
cattestdataset.loc[cattestdataset['MSZoning'] == 'C (all)','MSZoning'] = 'C'

nullcatcols1 = chkcatcolsdet(cattestdataset)

repmincntvals(cattestdataset,nullcatcols1)

nullcatcols1 = chkcatcolsdet(cattestdataset)

cattestdataset.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
catlabenctestdataset = labelencoderdataset(cattestdataset,cattestdataset.columns)
contconttestdataset = conttestdataset[['TotalBsmtSF', '1stFlrSF','GrLivArea','GarageArea']]

contdisctestdataset = conttestdataset[['OverallQual','FullBath','TotRmsAbvGrd','GarageCars']]

contbucktestdataset = conttestdataset[['YearBuilt','YearRemodAdd']]
nullcatcols3 = chkcatcolsdet(contdisctestdataset)
repmincntvals(contdisctestdataset,nullcatcols3)
# Test Dataset has GarageArea/TotalBsmtSF as NULL values - so Lets replace with mean values.

TotalBsmtSF_array = contconttestdataset[contconttestdataset["TotalBsmtSF"]!=np.nan]["TotalBsmtSF"]

contconttestdataset["TotalBsmtSF"].replace(np.nan,TotalBsmtSF_array.mean())

GarageArea_array = contconttestdataset[contconttestdataset["GarageArea"]!=np.nan]["GarageArea"]

contconttestdataset["GarageArea"].replace(np.nan,GarageArea_array.mean())

# Replace 660 column value with the mean value of TotalBDmtSF

contconttestdataset.loc[660,'TotalBsmtSF'] = np.mean(contconttestdataset['TotalBsmtSF'])
contconttestdataset[contconttestdataset['TotalBsmtSF'].isnull()]
contbucktestdataset['YearBuiltrange'] = pd.cut(contbucktestdataset['YearBuilt'],labels=year_ranges,bins=len(year_ranges))

contbucktestdataset['YearRemodAddrange'] = pd.cut(contbucktestdataset['YearRemodAdd'],labels=year_ranges,bins=len(year_ranges))

contbucktestdataset.drop(['YearBuilt','YearRemodAdd'],axis=1,inplace=True)

contbucklabtestdataset = labelencoderdataset(contbucktestdataset,contbucktestdataset.columns)
contconttestdataset2 = pd.DataFrame(scaler.fit_transform(contconttestdataset),columns=contconttestdataset.columns)
contconttestdataset2.isna().sum()
contconttestdataset2[contconttestdataset2['GarageArea'].isnull()]

contconttestdataset2.loc[1116,'GarageArea'] = np.mean(contconttestdataset2['GarageArea'])
contconttestdataset2[contconttestdataset2['GarageArea'].isnull()]
finaldataset = pd.concat([contconttestdataset2,contdisctestdataset,contbucklabtestdataset,catlabenctestdataset],axis=1)
finaldataset.dropna()
X_test_csv = finaldataset
X_test_csv[X_test_csv['TotalBsmtSF'].isnull() == True]
Y_test_adaboostcsv = AdaBoostfinalrgr.predict(X_test_csv)

Y_test_xgboostcsv = xgb.predict(X_test_csv)
Y_test_csv = np.exp(Y_test_xgboostcsv)
finalpredresult = pd.concat([resultid,pd.DataFrame(Y_test_csv,columns=['SalePrice'])],axis=1,names=['Id','SalePrice'])
finalpredresult
finalpredresult.to_csv('sasanka_submission2.csv', index=False)