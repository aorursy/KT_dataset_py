# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/train.csv")
dataset.shape
dataset.drop('Id',axis=1,inplace=True)
# Chi-Square Test for categorical variables. T-Test Z-test Annova
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(15,15))

colcorrdataset = dataset.drop('SalePrice',axis=1)

sns.heatmap(dataset.drop('SalePrice',axis=1).corr(),cmap='coolwarm',annot=True)
c = colcorrdataset.corr().abs()

s = c.unstack()

so = s.sort_values(kind="quicksort")
crit1 = so > 0.8

crit2 = so < 1

crit = crit1 & crit2

so[crit]
datasetcorr = dataset.corr()

crit1 = datasetcorr['SalePrice'] > 0.5

crit2 = datasetcorr['SalePrice'] < -0.5

crit = crit1 | crit2

datasetcorr[crit]['SalePrice']
# Based on the above corr below continous columns are the important 

column_names = ['OverallQual','YearBuilt','YearRemodAdd','TotalBsmtSF','GrLivArea','FullBath','GarageArea','SalePrice']
contdataset = dataset[column_names]
catdataset = dataset.select_dtypes(include=np.object)
catdataset.loc[catdataset['MSZoning'] == 'C (all)','MSZoning'] = 'C' # Chaning the C (all) -> C as thats more comfirtable for any processing
# Prints and displays 

def chkcatcolsdet(ds):

    nullcatcols = []

    for col in ds.columns:

            print("%s count:%d, nullvalues: %s, values: %s "%(col,len(ds[col].unique()),ds[col].isnull().sum(),ds[col].unique()))

            if ds[col].isnull().sum(): nullcatcols.append(col) 

    return nullcatcols



def repmincntvals(ds,nullcatcols):

    for col in nullcatcols:

        columnval = pd.DataFrame(ds.groupby([col])[col].count()).transpose().min().index[0]

        ds.groupby(by=col)[col].count()

        ds.loc[ds[col].isnull(),col] = columnval

        ds.groupby(by=col)[col].count()
nullcatcols1 = chkcatcolsdet(catdataset)

repmincntvals(catdataset,nullcatcols1)

chkcatcolsdet(catdataset) 
catdataset.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
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
# ['OverallQual','YearBuilt','YearRemodAdd','TotalBsmtSF','GrLivArea','FullBath','GarageArea','SalePrice']

# - Lets Bucketize these columns later as they are years 

# Discrete columns : OverallQual,FullBath,TotRmsAbvGrd,GarageCars,YearBuilt,YearRemodAdd

#Continous Variables : 'TotalBsmtSF', '1stFlrSF','GrLivArea','GarageArea'



contcontdataset = contdataset[['TotalBsmtSF', 'GrLivArea','GarageArea']]

contdiscdataset = contdataset[['OverallQual','FullBath']]

contbuckdataset = contdataset[['YearBuilt','YearRemodAdd']]



year_ranges = ["[{0} - {1})".format(year, year + 10) for year in range(1870, 2010, 10)]



contbuckdataset['YearBuiltrange'] = pd.cut(contbuckdataset['YearBuilt'],labels=year_ranges,bins=len(year_ranges))

contbuckdataset['YearRemodAddrange'] = pd.cut(contbuckdataset['YearRemodAdd'],labels=year_ranges,bins=len(year_ranges))
contbuckdataset.drop(['YearBuilt','YearRemodAdd'],axis=1,inplace=True)
contbucklabdataset = labelencoderdataset(contbuckdataset,contbuckdataset.columns)
nullcatcols1 = chkcatcolsdet(contdiscdataset)
labeldataset = pd.DataFrame(contdataset['SalePrice'])
contdataset1.drop(['SalePrice'],axis=1,inplace=True)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

contcontdataset2 = pd.DataFrame(scaler.fit_transform(contcontdataset),columns=contcontdataset.columns)
finaldataset = pd.concat([contcontdataset2,contdiscdataset,contbucklabdataset,catlabencdataset,labeldataset],axis=1)
from sklearn.ensemble import RandomForestRegressor

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import train_test_split
# Lets see what are the important features based on RadomForestClassifier
X = finaldataset.drop(['SalePrice'],axis=1)

#Y = finaldataset['SalePrice'] # np.log(finaldataset['SalePrice']) - lets try the skewed data later

Y = np.log(finaldataset['SalePrice']) # Metric to calculate is log of the price
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.33, random_state=101)
from sklearn.decomposition import PCA
pca = PCA(.8)
X_pca = pca.fit_transform(X)
X_pca.shape
pca.explained_variance_ratio_
X_pca.shape,Y.shape
x_train,x_test,y_train,y_test = train_test_split(X_pca,Y,test_size=0.1)
from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import RandomizedSearchCV



param_dist = {

 'n_estimators': [50,100,150,200,250,350,450,550,650,750,1000],

 'learning_rate' : [0.001,0.01,0.05,0.1,0.3,1],

 'loss' : ['linear', 'square', 'exponential']

 }

AdaBoostRgr = RandomizedSearchCV(AdaBoostRegressor(DecisionTreeRegressor(max_depth=50)),

 param_distributions = param_dist,

 cv=4,

 n_iter = 50,

 n_jobs=-1)



AdaBoostRgr.fit(x_train,y_train)
# Create the dataset

AdaBoostfinalrgr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=50),

                          n_estimators=AdaBoostRgr.best_params_['n_estimators'], random_state=3,learning_rate=AdaBoostRgr.best_params_['learning_rate'],loss='square')
AdaBoostfinalrgr.fit(x_train,y_train)
y_pred_adaboost = AdaBoostfinalrgr.predict(x_test)
mean_squared_error(y_test,y_pred_adaboost)**0.5
# Lets Try calculating zscore and figure outliers
X = finaldataset.drop(['SalePrice'],axis=1)

#Y = finaldataset['SalePrice'] # np.log(finaldataset['SalePrice']) - lets try the skewed data later

Y = np.log(finaldataset['SalePrice']) # Metric to calculate is log of the price
# Lets Scale the data and then apply zscore to figure out outliers

from sklearn.preprocessing import StandardScaler

stdscl = StandardScaler()

X_scale = stdscl.fit_transform(X)



from scipy import stats

import numpy as np

z = np.abs(stats.zscore(X_scale))

print(z)
threshold = 10

print(np.where(z > threshold))
X_scale_ds = pd.DataFrame(X_scale)

X_scale_ds.drop(np.where(z > threshold)[0],inplace = True)
Y.drop(np.where(z > threshold)[0],inplace = True)
#X = finaldataset.drop(['SalePrice'],axis=1)

#Y = np.log(labeldataset)
x_train,x_test,y_train,y_test = train_test_split(X_scale_ds,Y,test_size=0.2,random_state=7)
from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import RandomizedSearchCV

param_dist = {

 'n_estimators': [50,100,150,200,250,350,450,550,650,750,1000],

 'learning_rate' : [0.01,0.02,0.05,0.07,0.1,0.3,0.5,0.7,1],

 'loss' : ['linear', 'square', 'exponential']

 }

AdaBoostRgr = RandomizedSearchCV(AdaBoostRegressor(DecisionTreeRegressor(max_depth=100)),

 param_distributions = param_dist,

 cv=4,

 n_iter = 25,

 n_jobs=-1)

AdaBoostRgr.fit(x_train,y_train)
# Create the dataset

AdaBoostfinalrgr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=100),

                          n_estimators=AdaBoostRgr.best_params_['n_estimators'], random_state=33,learning_rate=AdaBoostRgr.best_params_['learning_rate'],loss='square')
AdaBoostfinalrgr.fit(x_train,y_train)
y_pred_adaboost_o = AdaBoostfinalrgr.predict(x_test)
mean_squared_error(y_test,y_pred_adaboost_o)**0.5
y_pred_adaboost_o[:6]
from xgboost import XGBRegressor
param_dist = {

 'n_estimators': [50,100,150,200,250,350,450,550,650,750,1000],

 'learning_rate' : [0.5,0.6,0.7,0.8,0.9,1]

 }

XGBRgrs = RandomizedSearchCV(XGBRegressor(max_depth=100),param_dist)
XGBRgrs.fit(x_train,y_train)
xgb = XGBRegressor(max_depth=100,n_estimators=XGBRgrs.best_params_['n_estimators'],learning_rate=XGBRgrs.best_params_['learning_rate'])
xgb.fit(x_train,y_train)
xgb_pred = xgb.predict(x_test)
mean_squared_error(y_test,xgb_pred)**0.5
testdataset = pd.read_csv('../input/test.csv')
column_names = ['OverallQual','YearBuilt','YearRemodAdd','TotalBsmtSF','GrLivArea','FullBath','GarageArea']

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
contconttestdataset = conttestdataset[['TotalBsmtSF', 'GrLivArea','GarageArea']]

contdisctestdataset = conttestdataset[['OverallQual','FullBath']]

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
contbucktestdataset['YearBuiltrange'] = pd.cut(contbucktestdataset['YearBuilt'],labels=year_ranges,bins=len(year_ranges))

contbucktestdataset['YearRemodAddrange'] = pd.cut(contbucktestdataset['YearRemodAdd'],labels=year_ranges,bins=len(year_ranges))

contbucktestdataset.drop(['YearBuilt','YearRemodAdd'],axis=1,inplace=True)

contbucklabtestdataset = labelencoderdataset(contbucktestdataset,contbucktestdataset.columns)
contconttestdataset2 = pd.DataFrame(scaler.fit_transform(contconttestdataset),columns=contconttestdataset.columns)
contconttestdataset2[contconttestdataset2['GarageArea'].isnull()]

contconttestdataset2.loc[1116,'GarageArea'] = np.mean(contconttestdataset2['GarageArea'])
finaltestdataset = pd.concat([contconttestdataset2,contdisctestdataset,contbucklabtestdataset,catlabenctestdataset],axis=1)
testcontdataset = finaltestdataset[['TotalBsmtSF', 'GrLivArea', 'GarageArea', 'OverallQual']]
testcatdataset = finaltestdataset[catdataset.columns]
finaltestdataset.columns
finaldataset.columns
y_adaboost_csv = AdaBoostRgr.predict(stdscl.fit_transform(finaltestdataset))
Y_test_csv = np.exp(y_adaboost_csv)
finalpredresult = pd.concat([resultid,pd.DataFrame(Y_test_csv,columns=['SalePrice'])],axis=1,names=['Id','SalePrice'])
finalpredresult
x_train.shape
finalpredresult.to_csv('adaboost_zscore_standscale.csv', index=False)
''' DNN Can be tried in another Kernel but i couldnt get more accuracy than Adaboost as above, hence commenting it for now

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from keras import regularizers

# import noise layer

from keras.layers import GaussianNoise



X = finaldataset.drop(['SalePrice'],axis=1)

Y = np.log(labeldataset)



import seaborn as sns

vards = pd.DataFrame(X.var())



vards.plot(kind='hist')



from sklearn.decomposition import PCA

pca = PCA(0.95)

x_pca = pca.fit_transform(X)

pca.explained_variance_ratio_



x_train,x_test,y_train,y_test = train_test_split(pd.DataFrame(x_pca),Y,test_size=0.1)



from sklearn.preprocessing import StandardScaler

stdscale = StandardScaler()

x_train_scale = stdscale.fit_transform(x_train)

x_test_scale = stdscale.fit_transform(x_test)



def dnnmodel():

    model = Sequential()

    model.add(Dense(256,input_shape=(45,),activation="relu"))

    model.add(GaussianNoise(0.1))

    model.add(Dense(256,activation="relu"))

    model.add(Dense(256,activation="relu"))

    model.add(Dense(256,activation="relu"))

    model.add(Dense(256,activation="relu"))

    model.add(Dense(1))

    # Compile model

    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['acc','mean_squared_error'])

    return model



model = dnnmodel()

model.summary()



# simple early stopping

#es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1,patience=200)

# model fit

history = model.fit(x_train,y_train,epochs=100,batch_size=30,validation_split=0.1)



import matplotlib.pyplot as plt

# Plot training & validation accuracy values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



y_test_dnn = model.predict(x_test)

mean_squared_error(y_test,y_test_dnn)**0.5

(np.exp(y_test_dnn[:5]),np.exp(y_test[:5]))'''