# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from sklearn.ensemble import RandomForestRegressor



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

from sklearn.model_selection import KFold

import seaborn as sns

from scipy import stats 

from scipy.stats import norm, skew ,zscore#for some statistics

import matplotlib.pyplot as plt  # Matlab-style plotting



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.linear_model import LinearRegression,Lasso,ElasticNet

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
'''

PATH1 = os.path.join(os.getcwd(), os.path.join('data', 'trainHouses.csv'))

PATH2= os.path.join(os.getcwd(), os.path.join('data', 'testHouses.csv'))



train = pd.read_csv(PATH1, delimiter=',')

test = pd.read_csv(PATH2, delimiter=',')



train.head()

'''

train=pd.read_csv("../input/train.csv")

test=pd.read_csv('../input/test.csv')

train.shape,test.shape








train.info()
test.columns^train.columns


numerc_fet=train.select_dtypes(include=np.number)
corr=numerc_fet.corr()
corr['SalePrice'].sort_values(ascending=False)[:9]

#Handling outliers

plt.scatter(Xtrain['GrLivArea'], ytrain)

plt.ylabel('SalePrice')

plt.xlabel('GrLivArea')

plt.show()
train=train[train['GrLivArea']<3200]
ytrain=train['SalePrice']

Xtrain=train.drop('SalePrice',axis=1)
Xtrain.shape
test.columns^Xtrain.columns  #for a check
ytrain.shape,Xtrain.shape,test.shape
totalData=pd.concat([Xtrain,test])

totalData.shape
missing=totalData.isnull().sum().sort_values(ascending=False)

missing=missing[missing>0]

missing

#for catergorical variables, we replece missing data with None

Miss_cat=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 

          'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 

          'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass']

for col in Miss_cat:

    totalData[col].fillna('None',inplace=True)

# for numerical variables, we replace missing value with 0

Miss_num=['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 

          'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'] 

for col in Miss_num:

    totalData[col].fillna(0, inplace=True)
rest_val=['MSZoning','Functional','Utilities','Exterior1st', 'SaleType','Electrical', 'Exterior2nd','KitchenQual']

for col in rest_val:

    totalData[col].fillna(totalData[col].mode()[0],inplace=True)
totalData['LotFrontage']=totalData.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
totalData=totalData.drop('Id',axis=1)


sns.distplot(ytrain , fit=norm)



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(ytrain)

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(ytrain, plot=plt)

plt.show()
ytrain.skew()
ytrain=np.log(ytrain)


ytrain=pd.DataFrame(ytrain)

ytrain.plot.hist()
ytrain.head(2)
#convert the numeric values into string becuse there are many repetition 

totalData['YrSold'] = totalData['YrSold'].astype(str)

totalData['MoSold'] = totalData['MoSold'].astype(str)

totalData['MSSubClass'] = totalData['MSSubClass'].astype(str)

totalData['OverallCond'] = totalData['OverallCond'].astype(str)



from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(totalData[c].values)) 

    totalData[c] = lbl.transform(list(totalData[c].values))



# shape        

print('Shape totalData: {}'.format(totalData.shape))
totalData=pd.DataFrame(totalData)

ytrain=pd.DataFrame(ytrain)
numeric_feats = totalData.dtypes[totalData.dtypes != "object"].index

string_feats=totalData.dtypes[totalData.dtypes == "object"].index



numeric_feats
string_feats
totalData.shape
#totalData.plot(kind="density", figsize=(50,50))

#totalData.plot.hist()

#pd.plotting.scatter_matrix(totalData,figsize=(12,12))
dumies = pd.get_dummies(totalData[string_feats])

print(dumies.shape)
totalData=pd.concat([totalData,dumies],axis='columns')
totalData.shape
totalData=totalData.drop(string_feats,axis=1)
x=len(ytrain)
totalData=pd.DataFrame(totalData)
train_feature=totalData.iloc[:x,:]

test_feature=totalData.iloc[x:,:]
#from sklearn.preprocessing import MinMaxScaler



#sc_X = MinMaxScaler()

train_sc = train_feature

test_sc = test_feature



ytrain_sc=ytrain
train_feature.shape,test_feature.shape,ytrain.shape
train_sc=pd.DataFrame(train_sc)

ytrain_sc=pd.DataFrame(ytrain_sc)
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(train_sc,ytrain_sc,test_size=0.2,random_state=42)
model1= LinearRegression()

model1.fit(X_train,Y_train)
ypre1=model1.predict(X_test)
mean = mean_squared_error(y_pred=ypre1,y_true=Y_test)

r2_scor = r2_score(y_pred=ypre1,y_true=Y_test)

absloute = mean_absolute_error(y_pred=ypre1,y_true=Y_test)

print(mean,r2_scor,absloute)
# Test Options and Evaluation Metrics

num_folds = 5

scoring = "neg_mean_squared_error"

# Spot Check Algorithms

models = []

models.append(('LR', LinearRegression()))

models.append(('LASSO', Lasso()))

models.append(('EN', ElasticNet()))

models.append(('KNN', KNeighborsRegressor()))

models.append(('CART', DecisionTreeRegressor()))

models.append(('SVR', SVR()))

models.append(('RFR', RandomForestRegressor()))





results = []

names = []

for name, model in models:

    kfold = KFold(n_splits=num_folds, random_state=0)

    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold,    scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(),   cv_results.std())

    print(msg)
Y_train.skew()
model4= Lasso()

model4.fit(X_train,Y_train)
ypre4=model4.predict(X_test)
mean = mean_squared_error(y_pred=ypre4,y_true=Y_test)

r2_scor = r2_score(y_pred=ypre4,y_true=Y_test)

absloute = mean_absolute_error(y_pred=ypre4,y_true=Y_test)

print(mean,r2_scor,absloute)
submission = pd.DataFrame()

submission['Id'] = test.Id
test_sc=pd.DataFrame(test_sc)

yprett=model4.predict(test_sc)

final_predictions = np.exp(yprett)

print ("Original predictions are: \n", yprett[:5], "\n")

print ("Final predictions are: \n", final_predictions[:5])
submission['SalePrice'] = final_predictions

submission.head()
#yprett=np.log1p(yprett)
submission.to_csv('submission1.csv', index=False)
