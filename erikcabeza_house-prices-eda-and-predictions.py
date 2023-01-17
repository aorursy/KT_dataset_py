

#Import the packages to work with 

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go



from sklearn.impute import SimpleImputer

from sklearn.impute import KNNImputer

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import Lasso





from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor

from sklearn import svm

from sklearn import metrics

from sklearn.model_selection import cross_val_score



#build the training dataset and the test dataset

train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.head() 
train.info()
#drop columns with a lot of NaN values

train=train.drop(['Alley', 'FireplaceQu','PoolQC','Fence','MiscFeature'], axis = 1) 
#correlation of numerical variables

corr=train.corr()

fig, ax = plt.subplots(figsize=(40,40))   

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns,cmap="YlGnBu",annot=True)
cor_target = abs(corr["SalePrice"])#Selecting highly correlated features

important_numerical_features = cor_target[cor_target>0.5]

important_numerical_features
#there are columns with NaN or 0 values, which represent that the house hasn't that feature. 

#For example, if GarageArea is 0, it means that the house hasn't a garage

#I'm going to substitute 0 for 'None'

for i in ['FullBath','GarageCars','GarageArea']:

    train[i].replace(0, 'None', inplace=True)

        
#correlation between categorical variables and target variable 'SalePrice'

#test ANOVA

#1-MSZoning

from scipy import stats

F, p = stats.f_oneway(train[train.MSZoning=='RL'].SalePrice,train[train.MSZoning=='RM'].SalePrice,train[train.MSZoning=='C (all)'].SalePrice,

                     train[train.MSZoning=='FV'].SalePrice,train[train.MSZoning=='RH'].SalePrice)

print(F)
#2-Street

F, p = stats.f_oneway(train[train.Street=='Pave'].SalePrice,train[train.Street=='Grvl'].SalePrice)

print(F)
#3-LotShape ['Reg', 'IR1', 'IR2', 'IR3']

F, p = stats.f_oneway(train[train.LotShape=='Reg'].SalePrice,train[train.LotShape=='IR1'].SalePrice,train[train.LotShape=='IR2'].SalePrice,

      train[train.LotShape=='IR3'].SalePrice)

print(F)
#4-LandContour ['Lvl', 'Bnk', 'Low', 'HLS']

F, p = stats.f_oneway(train[train.LandContour=='Lvl'].SalePrice,train[train.LandContour=='Bnk'].SalePrice,train[train.LandContour=='Low'].SalePrice,

                     train[train.LandContour=='HLS'].SalePrice)

print(F)
#5-LotConfig ['Inside', 'FR2', 'Corner', 'CulDSac', 'FR3']

F, p = stats.f_oneway(train[train.LotConfig=='Inside'].SalePrice,train[train.LotConfig=='FR2'].SalePrice,train[train.LotConfig=='Corner'].SalePrice,

      train[train.LotConfig=='CulDSac'].SalePrice,train[train.LotConfig=='FR3'].SalePrice)

print(F)
new_train=train.drop(['OverallQual','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd','GarageCars'

                     ,'GarageArea','LotConfig','LandContour','LotShape','Street','MSZoning'],axis=1)

new_train=pd.get_dummies(new_train)
new_corr=new_train.corr()

cor2_target = abs(new_corr["SalePrice"])#Selecting highly correlated features

important_categorical_features = cor2_target[cor2_target>0.5]

important_categorical_features

#the column BsmtQual has nan values to indicate that the basement has no height. Let's replace them with 'None'

train['BsmtQual']=train['BsmtQual'].fillna('None')
chosen_columns=['SalePrice','ExterQual','BsmtQual','KitchenQual','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF','GrLivArea',

               'FullBath','TotRmsAbvGrd','GarageCars','GarageArea']



final_train=train[chosen_columns]
final_train.head()
final_train.info()
#Some of the columns should be categorical instead numerical

final_train['YearBuilt']=final_train['YearBuilt'].apply(str)

final_train['YearRemodAdd']=final_train['YearRemodAdd'].apply(str)

final_train['TotRmsAbvGrd']=final_train['TotRmsAbvGrd'].apply(str)
#add new columns

#hasGarage/hasBasement/hasBath/hasBeenRemodeled
final_train['GarageCars'].value_counts()
hasGarage=[]

for i in final_train.GarageCars:

    if i=='None':

        hasGarage.append('No garage')

    if i==1:

        hasGarage.append('Garage for one or two cars')

    if i==2:

        hasGarage.append('Garage for one or two cars')

    if i==3:

        hasGarage.append('Garage for more than two cars')

    if i==4:

        hasGarage.append('Garage for more than two cars')

        

final_train['hasGarage']=hasGarage
F, p = stats.f_oneway(final_train[final_train.hasGarage=='No garage'].SalePrice,final_train[final_train.hasGarage=='Garage for one or two cars'].SalePrice,

      final_train[final_train.hasGarage=='Garage for more than two cars'].SalePrice)

print(F)
hasBasement=[]

for i in final_train.BsmtQual:

    if i=='None':

        hasBasement.append('No basement')

    if i=='Ex':

        hasBasement.append('Excelent basement')

    if i=='Gd':

        hasBasement.append('Normal basement')

    if i=='TA':

        hasBasement.append('Normal basement')

    if i=='Fa':

        hasBasement.append('Normal basement')



final_train['hasBasement']=hasBasement
F, p = stats.f_oneway(final_train[final_train.hasBasement=='Excelent basement'].SalePrice,final_train[final_train.hasBasement=='Normal basement'].SalePrice,

                     final_train[final_train.hasBasement=='No basement'].SalePrice)

print(F)
final_train['TotalBsmtSF']=final_train['TotalBsmtSF'].astype(float)

hasBigBasement=[]



for i in final_train.TotalBsmtSF:

    if i==0.0:

        hasBigBasement.append('There is no basement')

    if i>0.0:

        hasBigBasement.append('Regular size')

    elif i>=1000.0:

        hasBigBasement.append('Big size')

    

    

final_train['hasBigBasement']=hasBigBasement

        



    

    

F, p = stats.f_oneway(final_train[final_train.hasBigBasement=='There is no basement'].SalePrice,final_train[final_train.hasBigBasement=='Big size'].SalePrice,

                     final_train[final_train.hasBigBasement=='Regular size'].SalePrice)

print(F)
final_train.FullBath.unique()
hasBath=[]

for i in final_train.FullBath:

    if i=='None':

        hasBath.append('No baths')

    if i==1:

        hasBath.append('One bathroom')

    if i==2:

        hasBath.append('More than one bathroom')

    if i==3:

        hasBath.append('More than one bathroom')

    if i==4:

        hasBath.append('More than one bathroom')



final_train['hasBath']=hasBath
F, p = stats.f_oneway(final_train[final_train.hasBath=='No baths'].SalePrice,final_train[final_train.hasBath=='One bathroom'].SalePrice,

                     final_train[final_train.hasBath=='More than one bathroom'].SalePrice)

print(F)
final_train=final_train[['SalePrice','hasGarage','hasBasement','hasBath','hasBigBasement','YearBuilt','1stFlrSF','TotRmsAbvGrd',

                        'KitchenQual','ExterQual','GrLivArea']]
final_train.head()
final_train.info()
fig = px.histogram(final_train, x="SalePrice")

fig.show()
final_train['SalePrice'].skew()
final_train['SalePrice'] = np.log1p(final_train.SalePrice)

final_train['SalePrice'].skew()

fig = px.histogram(final_train, x="SalePrice")

fig.show()
fig = px.histogram(final_train, x="GrLivArea")

fig.show()
final_train['GrLivArea'].skew()
final_train['GrLivArea'] = np.log1p(final_train.GrLivArea)

final_train['GrLivArea'].skew()
fig = px.histogram(final_train, x="GrLivArea")

fig.show()
print(final_train['1stFlrSF'].skew())

final_train['1stFlrSF'] = np.log1p(final_train['1stFlrSF'])

print(final_train['1stFlrSF'].skew())
test.info()
test.FullBath.unique()
hasBathTest=[]

for i in test.FullBath:

    if i==0:

        hasBathTest.append('No baths')

    if i==1:

        hasBathTest.append('One bathroom')

    if i==2:

        hasBathTest.append('More than one bathroom')

    if i==3:

        hasBathTest.append('More than one bathroom')

    if i==4:

        hasBathTest.append('More than one bathroom')



test['hasBath']=hasBathTest
test.BsmtQual.unique()
test['BsmtQual']=test['BsmtQual'].fillna('None')
hasBasementTest=[]

for i in test.BsmtQual:

    if i=='None':

        hasBasementTest.append('No basement')

    if i=='Ex':

        hasBasementTest.append('Excelent basement')

    if i=='Gd':

        hasBasementTest.append('Normal basement')

    if i=='TA':

        hasBasementTest.append('Normal basement')

    if i=='Fa':

        hasBasementTest.append('Normal basement')



test['hasBasement']=hasBasementTest
test['TotalBsmtSF']=test['TotalBsmtSF'].fillna(0.0)
hasBigBasementTest=[]



for i in test.TotalBsmtSF:

    if i==0.0:

        hasBigBasementTest.append('There is no basement')

    if i>0.0:

        hasBigBasementTest.append('Regular size')

    elif i>=1000.0:

        hasBigBasementTest.append('Big size')

    

    

test['hasBigBasement']=hasBigBasementTest

        
test.GarageCars.unique()
test['GarageCars']=test['GarageCars'].fillna(0.0)
test.GarageCars.unique()
hasGarageTest=[]

for i in test.GarageCars:

    if i==0.0:

        hasGarageTest.append('No garage')

    if i==1.0:

        hasGarageTest.append('Garage for one or two cars')

    if i==2.0:

        hasGarageTest.append('Garage for one or two cars')

    if i==3.0:

        hasGarageTest.append('Garage for more than two cars')

    if i==4.0:

        hasGarageTest.append('Garage for more than two cars')

    if i==5.0:

        hasGarageTest.append('Garage for more than two cars')

        

test['hasGarage']=hasGarageTest
test=test[['hasGarage','hasBasement','hasBath','hasBigBasement','YearBuilt','1stFlrSF','TotRmsAbvGrd','KitchenQual','ExterQual','GrLivArea']]
test['YearBuilt']=test['YearBuilt'].apply(str)

test['TotRmsAbvGrd']=test['TotRmsAbvGrd'].apply(str)
test.info()
#there is a nan value in kitchen qual->use median to fill it

test['KitchenQual']=test['KitchenQual'].fillna(test['KitchenQual'].mode(), inplace=True)
test.GrLivArea.skew()
test['GrLivArea'] = np.log1p(test['GrLivArea'])

test.GrLivArea.skew()
test['1stFlrSF'].skew()
test['1stFlrSF'] = np.log1p(test['1stFlrSF'])

test['1stFlrSF'].skew()
final_train.info()
cat_columns=[column_name for column_name in final_train.columns if final_train[column_name].dtype=="object"]

num_columns=[column_name for column_name in final_train.columns if final_train[column_name].dtype in ["int64", "float64"]]
print(num_columns)

print(cat_columns)
# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('cat', categorical_transformer, cat_columns)

    ])
import xgboost as xgb

model=xgb.XGBRegressor(max_depth=3,eta=0.05,min_child_weight=4)

#build the pipeline

pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])
features=final_train.drop('SalePrice',axis=1)

y=final_train['SalePrice']
pipeline.fit(features,y)

predictions=pipeline.predict(test)
finalPred=np.expm1(predictions)
sample_submission=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

submission = pd.DataFrame({'Id':sample_submission['Id'],'SalePrice':finalPred})

submission
filename = 'Submission.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)