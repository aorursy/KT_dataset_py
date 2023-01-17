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
# Supress Warnings

import warnings
warnings.filterwarnings('ignore')

# Import Libraries to read the data
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# Read the train data
df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df.head()
# Inspection Code (to understand the data)
df.shape
df.info()
df.describe()
# changing datatyppe of numeric columns 
df['MSSubClass']=df['MSSubClass'].astype('object')
df['OverallQual']=df['OverallQual'].astype('object')
df['OverallCond']=df['OverallCond'].astype('object')
df['BsmtFullBath']=df['BsmtFullBath'].astype('object')
df['BsmtHalfBath']=df['BsmtHalfBath'].astype('object')
df['FullBath']=df['FullBath'].astype('object')
df['HalfBath']=df['HalfBath'].astype('object')
df['BedroomAbvGr']=df['BedroomAbvGr'].astype('object')
df['KitchenAbvGr']=df['KitchenAbvGr'].astype('object')
df['TotRmsAbvGrd']=df['TotRmsAbvGrd'].astype('object')
df['Fireplaces']=df['Fireplaces'].astype('object')
df['GarageCars']=df['GarageCars'].astype('object')
df.info()
# Null percentage
round(100*df.isnull().sum()/len(df.index),4)
# Drop columns with null percentage greater than 45%
df = df[df.columns[df.isnull().mean() < 0.45]]
df.head()
df.shape
#Extracting the categorical data to know the value counts
categorical=df[['MSSubClass', 'MSZoning','Street',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'OverallQual', 'OverallCond','RoofStyle','RoofMatl', 'Exterior1st', 
        'Exterior2nd', 'MasVnrType','ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1','BsmtFinType2', 'Heating', 'HeatingQC',
         'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
         'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars',
       'CentralAir', 'Electrical','KitchenQual','Functional','GarageType','GarageYrBlt'
        ,'GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition']].copy()
def val_cnt(x):
    print('\n')
    print(x.value_counts(normalize=True,dropna=False))
categorical.apply(val_cnt)
df.drop(['Street','LandContour','Utilities','LandSlope','Condition1','Condition2',
         'BldgType','RoofMatl','ExterCond','BsmtCond','BsmtFinType2','Heating',
         'CentralAir','Electrical','Functional','GarageQual','GarageCond',
         'PavedDrive','SaleType','SaleCondition','BsmtHalfBath','KitchenAbvGr'],axis=1,inplace=True)
# Drop columns which are extra with respect to business
df.drop(['Id'],axis=1,inplace=True)
df.drop(['MiscVal'],axis=1,inplace=True)
df.drop(['MoSold'],axis=1,inplace=True)
df.shape
# Box plot analysis for numerical columns
plt.figure(figsize=(15,12))
plt.subplot(3,3,1)
sns.boxplot(y='LotFrontage',data=df)
plt.subplot(3,3,2)
sns.boxplot(y='MasVnrArea',data=df)
plt.subplot(3,3,3)
sns.boxplot(y='GarageYrBlt',data=df)
plt.show()
# Impute null values with mean and median
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].mean())
# Impute null values in 'MasVnrType' with mode
df['MasVnrType'] = df['MasVnrType'].fillna('None')
df['BsmtQual'] = df['BsmtQual'].fillna('TA')
df['BsmtExposure'] = df['BsmtExposure'].fillna('NA')
df['BsmtFinType1'] = df['BsmtFinType1'].fillna('Unf')
df['GarageType'] = df['GarageType'].fillna('NA')
df['GarageFinish'] = df['GarageFinish'].fillna('Unf')

from datetime import date
present_year=date.today()
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(present_year.year)
# Check null value percentage
round(100*df.isnull().sum()/len(df.index),4)
# Datatype correction
df['GarageYrBlt']=df['GarageYrBlt'].astype('int64')
df.info()
# Combining rows in categorical columns to make the analysis more predictable

# Neighborhood
df['Neighborhood']=df['Neighborhood'].replace(['ClearCr','StoneBr','SWISU','Blmngtn',
                                               'MeadowV','BrDale','Veenker','NPkVill','Blueste'],'Others')

# HouseStyle
df['HouseStyle']=df['HouseStyle'].replace(['SLvl','SFoyer','1.5Unf','2.5Unf','2.5Fin'],'Others')

# OverallQual
df['OverallQual']=df['OverallQual'].replace([9,3,10,2,1],'Others')

# OverallCond
df['OverallCond']=df['OverallCond'].replace([9,2,1],'Others')

# RoofStyle
df['RoofStyle']=df['RoofStyle'].replace(['Flat','Gambrel','Mansard','Shed'],'Others')

# Exterior1st
df['Exterior1st']=df['Exterior1st'].replace(['WdShing','Stucco','AsbShng','Stone','BrkComm','AsphShn','CBlock','ImStucc'],'Others')

# Exterior2nd
df['Exterior2nd']=df['Exterior2nd'].replace(['Wd Shng','Stucco','BrkFace','AsbShng','ImStucc','Brk Cmn','Stone','AsphShn','CBlock ','Other'],'Others')

# ExterQual
df['ExterQual']=df['ExterQual'].replace(['Ex','Fa'],'Others')

# Foundation
df['Foundation']=df['Foundation'].replace(['Slab','Stone','Wood'],'Others')

# HeatingQC
df['HeatingQC']=df['HeatingQC'].replace(['Fa','Po'],'Others')

# KitchenQual
df['KitchenQual']=df['KitchenQual'].replace(['Ex','Fa'],'Others')

# GarageType
df['GarageType']=df['GarageType'].replace(['Basment','CarPort','2Types'],'Others')

# BedroomAbvGr
df['BedroomAbvGr']=df['BedroomAbvGr'].replace([1,5,6,0,8],'Others')

# TotRmsAbvGrd
df['TotRmsAbvGrd']=df['TotRmsAbvGrd'].replace([10,11,3,12,14,2],'Others')

# Fireplaces
df['Fireplaces']=df['Fireplaces'].replace([2,3],'Others')

# GarageCars
df['GarageCars']=df['GarageCars'].replace([0,4],'Others')
sns.distplot(df['SalePrice'], hist=True)
plt.xticks(rotation = 20)
plt.show()
print('Skewness before transformation : %f'%df['SalePrice'].skew())
# Log Transformation to remove skewness in traget variable
df['SalePrice'] = np.log(df['SalePrice'])
print('Skewness after transformation : %f'%df['SalePrice'].skew())
sns.distplot(df['SalePrice'], hist=True)
plt.show()
# create dummy variable for MSSubClass

MSSubClass = pd.get_dummies(df.MSSubClass,drop_first=True)
MSSubClass.rename(columns = {30:'MSSubClass30',40:'MSSubClass40',45:'MSSubClass45',
                             50:'MSSubClass50',60:'MSSubClass60',70:'MSSubClass70',
                             75:'MSSubClass75',80:'MSSubClass80',85:'MSSubClass85',
                             90:'MSSubClass90',120:'MSSubClass120',160:'MSSubClass160'
                            ,180:'MSSubClass180',190:'MSSubClass190'}, inplace=True)
MSSubClass.head()
# create dummy variable for MSZoning
MSZoning = pd.get_dummies(df.MSZoning,drop_first=True)
MSZoning.rename(columns = {'FV':'MSZoningFV','RH':'MSZoningRH','RL':'MSZoningRL',
                           'RM':'MSZoningRM'}, inplace=True)
MSZoning.head()
# create dummy variable for LotShape
LotShape = pd.get_dummies(df.LotShape,drop_first=True)
LotShape.rename(columns={'IR2':'LotShapeIR2','IR3':'LotShapeIR3','Reg':'LotShapeReg'},inplace=True)
LotShape.head()
# create dummy variable for LotConfig
LotConfig = pd.get_dummies(df.LotConfig,drop_first=True)
LotConfig.rename(columns={'CulDSac':'LotConfigCulDSac','FR2':'LotConfigFR2','FR3':'LotConfigFR3',
                         'Inside':'LotConfigInside'},inplace=True)
LotConfig.head()
# create dummy variable for Neighborhood
Neighborhood = pd.get_dummies(df.Neighborhood,drop_first=True)
Neighborhood.rename(columns={'NAmes':'NeighborhoodNames','Others':'NeighborhoodOthers',
                            'CollgCr':'NeighborhoodCollgCr','OldTown':'NeighborhoodOldTown',
                             'Edwards':'NeighborhoodEdwards','Somerst':'NeighborhoodSomerst',
                             'Gilbert':'NeighborhoodGilbert','NridgHt':'NeighborhoodNridgHt',
                             'Sawyer':'NeighborhoodSawyer','NWAmes':'NeighborhoodNWAmes',
                             'SawyerW':'NeighborhoodSawyerW','BrkSide':'NeighborhoodBrkSide',
                             'Crawfor':'NeighborhoodCrawfor','Mitchel':'NeighborhoodMitchel',
                             'NoRidge':'NeighborhoodNoRidge','Timber':'NeighborhoodTimber',
                             'IDOTRR':'NeighborhoodIDOTRR'}, inplace=True)
Neighborhood.head()
# create dummy variable for HouseStyle
HouseStyle=pd.get_dummies(df.HouseStyle,drop_first=True)
HouseStyle.rename(columns={'1Story':'HouseStyle_1Story','2Story':'HouseStyle_2Story',
                           'Others':'HouseStyle_Others'},inplace=True)
HouseStyle.head()
# create dummies for OverallQual
OverallQual=pd.get_dummies(df.OverallQual,drop_first=True)
OverallQual.rename(columns={5:'OverallQual_5',6:'OverallQual_6',7:'OverallQual_7'
                           ,8:'OverallQual_8','Others':'OverallQual_Others'}, inplace=True)
OverallQual.head()
# create dummies for OverallCond
OverallCond=pd.get_dummies(df.OverallCond,drop_first=True)
OverallCond.rename(columns={4:'OverallCond_4',5:'OverallCond_5',6:'OverallCond_6',7:'OverallCond_7',
                           8:'OverallCond_8','Others':'OverallCond_Others'},inplace=True)
OverallCond.head()
# create dummies for RoofStyle
RoofStyle=pd.get_dummies(df.RoofStyle,drop_first=True)
RoofStyle.rename(columns={'Hip':'RoofStyleHip','Others':'RoofStyleOthers'},inplace=True)
RoofStyle.head()
# create dummies for Exterior1st
Exterior1st=pd.get_dummies(df.Exterior1st,drop_first=True)
Exterior1st.rename(columns={'CemntBd':'Exterior1stCemntBd', 'HdBoard': 'Exterior1stHdBoard', 
                            'MetalSd': 'Exterior1stMetalSd', 'Others': 'Exterior1stOthers', 
                            'Plywood': 'Exterior1stPlywood', 'VinylSd': 'Exterior1stVinylSd',
                            'Wd Sdng':'Exterior1stWd Sdng'},inplace=True)
Exterior1st.head()
# create dummies for Exterior2nd
Exterior2nd=pd.get_dummies(df.Exterior2nd,drop_first=True)
Exterior2nd.rename(columns={'CmentBd':'Exterior2ndCmentBd', 'HdBoard': 'Exterior2ndHdBoard',
                            'MetalSd': 'Exterior2ndMetalSd', 'Others': 'Exterior2ndOthers',
                            'Plywood': 'Exterior2ndPlywood', 'VinylSd': 'Exterior2ndVinylSd',
                            'Wd Sdng':'Exterior2ndWd Sdng'},inplace=True)
Exterior2nd.head()
# create dummies for MasVnrType
MasVnrType=pd.get_dummies(df.MasVnrType,drop_first=True)
MasVnrType.rename(columns={'BrkFace':'MasVnrTypeBrkFace','None':'MasVnrTypeNone',
                           'Stone':'MasVnrTypeStone'},inplace=True)
MasVnrType.head()

# create dummies for ExterQual
ExterQual=pd.get_dummies(df.ExterQual,drop_first=True)
ExterQual.rename(columns={'Others':'ExterQualOthers','TA':'ExterQualTA'},inplace=True)
ExterQual.head()
# create dummies for Foundation
Foundation=pd.get_dummies(df.Foundation,drop_first=True)
Foundation.rename(columns={'CBlock':'FoundationCBlock','Others':'FoundationOthers',
                           'PConc':'FoundationPConc'},inplace=True)
Foundation.head()
# create dummies for BsmtQual
BsmtQual=pd.get_dummies(df.BsmtQual,drop_first=True)
BsmtQual.rename(columns={'Fa':'BsmtQualFa','Gd':'BsmtQualGd','TA':'BsmtQualTA'},inplace=True)
BsmtQual.head()
# create dummies for BsmtExposure
BsmtExposure=pd.get_dummies(df.BsmtExposure,drop_first=True)
BsmtExposure.rename(columns={'Gd':'BsmtExposureGd','Mn':'BsmtExposureMn','NA':'BsmtExposureNA',
                            'No':'BsmtExposureNo'},inplace=True)
BsmtExposure.head()
# create dummies for BsmtFinType1
BsmtFinType1 = pd.get_dummies(df.BsmtFinType1,drop_first=True)
BsmtFinType1.rename(columns={'BLQ':'BsmtFinType1BLQ','GLQ':'BsmtFinType1GLQ',
                             'LwQ':'BsmtFinType1LwQ','Rec':'BsmtFinType1Rec',
                            'Unf':'BsmtFinType1Unf'},inplace=True)
BsmtFinType1.head()
# create dummies for HeatingQC
HeatingQC=pd.get_dummies(df.HeatingQC,drop_first=True)
HeatingQC.rename(columns={'Gd':'HeatingQCGd','Others':'HeatingQCOthers','TA':'HeatingQCTA'},inplace=True)
HeatingQC.head()
# create dummies for BsmtFullBath
BsmtFullBath=pd.get_dummies(df.BsmtFullBath,drop_first=True)
BsmtFullBath.rename(columns={1:'BsmtFullBath_1',2:'BsmtFullBath_2',3:'BsmtFullBath_3'},inplace=True)
BsmtFullBath.head()
# create dummies for FullBath
FullBath=pd.get_dummies(df.FullBath,drop_first=True)
FullBath.rename(columns={1:'FullBath_1',2:'FullBath_2',3:'FullBath_3'},inplace=True)
FullBath.head()
# create dummies for HalfBath
HalfBath=pd.get_dummies(df.HalfBath,drop_first=True)
HalfBath.rename(columns={1:'HalfBath_1',2:'HalfBath_2'},inplace=True)
HalfBath.head()
# create dummies for BedroomAbvGr
BedroomAbvGr=pd.get_dummies(df.BedroomAbvGr,drop_first=True)
BedroomAbvGr.rename(columns={3:'BedroomAbvGr_3',4:'BedroomAbvGr_4',
                             'Others':'BedroomAbvGr_Others'},inplace=True)
BedroomAbvGr.head()
# create dummies for KitchenQual
KitchenQual=pd.get_dummies(df.KitchenQual,drop_first=True)
KitchenQual.rename(columns={'Others':'KitchenQualOthers','TA':'KitchenQualTA'},inplace=True)
KitchenQual.head()
# create dummies for TotRmsAbvGrd
TotRmsAbvGrd=pd.get_dummies(df.TotRmsAbvGrd,drop_first=True)
TotRmsAbvGrd.rename(columns={5:'TotRmsAbvGrd_5',6:'TotRmsAbvGrd_6',7:'TotRmsAbvGrd_7',
                            8:'TotRmsAbvGrd_8',9:'TotRmsAbvGrd_9','Others':'TotRmsAbvGrdOthers'},inplace=True)
TotRmsAbvGrd.head()
# create dummies for Fireplaces
Fireplaces=pd.get_dummies(df.Fireplaces,drop_first=True)
Fireplaces.rename(columns={1:'Fireplaces_1','Others':'FireplacesOthers'},inplace=True)
Fireplaces.head()
# create dummies for GarageType
GarageType=pd.get_dummies(df.GarageType,drop_first=True)
GarageType.rename(columns={'BuiltIn':'GarageTypeBuiltIn','Detchd':'GarageTypeDetchd','NA':'GarageTypeNA',
                          'Others':'GarageTypeOthers'},inplace=True)
GarageType.head()
# create dummies for GarageFinish
GarageFinish=pd.get_dummies(df.GarageFinish,drop_first=True)
GarageFinish.rename(columns={'RFn':'GarageFinishRFn','Unf':'GarageFinishUnf'},inplace=True)
GarageFinish.head()
#create dummies for GarageCars
GarageCars=pd.get_dummies(df.GarageCars,drop_first=True)
GarageCars.rename(columns={2:'GarageCars_2',3:'GarageCars_3','Others':'GarageCarsOthers'},inplace=True)
GarageCars.head()
# Concat all dummy variables in the original dataframe
df = pd.concat([df,MSSubClass,MSZoning,LotShape,LotConfig,Neighborhood,HouseStyle,OverallQual,
                OverallCond,RoofStyle,Exterior1st,Exterior2nd,MasVnrType,ExterQual,Foundation,
                BsmtQual,BsmtExposure,BsmtFinType1,HeatingQC,BsmtFullBath,FullBath,HalfBath,
                BedroomAbvGr,KitchenQual,TotRmsAbvGrd,Fireplaces,GarageType,GarageFinish,
                GarageCars], axis = 1)
df.head()
df.shape
# Drop the columns whose dummy variables are created and added in the original dataframe
df.drop(['MSSubClass','MSZoning','LotShape','LotConfig','Neighborhood','HouseStyle','OverallQual',
         'OverallCond','RoofStyle','Exterior1st','Exterior2nd','MasVnrType','ExterQual',
         'Foundation','BsmtQual','BsmtExposure','BsmtFinType1','HeatingQC','BsmtFullBath',
         'FullBath','HalfBath','BedroomAbvGr','KitchenQual','TotRmsAbvGrd','Fireplaces',
         'GarageType','GarageFinish','GarageCars'], axis = 1, inplace = True)
df.head()
df.shape
# Subtract the year present in columns to present year (2020)
from datetime import date
present_year=date.today()
df['YearBuilt']=df['YearBuilt'].apply(lambda x:present_year.year-x)
df['YearRemodAdd']=df['YearRemodAdd'].apply(lambda x:present_year.year-x)
df['GarageYrBlt']=df['GarageYrBlt'].apply(lambda x:present_year.year-x)
df['YrSold']=df['YrSold'].apply(lambda x:present_year.year-x)
from sklearn.model_selection import train_test_split

np.random.seed(0)
df_train, df_test = train_test_split(df, train_size = 0.7, test_size = 0.3, 
                                               random_state = 100)
# min-max scaling - train data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
y_train = df.loc[:, 'SalePrice']
X_train = df.loc[:, df.columns != 'SalePrice']

# Apply scaler()
scaler.fit(X)

X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state = 1)
# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn import linear_model, metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
# Running RFE with the output number of the variable equal to 50
ridge=Ridge()
rfe = RFE(ridge, 50)
rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
# Top 50 features selected using RFE
col = X_train.columns[rfe.support_]
col
# Features which are not included in top 50
X_train.columns[~rfe.support_]
# Creating X_train dataframe with RFE selected variables
X_train_1 = X_train[col]
X_train_1.head()
X_train_1.shape
# Creating Test X_test dataframe with RFE selected variables
X_test_1 = X_test[col]
X_test_1.head()
X_test_1.shape
# list of alphas to tune
params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}


ridge = Ridge()

# cross validation
folds = 5
model_cv = GridSearchCV(estimator = ridge, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            
model_cv.fit(X_train_1, y_train) 
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results = cv_results[cv_results['param_alpha']<=200]
cv_results.head()
# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')
plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper right')
plt.show()
print(model_cv.best_params_)
print(model_cv.best_score_)
# model with optimal alpha
# Ridge regression
lm1 = Ridge(alpha=0.4)
lm1.fit(X_train_1, y_train)

# predict
y_train_pred = lm1.predict(X_train_1)
print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))
y_test_pred = lm1.predict(X_test_1)
print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))
print(metrics.mean_squared_error(y_test,y_test_pred))
# ridge model parameters
model_parameters = list(lm1.coef_)
model_parameters.insert(0, lm1.intercept_)
model_parameters = [round(x, 3) for x in model_parameters]
cols = X_train_1.columns
cols = cols.insert(0, "constant")
sorted(list(zip( model_parameters,cols)),reverse=True)

# RFE
lasso=Lasso()
rfe_lasso = RFE(lasso,50)
rfe_lasso=rfe.fit(X_train,y_train)
list(zip(X_train.columns,rfe_lasso.support_,rfe_lasso.ranking_))
col_lasso = X_train.columns[rfe_lasso.support_]
col_lasso
# list of alpha to tune
params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}


# cross validation
model_lasso = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            

model_lasso.fit(X_train[col_lasso], y_train) 
cv_results = pd.DataFrame(model_lasso.cv_results_)
cv_results.head()
# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.xscale('log')
plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper right')
plt.show()
print(model_lasso.best_params_)
print(model_lasso.best_score_)
# model with optimal alpha
# lasso regression
lm = Lasso(alpha=0.0001)
lm.fit(X_train[col_lasso], y_train)

# predict
y_train_pred = lm.predict(X_train[col_lasso])
print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))
y_test_pred = lm.predict(X_test[col_lasso])
print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))
print(metrics.mean_squared_error(y_test,y_test_pred))
# lasso model parameters
model_parameters = list(lm.coef_)
model_parameters.insert(0, lm.intercept_)
model_parameters = [round(x, 3) for x in model_parameters]
cols = X_train[col_lasso].columns
cols = cols.insert(0, "constant")
sorted(list(zip(model_parameters,cols)),reverse=True)
























