#Import Libraries

import sys 

import pandas as pd 

import matplotlib 

import numpy as np 

import scipy as sc

import IPython

from IPython import display 

import sklearn



#ignore warnings

import warnings

warnings.filterwarnings('ignore')

print('-'*25)



#Algorithms

from sklearn.ensemble import   GradientBoostingRegressor

import xgboost as xgb

import lightgbm as lgb





from scipy.stats import skew

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.tools.plotting import scatter_matrix





%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8





#Import Data

data_train = pd.read_csv('../input/train.csv')

data_test  = pd.read_csv('../input/test.csv')



data_submit = pd.read_csv('../input/sample_submission.csv')



data_train.head()

data_test.head()



#preview data

print (data_train.info())

print (data_test.info()) 

#Plot Graphs and analyze data

#Sns.pairplot(data_train) #Commented as this will take long time depending upon data and computing power 



#Function to plot Graph of each attribute vs target value

def PlotAttributeVsTarget(df,target):

    for y in df.columns:

        if y!= target:

            plt.figure(figsize=(12,5))

            sns.scatterplot(x=data_train[y],y=data_train[target])

    

#PlotAttributeVsTarget(data_train,'SalePrice') #Commented as this will take long time depending upon data and computing power
#Function to plot Graph of each attribute(Categorical) vs target value

def PlotAttributeCategoricalVsTarget(df,target):

    for y in df.columns:

        if (y!= target and df[y].dtypes != 'int64' and df[y].dtypes != 'float64'):

            plt.figure(figsize=(12,5))

            sns.boxplot(x=data_train[y],y=data_train[target])

    

#PlotAttributeCategoricalVsTarget(data_train,'SalePrice') #Commented as this will take long time depending upon data and computing powe
#Outliers



#After analysing all the graph  (Uncomment the PlotAttributeVsTarget and PlotAttributeCategoricalVsTarget)

#carefully we find some outliers in TotalBsmtSF, GrLivArea

#We gonna delete these values

print('Shape Of old df:',data_train.shape)



plt.figure(figsize=(6,4))

sns.scatterplot(x=data_train['TotalBsmtSF'],y=data_train['SalePrice'])

data_train.drop(data_train[data_train['TotalBsmtSF'] > 5000].index, inplace = True)

plt.figure(figsize=(6,4))

sns.scatterplot(x=data_train['TotalBsmtSF'],y=data_train['SalePrice'])

print('Shape Of new df:',data_train.shape)





plt.figure(figsize=(6,4))

sns.scatterplot(x=data_train['GrLivArea'],y=data_train['SalePrice'])

data_train.drop(data_train[(data_train['GrLivArea'] > 4000) & (data_train['SalePrice'] < 300000)].index, inplace = True)

plt.figure(figsize=(6,4))

sns.scatterplot(x=data_train['GrLivArea'],y=data_train['SalePrice'])

print('Shape Of new df:',data_train.shape)

#Add dfs

data_train_test = pd.concat([data_train, data_test]) #Remeber the index because we have to split data later (1458 in train )



#Drop col with same value above certain percentage level

def getuniquevalues(df,percent):

    totalrow = df.shape[0]

    dic = {}

    for col in df.columns:

        count = df[col].value_counts().max()

        per = (count*100)/totalrow

        if per > percent:

            dic[col] = per

    return dic



Columns_Dic = getuniquevalues(data_train_test,99.8)

print('Number of col with same values: ',len(Columns_Dic))



#Col with values sorted

import operator

sorted_x = sorted(Columns_Dic.items(), key=operator.itemgetter(1))

sorted_x.reverse()

sorted_x
#Deleting columns

Columns_List= list(Columns_Dic.keys())

print('Shape Of old df:',data_train_test.shape)

data_train_test = data_train_test.drop(Columns_List,axis=1)

print('Shape Of new df:',data_train_test.shape)


#Dealing with null values

nulls = np.sum(data_train_test.isnull())

nullcols = nulls.loc[(nulls != 0)]

dtypes = data_train_test.dtypes

dtypes2 = dtypes.loc[(nulls != 0)]

info = pd.concat([nullcols, dtypes2], axis=1).sort_values(by=0, ascending=False)

print(info)

print("There are", len(nullcols), "columns with missing values")

print('Null in  data:',data_train_test.isnull().sum().sum())

print('Shape of df:', data_train_test.shape)

#Analysis of each varibale  and filling missing value if present

#We can take help from above graph to fill null values



#Also drop Id

'''

#Graph

sns.countplot(x=data_train['col_name'])

sns.distplot(data_train['col_name'])

sns.scatterplot(x=data_train['col_name'],y=data_train['SalePrice'])

sns.barplot(x = 'col_name', y = 'SalePrice', data=data_train)

sns.pointplot(x = 'col_name', y = 'SalePrice', data=data_train)

sns.boxplot(x="col_name", y="SalePrice", data=data_train)

sns.boxplot(x="col_name", y="SalePrice", hue="col_name2", data=data_train, palette="Set3")

#sns.pairplot(data_train)

'''



#Id

data_train_test.drop(['Id'], axis=1, inplace = True)
#PoolQC

data_train_test['PoolQC'].fillna('No Pool', inplace = True)



#MiscFeature

data_train_test['MSZoning'].fillna('None', inplace = True)



#Alley

data_train_test['Alley'].fillna('No alley', inplace = True)



#Fence

data_train_test['Fence'].fillna(-999, inplace = True)



#FireplaceQu

data_train_test['FireplaceQu'].fillna(-999, inplace = True)



#LotFrontage

data_train_test['LotFrontage'].fillna(data_train_test['LotFrontage'].median(), inplace = True)



#GarageCars

data_train_test['GarageCars'].fillna(data_train_test['GarageCars'].mode()[0],inplace=True)



#GarageArea

data_train_test['GarageArea'].fillna(data_train_test['GarageArea'].median(),inplace=True)



#Garage

data_train_test['GarageType'].fillna(-999, inplace = True) 

data_train_test['GarageYrBlt'].fillna(-999, inplace = True)

data_train_test['GarageFinish'].fillna(-999, inplace = True) 

data_train_test['GarageQual'].fillna(-999, inplace = True) 

data_train_test['GarageCond'].fillna(-999, inplace = True) 



#BsmtCond

data_train_test['BsmtCond'].fillna(data_train_test['BsmtCond'].mode()[0], inplace = True)



#BsmtExposure

data_train_test['BsmtExposure'].fillna(data_train_test['BsmtExposure'].mode()[0], inplace = True)



#BsmtQual

data_train_test['BsmtQual'].fillna(data_train_test['BsmtQual'].mode()[0], inplace = True)



#BsmtFinType2

data_train_test['BsmtFinType2'].fillna(data_train_test['BsmtFinType2'].mode()[0], inplace = True)



#BsmtFinType1

data_train_test['BsmtFinType1'].fillna(data_train_test['BsmtFinType1'].mode()[0], inplace = True)



#BsmtFinSF1

data_train_test['BsmtFinSF1'].fillna(data_train_test['BsmtFinSF1'].median(), inplace = True)



#BsmtFinSF2

data_train_test['BsmtFinSF2'].fillna(data_train_test['BsmtFinSF2'].median(), inplace = True)



#BsmtUnfSF

data_train_test['BsmtUnfSF'].fillna(data_train_test['BsmtUnfSF'].mode()[0], inplace = True)



#TotalBsmtSF

data_train_test['TotalBsmtSF'].fillna(data_train_test['TotalBsmtSF'].median(), inplace = True)



#MasVnrType

data_train_test['MasVnrType'].fillna(data_train_test['MasVnrType'].value_counts().index[0], inplace = True)



#MasVnrArea

data_train_test['MasVnrArea'].fillna(data_train_test['MasVnrArea'].median(), inplace = True)



#MSZoning

data_train_test['MSZoning'].fillna(data_train_test['MSZoning'].mode()[0], inplace = True)



#BsmtFullBath

data_train_test['BsmtFullBath'].fillna(data_train_test['BsmtFullBath'].mode()[0], inplace = True)



#BsmtHalfBath

data_train_test['BsmtHalfBath'].fillna(data_train_test['BsmtHalfBath'].mode()[0], inplace = True)



#Functional

data_train_test['Functional'].fillna(data_train_test['Functional'].mode()[0], inplace = True)





#Single missing value



#Street

data_train_test['Street'].fillna(data_train_test['Street'].mode()[0], inplace = True)



#Exterior1st

data_train_test['Exterior1st'].fillna('VinylSd', inplace = True)



#Exterior2nd

data_train_test['Exterior2nd'].fillna('VinylSd', inplace = True)



#Electrical

data_train_test['Electrical'].fillna(data_train_test['Electrical'].mode()[0], inplace = True)



#KitchenQual

data_train_test['KitchenQual'].fillna(data_train_test['KitchenQual'].mode()[0], inplace = True)



#SaleType

data_train_test['SaleType'].fillna(data_test['SaleType'].mode()[0],inplace=True)
#SalePrice

sns.distplot(data_train['SalePrice'])

data_train['SalePrice'].mean()

data_train['SalePrice'].median()
#Wrong values in dataset



data_train_test.describe()



data_train_test[data_train_test['GarageYrBlt'] == 2207]

data_train_test.loc[1132, 'GarageYrBlt'] = 2007
#Skewness in dataset



numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics2 = []

for i in data_train_test.columns:

    if data_train_test[i].dtype in numeric_dtypes: 

        numerics2.append(i)



skew_features = data_train_test[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

skews = pd.DataFrame({'skew':skew_features})

skews
high_skew = skew_features[skew_features > 0.5]

high_skew = high_skew

skew_index = high_skew.index



for i in skew_index:

    data_train_test[i]= boxcox1p(data_train_test[i], boxcox_normmax(data_train_test[i]+1))



        

skew_features2 = data_train_test[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

skews2 = pd.DataFrame({'skew':skew_features2})

skews2
#Get dummies

print(data_train_test.shape)

data_train_test = pd.get_dummies(data_train_test,drop_first = True).reset_index(drop=True)

print(data_train_test.shape)

#Split Data        

df_train = data_train_test.iloc[:1458,:] #if you remeber we had 1458 rows in train dataframe

df_test = data_train_test.iloc[1458:,:]          

print('Shape of Train df:', df_train.shape,'\nShape of Test df:', df_test.shape, '\nShape of train_test df:', data_train_test.shape)

#Assign X and Y

X_train = df_train.iloc[:,df_train.columns != 'SalePrice']

y_train = df_train.iloc[:,df_test.columns == 'SalePrice']

X_test = df_test.iloc[:,df_test.columns != 'SalePrice']
#Gradient Boosting Regression 

model_GB = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

#XGBoost 

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)

#LightGBM :

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
"""

#Grid Search

parameters = {

    'colsample_bytree':[0.3,0.5,0.6],

    'gamma':[0],

    'min_child_weight':[.5,1.5,2.5],

    'learning_rate':[0.1,0.2,0.3],

    'max_depth':[4,5,6],

    'n_estimators':[1000,10000,100000],

    'reg_alpha':[.1,.01,.001],

    'reg_lambda':[.1,.01,.001],

    'subsample':[0.5,0.6,0.7]  

}



grid_search = GridSearchCV(estimator = model, param_grid = parameters, n_jobs=-1,iid=False, verbose=10,scoring='neg_mean_squared_error')

grid_search.fit(train_x,train_y)

print('best params')

print (grid_search.best_params_)

print('best score')

print (grid_search.best_score_)"""



model_GB.fit(X_train,y_train.values.ravel())

model_xgb.fit(X_train,y_train.values.ravel())

model_lgb.fit(X_train,y_train.values.ravel())
y_pred_model_GB = model_GB.predict(X_test)

y_pred_model_xgb = model_xgb.predict(X_test)

y_pred_model_lgb = model_lgb.predict(X_test)



y_pred=(y_pred_model_GB+y_pred_model_xgb+y_pred_model_lgb)/3



data_submit['SalePrice'] = y_pred

data_submit.to_csv("submit.csv", index=False)