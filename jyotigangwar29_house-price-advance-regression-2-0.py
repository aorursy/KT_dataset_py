!pip install feature_engine
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



import scipy.stats as stats

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline



from feature_engine import variable_transformers as vt

from feature_engine import missing_data_imputers as mdi

from feature_engine.categorical_encoders import RareLabelCategoricalEncoder

from feature_engine.categorical_encoders import OneHotCategoricalEncoder



from sklearn.preprocessing import StandardScaler



from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from xgboost.sklearn import XGBRegressor

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score, KFold



import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
#importing data and exploring it

path='../input/house-prices-advanced-regression-techniques/'

train_data=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_data=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

print('train data shape:',train_data.shape)

print(train_data.info())

train_data.head()

train_data.columns
train_data['MSSubClass'][train_data['MSSubClass'] == 20] = '1-STORY 1946 & NEWER ALL STYLES'

train_data['MSSubClass'][train_data['MSSubClass'] == 30] = '1-STORY 1945 & OLDER'

train_data['MSSubClass'][train_data['MSSubClass'] == 40] = '1-STORY W/FINISHED ATTIC ALL AGES'

train_data['MSSubClass'][train_data['MSSubClass'] == 45] = '1-1/2 STORY - UNFINISHED ALL AGES'

train_data['MSSubClass'][train_data['MSSubClass'] == 50] = '1-1/2 STORY FINISHED ALL AGES'

train_data['MSSubClass'][train_data['MSSubClass'] == 60] = '2-STORY 1946 & NEWER'

train_data['MSSubClass'][train_data['MSSubClass'] == 70] = '2-STORY 1945 & OLDER'

train_data['MSSubClass'][train_data['MSSubClass'] == 75] = '2-1/2 STORY ALL AGES'

train_data['MSSubClass'][train_data['MSSubClass'] == 80] = 'SPLIT OR MULTI-LEVEL'

train_data['MSSubClass'][train_data['MSSubClass'] == 85] = 'SPLIT FOYER'

train_data['MSSubClass'][train_data['MSSubClass'] == 90] = 'DUPLEX - ALL STYLES AND AGES'

train_data['MSSubClass'][train_data['MSSubClass'] == 120] = '1-STORY PUD (Planned Unit Development) - 1946 & NEWER'

train_data['MSSubClass'][train_data['MSSubClass'] == 150] = '1-1/2 STORY PUD - ALL AGES'

train_data['MSSubClass'][train_data['MSSubClass'] == 160] = '2-STORY PUD - 1946 & NEWER'

train_data['MSSubClass'][train_data['MSSubClass'] == 180] = 'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER'

train_data['MSSubClass'][train_data['MSSubClass'] == 190] = '2 FAMILY CONVERSION - ALL STYLES AND AGES'
#Find the columns having missing values

train_data.columns[train_data.isnull().sum()>0]

#Find percentage of missing values in each column

train_data.isnull().mean().sort_values(ascending=False)
train_data['MasVnrArea'][(train_data.MasVnrType.isnull()==True) & (train_data.MasVnrArea.isnull()==True)] =0

train_data['GarageYrBlt'][(train_data.GarageType.isnull()==True) & (train_data.GarageYrBlt.isnull()==True)] =0



ordinal_variable=['ExterQual','ExterCond','BsmtCond','BsmtQual','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC','BsmtExposure','BsmtFinType1','BsmtFinType2']

for i in ordinal_variable:

    train_data[i] = train_data[i].fillna('None')
#Function to convert quality variables into ordinal values

def fixing_ordinal_variables(data, variable):

    data[variable][data[variable] == 'Ex'] = 5

    data[variable][data[variable] == 'Gd'] = 4

    data[variable][data[variable] == 'TA'] = 3

    data[variable][data[variable] == 'Fa'] = 2

    data[variable][data[variable] == 'Po'] = 1

    data[variable][data[variable] == 'None'] = 0
fixing_ordinal_variables(train_data,'ExterQual')

fixing_ordinal_variables(train_data,'ExterCond')

fixing_ordinal_variables(train_data,'BsmtCond')

fixing_ordinal_variables(train_data,'BsmtQual')

fixing_ordinal_variables(train_data,'HeatingQC')

fixing_ordinal_variables(train_data,'KitchenQual')

fixing_ordinal_variables(train_data,'FireplaceQu')

fixing_ordinal_variables(train_data,'GarageQual')

fixing_ordinal_variables(train_data,'GarageCond')

fixing_ordinal_variables(train_data,'PoolQC')
train_data['BsmtExposure'][train_data['BsmtExposure'] == 'Gd'] = 4

train_data['BsmtExposure'][train_data['BsmtExposure'] == 'Av'] = 3

train_data['BsmtExposure'][train_data['BsmtExposure'] == 'Mn'] = 2

train_data['BsmtExposure'][train_data['BsmtExposure'] == 'No'] = 1

train_data['BsmtExposure'][train_data['BsmtExposure'] == 'None'] = 0
train_data['BsmtFinType1'][train_data['BsmtFinType1'] == 'GLQ'] = 6

train_data['BsmtFinType1'][train_data['BsmtFinType1'] == 'ALQ'] = 5

train_data['BsmtFinType1'][train_data['BsmtFinType1'] == 'BLQ'] = 4

train_data['BsmtFinType1'][train_data['BsmtFinType1'] == 'Rec'] = 3

train_data['BsmtFinType1'][train_data['BsmtFinType1'] == 'LwQ'] = 2

train_data['BsmtFinType1'][train_data['BsmtFinType1'] == 'Unf'] = 1

train_data['BsmtFinType1'][train_data['BsmtFinType1'] == 'None'] =0 
train_data['BsmtFinType2'][train_data['BsmtFinType2'] == 'GLQ'] = 6

train_data['BsmtFinType2'][train_data['BsmtFinType2'] == 'ALQ'] = 5

train_data['BsmtFinType2'][train_data['BsmtFinType2'] == 'BLQ'] = 4

train_data['BsmtFinType2'][train_data['BsmtFinType2'] == 'Rec'] = 3

train_data['BsmtFinType2'][train_data['BsmtFinType2'] == 'LwQ'] = 2

train_data['BsmtFinType2'][train_data['BsmtFinType2'] == 'Unf'] = 1

train_data['BsmtFinType2'][train_data['BsmtFinType2'] == 'None'] =0 
train_data.isnull().mean().sort_values(ascending=False)
train_data['house_age']=train_data['YrSold']-train_data['YearBuilt']

train_data['remod_age']=train_data['YrSold']-train_data['YearRemodAdd']

train_data['garage_age']=train_data['YrSold']-train_data['GarageYrBlt']

train_data['garage_age'] = train_data['garage_age'].map(lambda x: 0 if x > 1000 else x)

train_data['house_age'] = train_data['house_age'].map(lambda x: 0 if x < 0 else x)

train_data['remod_age'] = train_data['remod_age'].map(lambda x: 0 if x < 0 else x)

train_data['garage_age'] = train_data['garage_age'].map(lambda x: 0 if x < 0 else x)

train_data.drop(['YrSold','MoSold','YearBuilt','YearRemodAdd','GarageYrBlt'],axis=1,inplace=True)

train_data['garage_age'].hist(bins=30, figsize=(5,5))

plt.show()
train_data["TransformedPrice"] = np.log(train_data["SalePrice"])
train_data.columns
#Splitting train_data into training set and validation set

X_train, X_val, y_train, y_val = train_test_split(

    train_data[['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',

       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',

       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',

       'HouseStyle', 'OverallQual', 'OverallCond','RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',

       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',

       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',

       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',

       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',

       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',

       'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',

       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',

       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',

       'Fence', 'MiscFeature', 'MiscVal','SaleType',

       'SaleCondition','house_age','remod_age','garage_age']],  # predictors

    train_data['TransformedPrice'],  # target

    test_size=0.20,  # percentage of obs in test set

    random_state=0)  # seed to ensure reproducibility



X_train.shape, X_val.shape
X_train['LotFrontage'].hist(bins=30, figsize=(5,5))

plt.show()
median_train = X_train['LotFrontage'].median()

mean_train = X_train['LotFrontage'].mean()

print(median_train)

print(mean_train)
X_train.shape
pipe_1 = Pipeline([

    ('imputer_mode', mdi.FrequentCategoryImputer(variables=['Electrical'])),

    ('imputer_missing', mdi.CategoricalVariableImputer(variables=['Alley', 

       'GarageFinish', 'Fence','MiscFeature','MasVnrType','GarageType'])),

    ('imputer_median',mdi.MeanMedianImputer(imputation_method='median',variables=['LotFrontage']))

])
pipe_1.fit(X_train)

X_train = pipe_1.transform(X_train)

X_val = pipe_1.transform(X_val)



# let's check null values are gone

X_train.isnull().mean().sort_values(ascending=False)
#Get unique category of each categorical variable to see if there is Cardinality

cat_var=list(X_train.select_dtypes(include=['object']).columns)

for i in list(X_train.select_dtypes(include=['object']).columns):

    print(i)

    print('X_train')

    print (X_train[i].unique())

    print('X_val')

    print (X_val[i].unique())

    print('---------')

cat_var
for i in cat_var:

    plt.figure(figsize=(16, 4))



    plt.subplot(1, 2, 1)

    X_train[i].value_counts().plot(kind='bar')

    print(i)

    plt.title('Training data Bar Chart ')



    plt.subplot(1, 2, 2)

    X_val[i].value_counts().plot(kind='bar')

    plt.title('validation data Bar Chart:')



    plt.show()
for col in X_train.columns:

    

    if X_train[col].dtypes == 'O': 

        

        if X_train[col].nunique() < 3: 

            

            print(X_train.groupby(col)[col].count() / len(X_train))

            print()
for col in X_train.columns:

    if X_train[col].dtypes == 'O':

        print(X_train.groupby(col)[col].count() / len(X_train))

        print()
pipe_2 = Pipeline([

    ('Rare Label .03',RareLabelCategoricalEncoder(

    tol=0.03,n_categories=3,variables=[ 'MSZoning','Street',

       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',

       'LandSlope', 'Condition1', 'Condition2', 'BldgType',

       'HouseStyle','RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

        'Foundation', 'Heating','CentralAir', 'Electrical',

       'Functional',  'GarageType','GarageFinish', 'PavedDrive',

       'Fence', 'MiscFeature', 'SaleType',

       'SaleCondition'])

    ),

    ('Rare Label .05',RareLabelCategoricalEncoder(

    tol=0.05,n_categories=3,variables=[ 'Neighborhood','MSSubClass'])

    ),

])
pipe_2.fit(X_train)

X_train = pipe_2.transform(X_train)

X_val = pipe_2.transform(X_val)
for col in X_train.columns:

    if X_train[col].dtypes == 'O':

        print(X_train.groupby(col)[col].count() / len(X_train))

        print()
X_train.drop(['Street',

       'Alley', 'Utilities','LandSlope','Condition2','RoofMatl', 'Heating','MiscFeature','Id'], axis=1,inplace=True)

X_val.drop(['Street',

       'Alley', 'Utilities','LandSlope','Condition2','RoofMatl', 'Heating','MiscFeature','Id'], axis=1,inplace=True)
# function to create histogram and boxplot





def diagnostic_plots(df, variable):

   

    plt.figure(figsize=(16, 4))



    # histogram

    plt.subplot(1, 2, 1)

    sns.distplot(df[variable], bins=30)

    plt.title('Histogram')



    # boxplot

    plt.subplot(1, 2, 2)

    sns.boxplot(y=df[variable])

    plt.title('Boxplot')



    plt.show()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

for i in X_train.select_dtypes(include=numerics).columns:

    if i not in ordinal_variable:

        diagnostic_plots(X_train, i)
#function to calculate percentage of instances having value 0

def percentage_zero(df,variable):

    count=0

    for i in df.index:

        if df[variable][i] == 0:

            count+=1

    percentage=count/len(df)

    return percentage
#Calculating percentage of zero values

null_columns=[]

for i in X_train.select_dtypes(include=numerics).columns:

    if i not in ordinal_variable:

        print(i)

        print(percentage_zero(X_train,i))

        print('-----------')

#dropping fetures with very high 0 values

        if percentage_zero(X_train,i)>0.80:

            null_columns.append(i)

X_train.drop(null_columns,inplace=True,axis=1)

X_val.drop(null_columns,inplace=True,axis=1)

print(X_train.shape)

print(X_val.shape)

null_columns
#function for outlier handling by capping

def find_boundaries(df, variable):



    # the boundaries are the quantiles



    lower_boundary = df[variable].quantile(0.05)

    upper_boundary = df[variable].quantile(0.95)



    return upper_boundary, lower_boundary
#outliers handling

vars_with_outliers=[['LotFrontage', 'LotArea', 'MasVnrArea',

       'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',

       'GrLivArea','WoodDeckSF', 'OpenPorchSF', 'house_age', 'remod_age',

       'garage_age']]

for i in vars_with_outliers:

    upper_limit, lower_limit = find_boundaries(X_train, i)

    X_train[i]= np.where(X_train[i] > upper_limit, upper_limit,

                       np.where(X_train[i]< lower_limit,lower_limit, X_train[i]))

    X_val[i]= np.where(X_val[i] > upper_limit, upper_limit,

                       np.where(X_val[i]< lower_limit,lower_limit, X_val[i]))

data=X_train.join(y_train)

fig=plt.figure(figsize=(25,10))

sns.heatmap(data.corr(),annot=True)
low_corr_features = data.corr().index[abs(data.corr()['TransformedPrice'])<0.25]

low_corr_features
data.drop(['house_age', 'remod_age', 'garage_age','TotRmsAbvGrd','GarageCars', '1stFlrSF'],inplace=True,axis=1)

X_train.drop(['house_age', 'remod_age', 'garage_age','TotRmsAbvGrd','GarageCars', '1stFlrSF'],inplace=True,axis=1)

X_val.drop(['house_age', 'remod_age', 'garage_age','TotRmsAbvGrd','GarageCars', '1stFlrSF'],inplace=True,axis=1)

data.drop(low_corr_features,inplace=True,axis=1)

X_train.drop(low_corr_features,inplace=True,axis=1)

X_val.drop(low_corr_features,inplace=True,axis=1)

data.shape
X_train.columns
ohe_enc = OneHotCategoricalEncoder(

    top_categories=None,

    drop_last=True)



ohe_enc.fit(X_train)

X_train=ohe_enc.transform(X_train)

X_val=ohe_enc.transform(X_val)

print(X_train.shape)

print(X_val.shape)
X_train.head()
def grid_search(instance,parameters,X_training,y_training,X_validation,y_validation):

    grid_reg = GridSearchCV(instance, parameters, verbose=1 , scoring = "r2",cv=10)

    grid_reg.fit(X_training, y_training)

   

    print("Best Reg Model: " + str(grid_reg.best_estimator_))

    print("Best Score: " + str(grid_reg.best_score_))

    

    reg = grid_reg.best_estimator_

    reg.fit(X_training, y_training)

    y_pred_train = reg.predict(X_training)

    y_pred_val = reg.predict(X_validation)

    return r2_score(y_training, y_pred_train),np.sqrt(mean_squared_error(y_training, y_pred_train)),r2_score(y_validation, y_pred_val),np.sqrt(mean_squared_error(y_validation, y_pred_val))



    
linreg = LinearRegression()

linreg.fit(X_train,y_train)

y_pred_train=linreg.predict(X_train)

y_pred_val=linreg.predict(X_val)

r2_lin_train=r2_score(y_train, y_pred_train)

rmse_lin_train=np.sqrt(mean_squared_error(y_train, y_pred_train))

r2_lin_val=r2_score(y_val, y_pred_val)

rmse_lin_val=np.sqrt(mean_squared_error(y_val, y_pred_val))

print("R^2 Score Train: " + str(r2_lin_train))

print("RMSE Score Train: " + str(rmse_lin_train))

print("R^2 Score Test: " + str(r2_lin_val))

print("RMSE Score Test: " + str(rmse_lin_val))
lasso = Lasso()

parameters_lasso = {"alpha" : [1,0.1,.01,0.001,0.0001],"random_state":[0]}

r2_lasso_train,rmse_lasso_train,r2_lasso_val,rmse_lasso_val=grid_search(lasso,parameters_lasso,X_train,y_train,X_val,y_val)

print("R^2 Score: " + str(r2_lasso_train))

print("RMSE Score: " + str(rmse_lasso_train))

print("R^2 Score: " + str(r2_lasso_val))

print("RMSE Score: " + str(rmse_lasso_val))
ridge = Ridge()

parameters_ridge = {"alpha" : [1,10,15,20,25,30],"random_state":[0], "solver" : ["auto"]}

r2_ridge_train,rmse_ridge_train,r2_ridge_val,rmse_ridge_val=grid_search(ridge,parameters_ridge,X_train,y_train,X_val,y_val)

print("R^2 Score: " + str(r2_ridge_train))

print("RMSE Score: " + str(rmse_ridge_train))

print("R^2 Score: " + str(r2_ridge_val))

print("RMSE Score: " + str(rmse_ridge_val))
rf = RandomForestRegressor()

parameters_rf = {"n_estimators" : [105,110,115], "criterion" : ["mse" , "mae"], "min_samples_split" : [2, 3, 5, 10], 

                 "max_features" : ["auto", "log2","sqrt"],"random_state":[0]}

r2_rf_train,rmse_rf_train,r2_rf_val,rmse_rf_val=grid_search(rf,parameters_rf,X_train,y_train,X_val,y_val)

print("R^2 Score: " + str(r2_rf_train))

print("RMSE Score: " + str(rmse_rf_train))

print("R^2 Score: " + str(r2_rf_val))

print("RMSE Score: " + str(rmse_rf_val))
model_performances = pd.DataFrame({

    "Model" : ["Linear Regression", "Ridge", "Lasso", "Random Forest Regressor"],

    "R Squared_train" : [str(r2_lin_train)[0:5], str(r2_ridge_train)[0:5], str(r2_lasso_train)[0:5], str(r2_rf_train)[0:5]],

    "RMSE_train" : [str(rmse_lin_train)[0:8], str(rmse_ridge_train)[0:8], str(rmse_lasso_train)[0:8], str(rmse_rf_train)[0:8]],

    "R Squared_val" : [str(r2_lin_val)[0:5], str(r2_ridge_val)[0:5], str(r2_lasso_val)[0:5], str(r2_rf_val)[0:5]],

    "RMSE_val" : [str(rmse_lin_val)[0:8], str(rmse_ridge_val)[0:8], str(rmse_lasso_val)[0:8], str(rmse_rf_val)[0:8]]

})

model_performances.round(4)
print("Sorted by R Squared:")

model_performances.sort_values(by="R Squared_val", ascending=False)
print("Sorted by RMSE:")

model_performances.sort_values(by="RMSE_val", ascending=True)
test_data['MasVnrArea'][(test_data.MasVnrType.isnull()==True) & (test_data.MasVnrArea.isnull()==True)] =0

test_data['GarageYrBlt'][(test_data.GarageType.isnull()==True) & (test_data.GarageYrBlt.isnull()==True)] =0
test_data['house_age']=test_data['YrSold']-test_data['YearBuilt']

test_data['remod_age']=test_data['YrSold']-test_data['YearRemodAdd']

test_data['garage_age']=test_data['YrSold']-test_data['GarageYrBlt']

test_data['garage_age'] = test_data['garage_age'].map(lambda x: 0 if x > 1000 else x)

test_data['house_age'] = test_data['house_age'].map(lambda x: 0 if x < 0 else x)

test_data['remod_age'] = test_data['remod_age'].map(lambda x: 0 if x < 0 else x)

test_data['garage_age'] = test_data['garage_age'].map(lambda x: 0 if x < 0 else x)

test_data.drop(['YrSold','MoSold','YearBuilt','YearRemodAdd','GarageYrBlt'],axis=1,inplace=True)
test_data['MSSubClass'][test_data['MSSubClass'] == 20] = '1-STORY 1946 & NEWER ALL STYLES'

test_data['MSSubClass'][test_data['MSSubClass'] == 30] = '1-STORY 1945 & OLDER'

test_data['MSSubClass'][test_data['MSSubClass'] == 40] = '1-STORY W/FINISHED ATTIC ALL AGES'

test_data['MSSubClass'][test_data['MSSubClass'] == 45] = '1-1/2 STORY - UNFINISHED ALL AGES'

test_data['MSSubClass'][test_data['MSSubClass'] == 50] = '1-1/2 STORY FINISHED ALL AGES'

test_data['MSSubClass'][test_data['MSSubClass'] == 60] = '2-STORY 1946 & NEWER'

test_data['MSSubClass'][test_data['MSSubClass'] == 70] = '2-STORY 1945 & OLDER'

test_data['MSSubClass'][test_data['MSSubClass'] == 75] = '2-1/2 STORY ALL AGES'

test_data['MSSubClass'][test_data['MSSubClass'] == 80] = 'SPLIT OR MULTI-LEVEL'

test_data['MSSubClass'][test_data['MSSubClass'] == 85] = 'SPLIT FOYER'

test_data['MSSubClass'][test_data['MSSubClass'] == 90] = 'DUPLEX - ALL STYLES AND AGES'

test_data['MSSubClass'][test_data['MSSubClass'] == 120] = '1-STORY PUD (Planned Unit Development) - 1946 & NEWER'

test_data['MSSubClass'][test_data['MSSubClass'] == 150] = '1-1/2 STORY PUD - ALL AGES'

test_data['MSSubClass'][test_data['MSSubClass'] == 160] = '2-STORY PUD - 1946 & NEWER'

test_data['MSSubClass'][test_data['MSSubClass'] == 180] = 'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER'

test_data['MSSubClass'][test_data['MSSubClass'] == 190] = '2 FAMILY CONVERSION - ALL STYLES AND AGES'
ordinal_variable=['ExterQual','ExterCond','BsmtCond','BsmtQual','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC','BsmtExposure','BsmtFinType1','BsmtFinType2']

for i in ordinal_variable:

    test_data[i] = test_data[i].fillna('None')
fixing_ordinal_variables(test_data,'ExterQual')

fixing_ordinal_variables(test_data,'ExterCond')

fixing_ordinal_variables(test_data,'BsmtCond')

fixing_ordinal_variables(test_data,'BsmtQual')

fixing_ordinal_variables(test_data,'HeatingQC')

fixing_ordinal_variables(test_data,'KitchenQual')

fixing_ordinal_variables(test_data,'FireplaceQu')

fixing_ordinal_variables(test_data,'GarageQual')

fixing_ordinal_variables(test_data,'GarageCond')

fixing_ordinal_variables(test_data,'PoolQC')
test_data['BsmtExposure'][test_data['BsmtExposure'] == 'Gd'] = 4

test_data['BsmtExposure'][test_data['BsmtExposure'] == 'Av'] = 3

test_data['BsmtExposure'][test_data['BsmtExposure'] == 'Mn'] = 2

test_data['BsmtExposure'][test_data['BsmtExposure'] == 'No'] = 1

test_data['BsmtExposure'][test_data['BsmtExposure'] == 'None'] = 0
test_data['BsmtFinType1'][test_data['BsmtFinType1'] == 'GLQ'] = 6

test_data['BsmtFinType1'][test_data['BsmtFinType1'] == 'ALQ'] = 5

test_data['BsmtFinType1'][test_data['BsmtFinType1'] == 'BLQ'] = 4

test_data['BsmtFinType1'][test_data['BsmtFinType1'] == 'Rec'] = 3

test_data['BsmtFinType1'][test_data['BsmtFinType1'] == 'LwQ'] = 2

test_data['BsmtFinType1'][test_data['BsmtFinType1'] == 'Unf'] = 1

test_data['BsmtFinType1'][test_data['BsmtFinType1'] == 'None'] =0 
test_data['BsmtFinType2'][test_data['BsmtFinType2'] == 'GLQ'] = 6

test_data['BsmtFinType2'][test_data['BsmtFinType2'] == 'ALQ'] = 5

test_data['BsmtFinType2'][test_data['BsmtFinType2'] == 'BLQ'] = 4

test_data['BsmtFinType2'][test_data['BsmtFinType2'] == 'Rec'] = 3

test_data['BsmtFinType2'][test_data['BsmtFinType2'] == 'LwQ'] = 2

test_data['BsmtFinType2'][test_data['BsmtFinType2'] == 'Unf'] = 1

test_data['BsmtFinType2'][test_data['BsmtFinType2'] == 'None'] =0 
test_data.shape
test_data=test_data[['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',

       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',

       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',

       'HouseStyle', 'OverallQual', 'OverallCond','RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',

       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',

       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',

       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',

       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',

       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',

       'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',

       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',

       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',

       'Fence', 'MiscFeature', 'MiscVal','SaleType',

       'SaleCondition','house_age','remod_age','garage_age']]
test_data_original=test_data

test_data_original.shape
test_data = pipe_1.transform(test_data)

test_data = pipe_2.transform(test_data)

test_data.drop(['Street',

               'Alley', 'Utilities','LandSlope','Condition2','RoofMatl', 'Heating','MiscFeature',

                'Id','house_age', 'remod_age', 'garage_age','TotRmsAbvGrd','GarageCars', '1stFlrSF'], axis=1,inplace=True)

test_data.drop(low_corr_features,inplace=True,axis=1)
test_data.drop(null_columns,inplace=True,axis=1)
test_data=ohe_enc.transform(test_data)
test_data.isnull().mean().sort_values(ascending=False)
imputer=mdi.MeanMedianImputer(imputation_method='median',variables=[ 'BsmtFinSF1','TotalBsmtSF', 'GarageArea'])

imputer.fit(X_train)

test_data=imputer.transform(test_data)
test_data.isnull().mean().sort_values(ascending=False)
rf = RandomForestRegressor(n_estimators=110, criterion='mse', min_samples_split=2, 

                 max_features='sqrt',random_state=0)



rf.fit(X_train, y_train)

submission_predictions = np.exp(rf.predict(test_data))
submission = pd.DataFrame({

        "Id": test_data_original["Id"],

        "SalePrice": submission_predictions

    })



submission.to_csv("prices.csv", index=False)

print(submission.shape)