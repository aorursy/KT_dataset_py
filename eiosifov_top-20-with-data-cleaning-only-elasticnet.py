%load_ext autoreload

%autoreload 2

import os



%matplotlib inline
import numpy as np

import pandas as pd

from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype

from scipy.stats import norm, skew



import math

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")



#from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC



from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

import category_encoders as ce

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import StratifiedKFold



from sklearn.model_selection import train_test_split



from sklearn.metrics import mean_squared_log_error

from scipy.special import boxcox1p

from scipy.stats import boxcox





import string

import warnings

warnings.filterwarnings('ignore')
!ls ../input
PATH = "../input/house-prices-advanced-regression-techniques/"
df_train=pd.read_csv(f'{PATH}train.csv')#, index_col='Id')

df_test=pd.read_csv(f'{PATH}test.csv')#, index_col='Id')
# for the purpose of evaluation of current competition we transform target value

df_train.SalePrice = np.log1p(df_train.SalePrice)
print('Number of Training Examples = {}'.format(df_train.shape[0]))

print('Number of Test Examples = {}\n'.format(df_test.shape[0]))

print('Training X Shape = {}'.format(df_train.shape))

print('Training y Shape = {}\n'.format(df_train['SalePrice'].shape[0]))

print('Test X Shape = {}'.format(df_test.shape))

print('Test y Shape = {}\n'.format(df_test.shape[0]))

#print(df_train.columns)

#print(df_test.columns)
#print(df_train.info())

#df_train.sample(3)

#print(df_test.info())

#df_test.sample(3)
fig, ax = plt.subplots()

ax.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
# Deleting outliers

df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)



#Check the graphic again

fig, ax = plt.subplots()

ax.scatter(df_train['GrLivArea'], df_train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
#remember where to divide train and test

ntrain = df_train.shape[0]

ntest = df_test.shape[0]



#Save the 'Id' column

train_ID = df_train['Id']

test_ID = df_test['Id']
def concat_df(train_data, test_data):

    # Returns a concatenated df of training and test set on axis 0

    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)



df_all = concat_df(df_train, df_test)



df_train.name = 'Training Set'

df_test.name = 'Test Set'

df_all.name = 'All Set' 



dfs = [df_train, df_test]



df_all.shape
#Dividing Target column (Y)

y_train_full = df_train.SalePrice.values

df_all.drop(['SalePrice'], axis=1, inplace=True)

df_all.drop('Id',axis=1,inplace=True)
y_train_full
def mark_missing (df):

    for col in df.columns:

        if df_all[col].isnull().sum()>0:

            df_all[col+'_missed']=df_all[col].isnull()
mark_missing(df_all)
df_all.shape
def display_missing(df):

    for col in df.columns:

        print(col, df[col].isnull().sum())

    print('\n')

    

for df in dfs:

    print(format(df.name))

    display_missing(df)

    

    

    

#Check remaining missing values if any 

def display_only_missing(df):

    all_data_na = (df.isnull().sum() / len(df)) * 100

    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)

    missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

    print(missing_data)
display_only_missing(df_all)
# fill NA values (not missed) with None - based on data description -  - for non-Numerical (object) Columns

for col in ('Alley','MasVnrType','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 

            'BsmtFinType2','FireplaceQu','GarageType', 'GarageFinish', 'GarageQual', 

            'GarageCond','PoolQC','Fence','MiscFeature'):

    df_all[col] = df_all[col].fillna('None')
display_only_missing(df_all)
#fill NA numerical value with '0' - based on data description of correspondent Object columns - for Numerical Columns

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath','MasVnrArea'):

    df_all[col] = df_all[col].fillna(0)
display_only_missing(df_all)
# Fill missing value in corresponding columns with most frequent value in column

for col in ('Utilities','Functional','SaleType','KitchenQual','Exterior2nd','Exterior1st','Electrical'):

    df_all[col].fillna(df_all[col].mode()[0], inplace=True)

    

# Functional : data description says NA means typical

# BTW we just used df_all.Functional.mode() = use most frequent value (as 'Typ' is most frequent value)

#df_all["Functional"] = df_all["Functional"].fillna("Typ")
display_only_missing(df_all)
df_all.MSZoning.isnull().sum()
df_all["MSZoning"] = df_all["MSZoning"].fillna("None")
display_only_missing(df_all)
#### Iteration 2 - replacing by machine learning
df_all['LotFrontage'].isnull().sum()
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

df_all["LotFrontage"] = df_all.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))
df_all['LotFrontage'].isnull().sum()
display_only_missing(df_all)
df_all.info()
# Function Splitting Train - Validation

def quick_get_dumm(df):

    X_train_full=df.iloc[:ntrain] # Full Train set



    # Creating train and validation sets

    X_train, X_valid, y_train, y_valid = train_test_split(pd.get_dummies(X_train_full), y_train_full, random_state=42)

    return X_train, X_valid, y_train, y_valid
X_train, X_valid, y_train, y_valid = quick_get_dumm(df_all)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
# Defining evaluation functions

def rmse(x,y): return math.sqrt(((x-y)**2).mean())



def print_score(m,X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid):

    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),

                m.score(X_train, y_train), m.score(X_valid, y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
m_rf = RandomForestRegressor(n_estimators=160, min_samples_leaf=1, max_features=0.5, n_jobs=-1, oob_score=True, random_state=42)

m_rf.fit(X_train, y_train)

print_score(m_rf)
def elastic_score(X,y):

    elastic = ElasticNet(random_state=1)

    param = {'l1_ratio' : [0],

             'alpha' : [0.017]}

    elastic = GridSearchCV(elastic, param, cv=5, scoring='neg_mean_squared_error')

    elastic.fit(X,y)

    print('Elastic:', np.sqrt(elastic.best_score_*-1))

    return elastic
elastic_score(X_train, y_train)
m_xgb = XGBRegressor(n_estimators=160, learning_rate=0.05, random_state=42)

# using early_stop to find out where validation scores don't improve

#m_xgb.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)], verbose=False)

m_xgb.fit(X_train, y_train)

print_score(m_xgb)
# We created function to return NA values of feature/column back in place, 

# based on _missed column, we created to state what values was missed in original dataset



# returning original NA values back

def return_original_na(df, feature):

    df[feature].loc[df.index[df[feature+'_missed'] == True].tolist()]=np.nan

    return df[feature]
#Returning original NA values of MSZoning back in place

df_all['LotFrontage']=return_original_na(df_all, 'LotFrontage')
df_all['LotFrontage'].isnull().sum()
display_only_missing(df_all)
def filling_na_with_predictions(df, feature):

    """

    df - DataFrame without target column y. Train+Test DataFrame (df_all)

    feature - feature (column), containing real NA values we will fill



    Assumption:

    All other columns do not have NA values. In case of having we have to impute with some Statistical method (Median, etc)

    We do not do it inside this function

    """



    flag_object=0

    

    if df[feature].isnull().sum()>0:

        ## Store Indexes of rows with NA values (we can just call "_missed" column with True values, to check those indexes as well)

        ## Creating index based on NA values present in column

        na_rows_idxs=df[df[feature].isnull()].index 

            ## Creating index based on NA values being present in original DF column

            #na_rows_idxs=df.index[df[feature+'_missed'] == True].tolist()



        ## For fitting and predictiong - convert DF to dummies DF, ready for ML

        #df=pd.get_dummies(df)

        ## If feature object we cant just dummy all, we shouldn't dummy feature column

        df=pd.concat([ pd.Series(df[feature]), pd.get_dummies(df.drop([feature], axis=1)) ], axis=1)





        ## Splitting DF to Feature_Train_X, Feature_Train_y, Feature_Predict_X:

        ## Feature_Train_X = DF without NA values in "feature_with_NA"column

        ## Feature_Train_y = target values that we have. All values in "feature_with_NA" except NA values

        ## Feature_Predict_X = DF of correcponding to NA values in "feature_with_NA" without target vales (basically because they is equal to NA)

        Feature_Train_X=df.drop(df[df[feature].isnull()].index).drop([feature], axis=1)

        Feature_Train_y=df[feature].drop(df[df[feature].isnull()].index).values

        Feature_Predict_X=df[df[feature].isnull()].drop([feature], axis=1)



        ## If feature is NOT Numerical

        ## Label encoding of y values in case it is not numerical

        if is_string_dtype(df[feature]) or is_categorical_dtype(df[feature]):

            flag_object=1

            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()

            le.fit(Feature_Train_y)

            Feature_Train_y=le.transform(Feature_Train_y)

             

        ## Making predictions, what might be in NA fields based on Train DF

        #m_xgb = XGBRegressor(n_estimators=160, learning_rate=0.05)

        #m_xgb.fit(Feature_Train_X, Feature_Train_y)

        elastic = ElasticNet(random_state=1)

        param = {'l1_ratio' : [0],

             'alpha' : [0.017]}

        elastic = GridSearchCV(elastic, param, cv=5, scoring='neg_mean_squared_error')

        elastic.fit(Feature_Train_X,Feature_Train_y)

    

        ## Creating (Predicting) values to impute NA

        #fillna_values=m_xgb.predict(Feature_Predict_X)

        fillna_values=elastic.predict(Feature_Predict_X)



        ## If feature is NOT Numerical

        ## Return Encoded values back to Object/Category if feature NOT numerical

        if flag_object==1:

            fillna_values=le.inverse_transform(np.around(fillna_values).astype(int))

        

        ## Replacing NA values with predicted Series of values

        df[feature]=df[feature].fillna(pd.Series(index=na_rows_idxs,data=fillna_values))



        ## Returning feature column without NA values    

        return df[feature]

    else:

        print ('There were no NA values')
# Datafremas to predict missed LotFrontage in train test. We add SalePrice to exploit all data that we have, hence to have more accuracy.

# As next step we will use concatenated dataframe Train+Test but without SalePrice, as we don't have SalePrice column for test, hence can't use this info to restore missed values of LotFrontage in Test dataset

df_tmp_train=df_all.iloc[:ntrain] # Full Train set

df_tmp_train['SalePrice']=y_train_full

df_tmp_test=df_all.iloc[ntrain:] # Test set
# Replacing missing LotFrontage values in Train dataset

df_tmp_train['LotFrontage']=filling_na_with_predictions(df_tmp_train, "LotFrontage")

# Replacing missing LotFrontage values in Test dataset

df_tmp_test['LotFrontage']=filling_na_with_predictions(df_tmp_test, "LotFrontage")
df_tmp_train.drop(['SalePrice'], axis=1, inplace=True)

df_all = concat_df(df_tmp_train, df_tmp_test)
df_all['LotFrontage'].isnull().sum()
def evaluate(df):

    # Split dataset for train-validation

    X_train, X_valid, y_train, y_valid = quick_get_dumm(df)

    

    #ElasticNet

    elastic_score(X_train, y_train)



    #XGBoost

    m_xgb.fit(X_train, y_train)

    print('XGBoost')

    print_score(m_xgb,X_train, X_valid, y_train, y_valid)



    # Random Forest

    m_rf.fit(X_train, y_train)

    print('Random Forest')

    print_score(m_rf,X_train, X_valid, y_train, y_valid)
evaluate(df_all)
#Returning original NA values of MSZoning back in place

df_all['MSZoning']=return_original_na(df_all, 'MSZoning')
display_only_missing(df_all)
df_all[df_all['MSZoning'].isnull()].index
df_all['MSZoning']=filling_na_with_predictions(df_all, 'MSZoning')
df_all['MSZoning'].loc[df_all.index[df_all['MSZoning'+'_missed'] == True].tolist()]
evaluate(df_all)
##### Dealing with Missing values we replaced with most common - now replacing them with ML predictions
for col in ('Utilities','Functional','SaleType','KitchenQual','Exterior2nd','Exterior1st','Electrical'):

    print ('Filling with most common:\n',df_all[col].loc[df_all.index[df_all[col+'_missed'] == True].tolist()])

    df_all[col]=return_original_na(df_all, col)

    df_all[col]=filling_na_with_predictions(df_all, col)

    print ('Filling with predictions:\n',df_all[col].loc[df_all.index[df_all[col+'_missed'] == True].tolist()])
evaluate(df_all)
# Saving Train Dataset after cleaning

df_train_save=df_all.iloc[:ntrain]

df_train_save['SalePrice']=y_train_full



# Saving Test Dataset after cleaning

df_test_save=df_all.iloc[ntrain:] # Test set

#df_test_save['Id']=test_ID
df_test_save.head()
df_train_save.to_csv('train_clean.csv', index=False)

df_test_save.to_csv('test_clean.csv', index=False)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

skewness = df_all.select_dtypes(include=numerics).apply(lambda x: skew(x))

skew_index = skewness[abs(skewness) >= 0.85].index

skewness[skew_index].sort_values(ascending=False)
'''BoxCox Transform'''

lam = 0.15



for column in skew_index:

    df_all[column] = boxcox1p(df_all[column], lam)

# Evaluation after working with skewed data

evaluate(df_all)
df_all=pd.get_dummies(df_all)
df_all.shape
"""Dividing working DataFrame back to Train and Test"""

# split Validational/Test set from Training set after Categorical Value Engeneering

#def original_train_test(df_all):

X_test=df_all.iloc[ntrain:] # Test set

X_train_full=df_all.iloc[:ntrain] # Train set

X_train, X_valid, y_train, y_valid = train_test_split(pd.get_dummies(X_train_full), y_train_full)
# Saving all features for future comparison.

all_features = df_all.keys()

# Removing features.

df_all = df_all.drop(df_all.loc[:,(df_all==0).sum()>=(df_all.shape[0]*0.984)],axis=1)

df_all = df_all.drop(df_all.loc[:,(df_all==1).sum()>=(df_all.shape[0]*0.984)],axis=1) 

# Getting and printing the remaining features.

remain_features = df_all.keys()

remov_features = [st for st in all_features if st not in remain_features]

print(len(remov_features), 'features were removed:', remov_features)
# Evaluation after dropping not important features

evaluate(df_all)
from sklearn import preprocessing



scaler = preprocessing.RobustScaler()

df_all = pd.DataFrame(scaler.fit_transform(df_all))
# Evaluation after Normalization

evaluate(df_all)
"""Dividing working DataFrame back to Train and Test"""

# split Validational/Test set from Training set after Categorical Value Engeneering

#def original_train_test(df_all):

X_test=df_all.iloc[ntrain:] # Test set

X_train_full=df_all.iloc[:ntrain] # Train set

X_train, X_valid, y_train, y_valid = train_test_split(pd.get_dummies(X_train_full), y_train_full)
# ElasticNet

print('ElasticNet')

def elastic_score(X,y):

    elastic = ElasticNet(random_state=1)

    param = {'l1_ratio' : [0],

             'alpha' : [0.017]}

    elastic = GridSearchCV(elastic, param, cv=5, scoring='neg_mean_squared_error')

    elastic.fit(X,y)

    print('ElasticNet:', np.sqrt(elastic.best_score_*-1))

    return elastic

elastic_score(X_train, y_train)



# XGBoost

print('XGBoost')

m_xgb = XGBRegressor(n_estimators=160, learning_rate=0.05)

# using early_stop to find out where validation scores don't improve

#m_xgb.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)], verbose=False)

m_xgb.fit(X_train, y_train)

print_score(m_xgb,X_train, X_valid, y_train, y_valid)





# Random Forest

print('Random Forest')

m_rf = RandomForestRegressor(n_estimators=160, min_samples_leaf=1, max_features=0.5, n_jobs=-1, oob_score=True)

m_rf.fit(X_train, y_train)

print_score(m_rf,X_train, X_valid, y_train, y_valid)

def cv_train():

    elastic = ElasticNet(random_state=1)

    param = {'l1_ratio' : [0],

             'alpha' : [0.017]}

    elastic = GridSearchCV(elastic, param, cv=5, scoring='neg_mean_squared_error')

    elastic.fit(X_train_full, y_train_full)

    print('Elastic:', np.sqrt(elastic.best_score_*-1))

    return elastic

elastic = cv_train()
y_pred=np.expm1(elastic.predict(X_test)); y_pred
sub = pd.DataFrame()

sub['Id'] = test_ID

sub['SalePrice'] = y_pred

sub.to_csv('submission_31Aug19.csv',index=False)
sub.head()