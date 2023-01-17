import numpy as np

import pandas as pd



# Ignore warnings

import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder 

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from sklearn.linear_model import Ridge

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

import xgboost as xgb

import lightgbm as lgb

import matplotlib.pyplot as plt

import seaborn as sns 

from sklearn.decomposition import FactorAnalysis

from scipy.sparse import hstack

from scipy import sparse

import os,time

import lightgbm as lgb

from sklearn import metrics

# To plot pretty figures

%matplotlib inline
# Some global constants

MAX_FEATURES = 50000

NGRAMS = 3        

MAXDEPTH = 100
## Read train and test data files

tr= pd.read_csv('../input/train.csv')

test= pd.read_csv('../input/test.csv')

tr['SalePrice'].describe()
test.head(5)

tr.head(5)
sns.distplot(tr['SalePrice'])
#skewness and kurtosis

print("Skewness: %f" % tr['SalePrice'].skew())

print("Kurtosis: %f" % tr['SalePrice'].kurt())
var = '1stFlrSF'

data = pd.concat([tr['SalePrice'], tr[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
var = 'GrLivArea'

data = pd.concat([tr['SalePrice'], tr[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
var = 'OverallCond'

data = pd.concat([tr['SalePrice'], tr[var]], axis=1)

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000)
var = 'OverallQual'

data = pd.concat([tr['SalePrice'], tr[var]], axis=1)

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000)
var = 'YearBuilt'

data = pd.concat([tr['SalePrice'], tr[var]], axis=1)

f, ax = plt.subplots(figsize=(20, 10))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90)
#correlation matrix

cm = tr.corr()

f, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(cm, vmax=1, square=True)



#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = cm.nlargest(k, 'SalePrice')['SalePrice'].index

cmat = np.corrcoef(tr[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cmat, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', '1stFlrSF', 'FullBath', 'YearBuilt']

sns.pairplot(tr[cols], size = 2.5)

plt.show()
# Get a list of columns having NAs

def ColWithNAs(x):            

    z = x.isnull()

    df = np.sum(z, axis = 0)       # Sum vertically, across rows

    col = df[df > 0].index.values 

    return (col)



def numericalCol(x):            

     return x.select_dtypes(include=[np.number]).columns.values



# Fill in missing values

def MissingValuesAsOther(t , filler = "other"):

    return(t.fillna(value = filler))





def MissingValuesAsMean(t ):

    return(t.fillna(t.mean(), inplace = True))



def MissingValuesAsZero(t ):

     return(t.fillna(value=0, inplace = True))





# Delete the columns

def ColDelete(x,drop_column):

    x.drop(drop_column, axis=1, inplace = True)





def RegressionEvaluationMetrics(regr,X_test_sparse,y_test_sparse,title):

    predictions=regr.predict(X_test_sparse)

    plt.figure(figsize=(8,6))

    plt.scatter(predictions,y_test_sparse,cmap='plasma')

    plt.title(title)

    plt.show()

    print('MAE:', metrics.mean_absolute_error(y_test_sparse, predictions))

    print('MSE:', metrics.mean_squared_error(y_test_sparse, predictions))

    print('RMSE:', np.sqrt(metrics.mean_squared_error(np.log1p(y_test_sparse), np.log1p(predictions))))



def Regression(regr,X_test_sparse,y_test_sparse):

    start = time.time()

    regr.fit(X_train_sparse,y_train_sparse)

    end = time.time()

    rf_model_time=(end-start)/60.0

    print("Time taken to model: ", rf_model_time , " minutes" ) 

    

def SaveResult(SalePrice,test_ids,file):

    OutputRF = pd.DataFrame(data=SalePrice,columns = ['SalePrice'])

    OutputRF['Id'] = test_ids

    OutputRF = OutputRF[['Id','SalePrice']]

    OutputRF.to_csv(file,index=False)
test.head(3)
#standardizing data

saleprice_scaled = StandardScaler().fit_transform(tr['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
#deleting points

tr.sort_values(by = 'GrLivArea', ascending = False)[:2]

tr = tr.drop(tr[tr['Id'] == 1299].index)
# Assign 'price' column to a variable and drop it from tr

#     Also drop train_id/test_id columns

y = tr['SalePrice']            # This is also the target variable

k = test['Id']

tr.drop( ['SalePrice', 'Id'], inplace = True, axis = 'columns')

test.drop( ['Id'], inplace = True, axis = 'columns')
# Now check that both the datasets have the same columns

test.columns

tr.columns

test.head(3)

tr.head(3)
# Stack both train and test one upon another

frames = [tr,test]

df = pd.concat(frames, axis = 'index')    # Concatenate along index/rows

tr.shape

test.shape

df.shape

df.columns.values
#missing data

total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
#delete the columns Alley/PoolQC/MiscFeature in train dataset , has all NA values

preProcessData = df.dropna(axis=1)

preProcessData.isnull().sum().max()
#   Categorical and Numerical Columns 

#   Number of unique values per column. Maybe some columns are actually categorical in nature

numerical_columns_name = numericalCol(preProcessData)

categorical_columns_name = preProcessData.columns.difference(numerical_columns_name)

preProcessData[categorical_columns_name].head(5)
scaler = StandardScaler()

scaler.fit(preProcessData[numerical_columns_name])

scaled_data = scaler.transform(preProcessData[numerical_columns_name])

scaled_data.shape

type(scaled_data)

model_pca = PCA()

pca = PCA(n_components=4)

pca.fit(scaled_data)

num_pca = pca.transform(scaled_data)



num_pca.shape

type(num_pca)

sparse_num_pca=sparse.csr_matrix(num_pca)
print(pca.components_)



df_comp = pd.DataFrame(pca.components_,columns=numerical_columns_name)



plt.figure(figsize=(15,10))

sns.heatmap(df_comp,cmap='plasma',)
PCADataDF = pd.DataFrame(num_pca, columns = ['P1','P2','P3','P4'])
PCADataDF.head()
len(preProcessData.select_dtypes(include=[np.number]).columns.values)

Nunique = preProcessData[categorical_columns_name].nunique()

Nunique= Nunique.sort_values()

Nunique
#convert categorical variable into dummy

df_dummy = pd.get_dummies(preProcessData)
 #Concatenate Categorical + Numerical Data

df_sp = sparse.hstack([df_dummy,sparse_num_pca], format = 'csr')



df_sp.shape

type(df_sp)
##  Unstack train and test, sparse matrices

df_train = df_sp[ : tr.shape[0] , : ]

df_test = df_sp[test.shape[0] :, : ]

df_train.shape

df_test.shape

#  PArtition datasets into train + validation

#y_train = np.log1p(y)    # Criterion is rmsle

y_train = y

X_train_sparse, X_test_sparse, y_train_sparse, y_test_sparse = train_test_split(

                                     df_train, y_train,

                                     test_size=0.33,

                                     random_state=42

                                     )
# Just check if X_train

type(X_train_sparse)
## Ensemble based prediction



# Instantiate a RandomRegressor object

regr = RandomForestRegressor(n_estimators=1000,       # No of trees in forest

                             criterion = "mse",       # Can also be mae

                             max_features = "sqrt",  # no of features to consider for the best split

                             max_depth= MAXDEPTH,    #  maximum depth of the tree

                             min_samples_split= 2,   # minimum number of samples required to split an internal node

                             min_impurity_decrease=0, # Split node if impurity decreases greater than this value.

                             oob_score = True,       # whether to use out-of-bag samples to estimate error on unseen data.

                             n_jobs = -1,            #  No of jobs to run in parallel

                             random_state=0,

                             verbose = 10            # Controls verbosity of process

                             )

# Do regression

start = time.time()

regr.fit(X_train_sparse,y_train_sparse)

end = time.time()

rf_model_time=(end-start)/60.0

print("Time taken to model: ", rf_model_time , " minutes" )
# 14.2 What is OOB score?

regr.oob_score_
#  Prediction and performance

RegressionEvaluationMetrics(regr,X_test_sparse,y_test_sparse,"Random Forest Model")
# Prediction on test Data

SalePrice = regr.predict(df_test)

SalePrice
OutputRF = pd.DataFrame(data=SalePrice,columns = ['SalePrice'])

OutputRF['Id'] = k

OutputRF = OutputRF[['Id','SalePrice']]

OutputRF
OutputRF.to_csv('submission1.csv', index=False)
# Fit regression model

paramsLs = {'n_estimators': 5000, 

          'max_depth': 3, 

          'min_samples_split': 3,

          'learning_rate': 0.01}

gbr = GradientBoostingRegressor(**paramsLs)

#  Do regression

Regression(gbr,X_test_sparse,y_test_sparse)

 

#  Prediction and performance Of RandomForestRegressor

RegressionEvaluationMetrics(gbr,X_test_sparse,y_test_sparse,"Gradient Boosting Model")



# Prediction on test Data

SalePrice_GB = gbr.predict(df_test)

SalePrice_GB
OutputGB = pd.DataFrame(data=SalePrice_GB,columns = ['SalePrice'])

OutputGB['Id'] = k

OutputGB = OutputGB[['Id','SalePrice']]

OutputGB
OutputGB.to_csv('submission3.csv', index=False)
max_depth = 3

min_child_weight = 10

subsample = 0.5

colsample_bytree = 0.6

objective = 'reg:linear'

num_estimators = 5000

learning_rate = 0.01



xgbm = xgb.XGBRegressor(max_depth=max_depth,

                min_child_weight=min_child_weight,

                colsample_bytree=colsample_bytree,

                objective=objective,

                n_estimators=num_estimators,

                learning_rate=learning_rate)

#gbm.fit(X_test_sparse, y_test_sparse)

#  Do regression

Regression(xgbm,X_test_sparse,y_test_sparse)

 



#  Prediction and performance Of RandomForestRegressor

RegressionEvaluationMetrics(xgbm,X_test_sparse,y_test_sparse,"XGBoost Model")



# Prediction on test Data

SalePrice_XGB = xgbm.predict(df_test)

SalePrice_XGB
OutputXGB = pd.DataFrame(data=SalePrice_XGB,columns = ['SalePrice'])

OutputXGB['Id'] = k

OutputXGB = OutputXGB[['Id','SalePrice']]

OutputXGB
OutputXGB.to_csv('submission4.csv', index=False)