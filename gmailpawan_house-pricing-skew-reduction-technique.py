# Author: Pawan Kumar

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# For processing the data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #conversion of categorical to dummy
from sklearn.model_selection import train_test_split # For splitting arrays into train/test
from scipy.sparse import  hstack # Manipulate sparse matricies
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from scipy.sparse import csr_matrix
import seaborn as sns
from scipy.stats import skew
from scipy.sparse import csr_matrix
from scipy import sparse
import matplotlib.pyplot as plt

#  For modeling
from sklearn.ensemble import RandomForestRegressor 
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LassoCV

#  Miscellenous libraries
import time # Timing processes
import os, gc # OS related and releasing memory
import sys # System related
# Read train and test data files
os.chdir("../input")
tr= pd.read_csv("train.csv", header = 0, delimiter=',')  
test= pd.read_csv("test.csv", header = 0, delimiter=',') 
# 7.1 Shape of both the dataset and Comapre the columns
tr.shape, test.shape, tr.columns.difference(test.columns)
# Sales Price Visualisation
tr['SalePrice'].describe()
sns.distplot(tr['SalePrice'], color ="r")
print("Skew of Rawdata: %f" % tr['SalePrice'].skew())
print("Kurt of Rawdata: %f" % tr['SalePrice'].kurt())
# Reduction of Skewness by taking Logrithm of Variable
SingleLog_y = np.log1p(tr['SalePrice'])              # Log transformation of the target variable
sns.distplot(SingleLog_y, color ="r")
print("Skew after 1st Log Transformation: %f" % SingleLog_y.skew())
print("Kurt after 1st Log Transformation: %f" % SingleLog_y.kurt())
# Further Reduction of Skewness by taking Logrithm of Logrithm
DoubleLog_y = np.log1p(SingleLog_y)
sns.distplot(DoubleLog_y, color ="r")
print("Skew after 2nd Log Transformation: %f" % DoubleLog_y.skew())
print("Kurt after 2nd Log Transformation: %f" % DoubleLog_y.kurt())
# Correlation Matrix
k = 10 #number of variables for heatmap
corrmat = tr.corr()
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(tr[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
# Data Alignment between train & test data
tr.drop( ['SalePrice', 'Id'], inplace = True, axis = 'columns')
test.drop( ['Id'], inplace = True, axis = 'columns')    
# Verified that both the datasets have the same columns
tr.columns.difference(test.columns)
# Stacking train and test one upon another for preprocessing and Segragating Numdata & Categorical Data
frames = [tr,test]
combo = pd.concat(frames, axis = 'index')    # Concatenate along index/rows
Numdata= combo.select_dtypes(include=[np.number]) # Segragating Numerical data
Catdata= combo.select_dtypes(exclude=[np.number]) # Segragating Non-Numerical data
Numdata.shape, Catdata.shape
## Fill in missing values
def MissingValues(t , filler = "other"):
    return(t.fillna(value = filler))
    
def MissingValuesAsMean(t):
    return(t.fillna(t.mean(), inplace = True))

from math import log10  #instead of zero, log10(1.1) choosen to fill the na for better responding by SVD.
def MissingValuesAsNearZero(t):
     return(t.fillna(value=log10(1.1)))

# Delete the columns
def ColDelete(x,drop_column):
    x.drop(drop_column, axis=1, inplace = True)


def ColWithNAs(x):      
    y=x.isnull().sum()
    z=y[y>0].sort_values(ascending=False)
    return (z)
# Checking the missing values for Categorical Data
Cat_nas= ColWithNAs(Catdata)
Cat_nas # 23 columns
# Dropped the Columns with excessasive missing values. 
ColDelete(Catdata, ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu' ])

Cat_nas= ColWithNAs(Catdata)
Cat_nas # 18 columns
# Replaced missing values on remaining categorical columns with "Others" 
Cat_nas= ColWithNAs(Catdata)
Cat_nas # 18 columns
Cat_nas.count()
Cat_index=Cat_nas.index.values
Cat_index
Catdata = MissingValues(Catdata) 
Cat_nas= ColWithNAs(Catdata)
Cat_nas.count()
Catdata.info()
# Replacing missing values on Numdata with " Near zero" 

Num_nas= ColWithNAs(Numdata)
Num_index=Num_nas.index.values
Num_index

# Numdata - Filling NA Values by Near Zero Value

Numdata = MissingValuesAsNearZero(Numdata) # Filled by near Zero
Numdata.info()
# Log transformation of numerical column with High Skewness
Num_skewed = Numdata.apply(lambda x: skew(x)) #compute skewness
Num_skewed = Num_skewed[Num_skewed > 0.70]     #identify column with skewness >0.70 
#(0.70 identified through various trial and run)
Num_skewed = Num_skewed.index
Numdata[Num_skewed] = np.log1p(Numdata[Num_skewed])
Num_skewed
# Convert categorical features to dummy
def DoDummy(x):
    le = LabelEncoder()
    y = x.apply(le.fit_transform)
    enc = OneHotEncoder(categorical_features = "all")  # ‘all’: All features are treated as categorical.
    enc.fit(y)
    trans = enc.transform(y)
    return(trans)   


# SVD trasnformation, NC choosen by trial & error based on precision
def doSVD(x, nc=100): 
    svd = TruncatedSVD(n_components=nc, n_iter=10, random_state=42)
    abc = svd.fit_transform(x)
    print(np.sum(svd.explained_variance_ratio_))
    return (abc)

# Convert categorical data to dummy
Cat_dummy = DoDummy(Catdata)
Cat_dummy.shape, type(Cat_dummy)
#Scaling, PCA & sparse of Numerical data
scaler = StandardScaler()
scaler.fit(Numdata)
scaled_data = scaler.transform(Numdata)
model_pca = PCA()
pca = PCA()
pca.fit(scaled_data)
num_pca = pca.transform(scaled_data)
sparse_num_pca=sparse.csr_matrix(num_pca)
sparse_num_pca.shape, type(sparse_num_pca)
# Concatenate Categorical dummy + Numerical Data PCA
df_sp = sparse.hstack([Cat_dummy,sparse_num_pca], format = 'csr')

df_sp.shape, type(df_sp)
#Reduce dimensionality by SVD
combo_svd = doSVD(df_sp)
combo_svd.shape, type(combo_svd)
# Unstack tr and test, sparse matrices
X_train = combo_svd[ : tr.shape[0] , : ]
X_test = combo_svd[tr.shape[0] :, : ]
X_train.shape, X_test.shape
# PArtition datasets into train + validation
X_train_sparse, X_test_sparse, y_train_sparse, y_test_sparse = train_test_split(
                                     X_train, DoubleLog_y,
                                     test_size=0.50,
                                     random_state=42
                                     )
type(X_train_sparse)
# Ensemble based prediction - Random Forest Regressor

MAX_FEATURES = 500000   
NGRAMS = 3           
MAXDEPTH = 35        

regr = RandomForestRegressor(n_estimators=300,       # No of trees in forest
                             criterion = "mse",       # Can also be mae
                             max_features = "sqrt",  # no of features to consider for the best split
                             max_depth= MAXDEPTH,    #  maximum depth of the tree
                             min_samples_split= 2,   # minimum number of samples required to split an internal node
                             min_impurity_decrease=0, # Split node if impurity decreases greater than this value.
                             oob_score = True,       # whether to use out-of-bag samples to estimate error on unseen data.
                             n_jobs = -1,            #  No of jobs to run in parallel
                             random_state=0,
                             verbose = 0            # Controls verbosity of process
                             )

regr.fit(X_train_sparse,y_train_sparse)

# OOB score
regr.oob_score_
# Prediction and performance
rf_sparse=regr.predict(X_test_sparse)
squared = np.square(rf_sparse - y_test_sparse)
rf_error = np.sqrt(np.sum(squared)/len(y_test_sparse))
rf_error
# Gradient Boosting Model - Lightgbm model

params = {
    'learning_rate': 0.25,
    'application': 'regression',
    'is_enable_sparse' : 'true',
    'max_depth': 3,
    'num_leaves': 60,
    'verbosity': -1,
    'bagging_fraction': 0.5,
    'nthread': 4,
    'metric': 'RMSE'
}

d_train = lgb.Dataset(X_train_sparse, label=y_train_sparse)
d_test = lgb.Dataset(X_test_sparse, label = y_test_sparse)
watchlist = [d_train, d_test]

gbm_model = lgb.train(params,
                  train_set=d_train,
                  num_boost_round=240,
                  valid_sets=watchlist,
                  early_stopping_rounds=20,
                  verbose_eval=0)
    
lgb_pred = gbm_model.predict(X_test_sparse)
squared = np.square(lgb_pred - y_test_sparse)
lgb_error = np.sqrt(np.sum(squared)/len(y_test_sparse))
lgb_error
# Linear Models: Ridge Regression
Ridge_model = Ridge(alpha = 20)
Ridge_model.fit(X_train_sparse, y_train_sparse)
ridge_pre = Ridge_model.predict(X_test_sparse)
squared = np.square(ridge_pre-y_test_sparse)
ridge_error = np.sqrt(np.sum(squared)/len(y_test_sparse))
ridge_error
# Parameter Tuning
def rmse_ridge(Ridge_model):
    Ridge_model.fit(X_train_sparse, y_train_sparse)
    ridge_pre = Ridge_model.predict(X_test_sparse)
    squared = np.square(ridge_pre-y_test_sparse)
    ridge_error = np.sqrt(np.sum(squared)/len(y_test_sparse))
    return(ridge_error)

alphas = [0.01, 0.02, 0.1, 0.3, 1, 3, 5, 10, 20, 30,40,100]
cv_ridge = [rmse_ridge(Ridge(alpha = alpha )).mean() 
            for alpha in alphas]

cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = " Parameter tuning - Ridge")
#plt.xlabel("alpha")
#plt.ylabel("rmse")
cv_ridge.min()
model_lasso = LassoCV(alphas = [0.001,0.005,0.01, 0.02,0.03,0.04, 0.1, 0.3, 1, 3, 5, 10]).fit(X_train_sparse, y_train_sparse)

coef = pd.Series(model_lasso.coef_)
rmse_ridge(model_lasso).mean(), str(sum(coef != 0)), str(sum(coef == 0))

# RMSE of Models used
lgb_error, rf_error, ridge_error
submission = pd.read_csv("../input/sample_submission.csv", header = 0)
test_predict = Ridge_model.predict(X_test)
test_predict.shape
firstantilog_SalePrice = np.expm1(test_predict)
PredictedSalePrice= np.expm1(firstantilog_SalePrice)
submission['SalePrice'] = PredictedSalePrice
submission.head(10)