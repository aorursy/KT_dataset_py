# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.model_selection import train_test_split
from scipy.sparse import  hstack
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from scipy import sparse
from scipy.sparse import  hstack
import time
import os, gc
import sys
# Set display options
#Limiting floats output to 3 decimal points 
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) 
# Functions
def MissingValues(x):
    z1 = x.isnull()
    z2 = np.sum(z1, axis = 0)
    z3 = z2[z2 > 0]
    z4 = ( z3 / len(df) ) * 100
    return (z4)

def DoDummy(x):    
    le = LabelEncoder()
    y = x.apply(le.fit_transform)
    enc = OneHotEncoder(categorical_features = "all")
    enc.fit(y)
    trans = enc.transform(y)
    return(trans)  

def scaleDataset(df):
    col_names = df.columns.values
    ss = StandardScaler()
    return pd.DataFrame(ss.fit_transform(df), columns=col_names)

tr= pd.read_csv("../input/train.csv", header = 0, delimiter=',')
test= pd.read_csv("../input/test.csv", header = 0, delimiter=',')
# View tr data
tr.shape
# View tr data
tr.info()
# View test data
test.shape
test.info()
# Visualization
#dist plot of sale price
sns.distplot(tr.SalePrice)
# Viewing a more normalized distribution by plotting log of Sale Price
logPrice = np.log1p(tr.SalePrice)
sns.distplot(logPrice)
# Correlation of Sale Price (target variable) with other variables
corplot = tr.corr()
corplot.sort_values(["SalePrice"], ascending = False, inplace = True)
print(corplot.SalePrice)
#Plot of the correlation
plt.subplots(figsize=(20,20))
sns.heatmap(corplot)
# From here, we pick 2 variables that show a high correlation with Sale Price
# OverallQual (categorical) & GrLivArea (numeric)

# Plotting OverallQual vs SalePrice
plt.figure(figsize = (12, 6))
sns.boxplot(x = 'OverallQual', y = 'SalePrice',  data = tr) # shows a strong relationship between OverallQual & SalePrice

# Plotting GrLivArea vs SalePrice
plt.figure(figsize = (12, 6))
plt.scatter(x = 'GrLivArea', y = 'SalePrice',  data = tr) # shows an almost linear relationship between GrLivArea & SalePrice

# Here we see 2 outliers that can be removed
tr = tr.drop(tr[(tr['GrLivArea']>4000) & (tr['SalePrice']<300000)].index)

#View the plot again
plt.figure(figsize = (12, 6))
plt.scatter(x = 'GrLivArea', y = 'SalePrice',  data = tr) 

# Data pre-processing
# Drop the Id & SalePrice columns from tr
tr.shape
s1 = tr['SalePrice']
tr.drop( ['Id','SalePrice'], inplace = True, axis = 'columns')
tr.shape
# Drop Id column from test
test.shape
test.drop( ['Id'], inplace = True, axis = 'columns')
test.shape
# Stack the datasets
frames = [tr,test]
df = pd.concat(frames, axis = 'index')    # Concatenate along index/rows
df.shape
# Find cols with missing data
df_na = MissingValues(df).sort_values(ascending=False)
df_na
# Plot of missing values
plt.figure(figsize = (12, 6))
sns.barplot(x = df_na, y = df_na.index.values)
#remove NAs
# Missing values for these fields can be replaced with NA
FillWithNA = ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageFinish',
              'GarageQual', 'GarageCond', 'GarageType', 'BsmtExposure',
              'BsmtCond', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'KitchenQual')

for col in FillWithNA:
    df[col] = df[col].fillna("NA")
# Missing values for these fields can be replaced with 0
FillWith0 = ('LotFrontage', 'GarageYrBlt', 'MasVnrArea', 'BsmtFullBath',
             'BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
             'GarageCars', 'GarageArea', 'TotalBsmtSF')

for col in FillWith0:
    df[col] = df[col].fillna(0)
# Missing values for these fields can be replaced with None
FillWithNone = ('MasVnrType', 'MSZoning', 'Utilities', 'Functional',
             'Exterior2nd', 'Exterior1st', 'Electrical')

for col in FillWithNone:
    df[col] = df[col].fillna("None")
# Missing values for these fields can be replaced with Other
df["SaleType"] = df["SaleType"].fillna("Oth")
# Recheck cols with missing data
df_na = MissingValues(df)
df_na
# Convert categorical data
num_cols = df.select_dtypes(include=[np.number]).columns.values
cat_cols = df.columns.difference(num_cols)

df_dummy = DoDummy(df[cat_cols])
df_dummy.shape
# Now bind both
df_sp = hstack((df_dummy, df[num_cols]), format = "csr")   # Output is csr-sparse format
df_sp.shape
#Unstack tr and test, sparse matrices
df_train = df_sp[ : tr.shape[0] , : ]
df_test = df_sp[tr.shape[0] :, : ]
df_train.shape
df_test.shape
# Partition datasets into train + validation data
y = np.log1p(s1)    
X_train, X_valid, y_train, y_valid = train_test_split(
                                     df_train, y,
                                     test_size=0.60,
                                     random_state=60
                                     )
# Random Forest Regressor
# Instantiate a RandomRegressor object
start = time.time()
regr = RandomForestRegressor(n_estimators=600,       # No of trees in forest
                             criterion = "mse",       # Can also be mae
                             max_features = "sqrt",  # no of features to consider for the best split
                             max_depth= 60,    #  maximum depth of the tree
                             min_samples_split= 2,   # minimum number of samples required to split an internal node
                             min_impurity_decrease=0, # Split node if impurity decreases greater than this value.
                             oob_score = True,       # whether to use out-of-bag samples to estimate error on unseen data.
                             n_jobs = -1,            #  No of jobs to run in parallel
                             random_state=0,
                             verbose = 10            # Controls verbosity of process
                             )


regr.fit(X_train,y_train)

# Prediction and performance
rf_sp=regr.predict(X_valid)
squared = np.square(rf_sp - y_valid)
rf_error = np.sqrt(np.sum(squared)/len(y_valid))

end = time.time()
rf_model_time=(end-start)/60.0

print("Time taken to model: ", rf_model_time , " minutes" )
print("OOB score: ", regr.oob_score_ )
print("RMSE for Random Regressor : ", rf_error)
# Ridge Regression
start = time.time()
modelr = Ridge(alpha = 1.0,            
              solver = "lsqr",        
              fit_intercept=False     
              )

modelr.fit(X_train, y_train)
ridge_pre = modelr.predict(X_valid)
squared = np.square(ridge_pre-y_valid)
ridge_error = np.sqrt(np.sum(squared)/len(y_valid))

end = time.time()
ridge_model_time=(end-start)/60.0

print("Time taken to model: ", ridge_model_time , " minutes" )
print("RMSE for Ridge: ", ridge_error)

