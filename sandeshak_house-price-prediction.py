# Build a model to predict house price



# Load the required libraries



import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split

from scipy.sparse import  hstack

from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb

from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# Prepare and initialize the variables

MAX_FEATURES = 100   

NGRAMS = 2           

MAXDEPTH = 20

# Load the data and verify the contents

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.columns.values



# Let us do a pairplot and see the correlation between Saleprice, Yearbuilt, Overall Quality & HouseStyle

with sns.plotting_context(font_scale=1.5):

    g = sns.pairplot(train[['SalePrice','YearBuilt','HouseStyle', 'OverallQual']],

                     hue='HouseStyle', palette='tab10',size=3)

g.set(xticklabels=[])

sns.utils.plt.show()



# Let us build the model

# Assign the Saleprice column to be used a target variable

y = train['SalePrice']



# Drop the Saleprice and the Id column from both the train and test data.

train.drop( ['SalePrice', 'Id'], inplace = True, axis = 'columns')

test.drop( ['Id'], inplace = True, axis = 'columns')



# Merge both the dataframes for preprocessing and preparing the data. We need to eliminate the missing values for a good prediction

frames = [train,test]

df = pd.concat(frames, axis = 'index')



# These functions identify the NA's and fill in or eliminate them

def ColWithNAs(x):            

    z = x.isnull()

    df = np.sum(z, axis = 0)       # Sum vertically, across rows

    col = df[df > 0].index.values 

    return (col)



def MissingValues(t , filler = "none"):

    return(t.fillna(value = filler))

            

def DoDummy(x):

    le = LabelEncoder()

    y = x.apply(le.fit_transform)

    enc = OneHotEncoder(categorical_features = "all")

    enc.fit(y)

    trans = enc.transform(y)

    return(trans)



# Search for columns having more than 50% of NA's

df.isnull().sum().sort_values()



# We can see the columns "FireplaceQu" "Fence" "Alley" "MiscFeature" & "PoolQC" are having more than 50% of missing values. Hence they don't add much value in the prediction process and we can drop them

df.drop(["FireplaceQu", "Fence", "Alley", "MiscFeature", "PoolQC"], inplace=True, axis=1)



# Identify the numeric columns and fill those missing values with mean values

numCol = df._get_numeric_data().columns

df[numCol].isnull().sum()

for i in numCol:

    df[i].fillna(df[i].mean(), inplace=True)

    

# Verify if all the missing values are updated

df[numCol].isnull().sum()



# Get the Char columns with missing values or NA's and update them as none.

col = ColWithNAs(df)

col                        

df[col] = MissingValues(df[col])



# Convert categorical features to dummy

df_dummy = DoDummy(df[col])

DoDummy(df[col])



# Merge both the numeric and character dataframes in Compresses Sparse Row format

df_sp = hstack((df_dummy, df[numCol]),  format = "csr")

df_sp.shape



# Unstack train and test, sparse matrices

df_train = df_sp[ : train.shape[0] , : ]

df_test = df_sp[train.shape[0] :, : ]

df_train.shape

df_test.shape



# Partition datasets into train + validation

y_train = y

X_train_sparse, X_test_sparse, y_train_sparse, y_test_sparse = train_test_split(

                                     df_train, y_train,

                                     test_size=0.50,

                                     random_state=42

                                     )



# Let us do Ensemble based prediction

# Random Forest Regression

regr = RandomForestRegressor(n_estimators=300,

                             criterion = "mse",

                             max_features = "sqrt",

                             max_depth= MAXDEPTH,

                             min_samples_split= 2,

                             min_impurity_decrease=0,

                             oob_score = True,

                             n_jobs = -1,

                             random_state=0,

                             verbose = 10)

regr.fit(X_train_sparse,y_train_sparse)

rf_pred=regr.predict(X_test_sparse)

forest_mse = mean_squared_error(rf_pred, y_test_sparse)

forest_rmse = np.sqrt(forest_mse)



# Ridge Regression

model = Ridge(alpha = 1.0,        

              solver = "lsqr",     

              fit_intercept=False

              )



model.fit(X_train_sparse, y_train_sparse)

ridge_pred = model.predict(X_test_sparse)

ridge_mse = mean_squared_error(ridge_pred, y_test_sparse)

ridge_rmse = np.sqrt(ridge_mse)



# LightGBM Model

params = {

    'learning_rate': 0.25,

    'application': 'regression',

    'is_enable_sparse' : 'true',

    'max_depth': 5,

    'max_bin' : 50,

    'num_leaves': 60,

    'verbosity': -1,

    'bagging_fraction': 0.5,

    'nthread': 4,

    'metric': 'RMSE'}



d_train = lgb.Dataset(X_train_sparse, label = y_train_sparse)

d_test = lgb.Dataset(X_test_sparse, label = y_test_sparse)

watchlist = [d_train, d_test]



model_lgb = lgb.train(params,

                  train_set=d_train,

                  num_boost_round=240,

                  valid_sets=watchlist,

                  early_stopping_rounds=20,

                  verbose_eval=10)



lgb_pred = model_lgb.predict(X_test_sparse)



lgb_mse = mean_squared_error(lgb_pred, y_test_sparse)

lgb_rmse = np.sqrt(lgb_mse)



# Let us look at the RMSE value of each of the model

print('Random Forest RMSE: %.4f' % forest_rmse)

print('Ridge CV RMSE: %.4f' % ridge_rmse)

print('LightGBM RMSE: %.4f' % lgb_rmse )



# Let us pick the Model with best RMSE value



submission = pd.read_csv("../input/sample_submission.csv", header = 0)

test_predict = model_lgb.predict(df_test)

submission['SalePrice'] = test_predict

submission.head()

submission.to_csv("submit.csv", index = False)