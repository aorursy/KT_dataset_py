%reset -f

import numpy as np  

import pandas as pd  

from sklearn.decomposition import PCA

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.preprocessing import StandardScaler  

from sklearn.model_selection import train_test_split            # For Splitting data

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

import lightgbm as lgb

from sklearn.linear_model import Ridge



import matplotlib.pyplot as plt

# %matplotlib inline

import seaborn as sns



from scipy import sparse

from scipy.sparse import  hstack



import os

import time



import xgboost as xgb

from sklearn.metrics import mean_squared_error

def scaleDataset(df):

  col_names = df.columns.values

  ss = StandardScaler()

  return pd.DataFrame(ss.fit_transform(df), columns=col_names)
def DoDummy(x):

    # Try: le.fit_transform(list('abc'))

    le = LabelEncoder()

    # Apply across all columns of x

    y = x.apply(le.fit_transform)

    # Try:  enc.fit_transform([[1,2],[2,1]]).toarray()

    enc = OneHotEncoder(categorical_features = "all")  # ‘all’: All features are treated as categorical.

    enc.fit(y)

    trans = enc.transform(y)

    return pd.DataFrame(trans.toarray())
train_df= pd.read_csv("../input/train.csv", header = 0)  # 340MB

test_df= pd.read_csv("../input/test.csv", header = 0)
## Correlation plot

corrMatrix=train_df.corr()

sns.set(font_scale=1.10)

plt.figure(figsize=(20, 20))



sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,

            square=True,annot=True,cmap='viridis',linecolor="white")

plt.title('Correlation between features');
##Univariate Plot of Salesprice

sns.distplot(train_df['SalePrice'], color="r", kde=False)

plt.title("Distribution of Sale Price")

plt.ylabel("Number of Occurences")

plt.xlabel("Sale Price");
#Shows right skweness and peakedness

print ('Skweness and Peakedness of SalePrice ',train_df['SalePrice'].skew(),train_df['SalePrice'].kurt())

sns.jointplot(x='SalePrice', y='YearBuilt', data=train_df,kind = 'hex');

plt.show()
sns.jointplot(x='SalePrice', y='YearRemodAdd', data=train_df,kind = 'hex');

plt.show()
train_df.shape
sns.pairplot(train_df.iloc[:,[17,38,43,46,62,80]], size=2);
fig,ax = plt.subplots(2,2, figsize=(10,10))

sns.stripplot(x="GarageType", y="SalePrice", data=train_df,jitter=True,ax= ax[0,0]);

sns.stripplot(x="GarageQual", y="SalePrice", data=train_df, jitter=True, ax=ax[0,1]);

sns.swarmplot(x="GarageCond", y="SalePrice",data=train_df, ax=ax[1,0]);

sns.swarmplot(x="GarageFinish", y="SalePrice",data=train_df, ax=ax[1,1]);
train_df.info()
y = train_df['SalePrice'] 

Id = test_df['Id']

train_df = train_df.drop('SalePrice', axis=1)

all_data = pd.concat([train_df, test_df], ignore_index=True)

ID = all_data['Id']

all_data = all_data.drop('Id', axis=1)
null_columns=all_data.columns[all_data.isnull().any()]

null_coulmn = all_data[null_columns].isnull().sum()

null_coulmn.sort_values(ascending=False)
f, ax = plt.subplots(figsize=(10, 8))

plt.xticks(rotation='90')

sns.barplot(x = null_coulmn.index, y = 100*null_coulmn)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percentage of missing values', fontsize=10)

plt.title('Percentage of missing data of features', fontsize=10)
numerical_columns_name = all_data.select_dtypes(include=[np.number]).columns.values

categorical_columns_name = all_data.columns.difference(numerical_columns_name)
for col in (categorical_columns_name):

    all_data[col] = all_data[col].fillna('None')
for col in (numerical_columns_name):

    all_data[col] = all_data[col].fillna(0)
all_data.columns[all_data.isnull().any()]
cols = train_df.select_dtypes(exclude = [np.number]).columns.values

print('numerical columns:', train_df.select_dtypes(include = [np.number]).columns.values.shape[0])

print('categorial columns:', cols.shape[0])
## Convert Categoricals variables to DummayVars

dummy_data = DoDummy(all_data[categorical_columns_name])
##Scale & Reduce dimensions

scale_data = scaleDataset(all_data[numerical_columns_name])
model_pca = PCA()

pca = PCA(n_components=4)

pca.fit(scale_data)

num_pca = pca.transform(scale_data)



num_pca.shape

type(num_pca)

sparse_num_pca=sparse.csr_matrix(num_pca)
# 11. Bind now dummy array with tfidif array, horizontally

#     Note that this different from np.hstack

df_sp = hstack((scale_data,sparse_num_pca), format = "csr")   # Output is csr-sparse format

df_sp.shape

type(df_sp)
# 12. Unstack tr and test, sparse matrices

df_train = df_sp[ : train_df.shape[0] , : ]

df_test = df_sp[train_df.shape[0] :, : ]

df_train.shape

df_test.shape
#   6.3 Add Id column back to the dataset

all_data.insert(0, 'Id', ID)
X_train, X_valid, y_train, y_valid = train_test_split(

                                     df_train, y,

                                     test_size=0.30,

                                     random_state=101

                                     )
MAX_FEATURES = 2100   # max text features-- try: 50000

NGRAMS = 1           # max number of pharses -- try: 3

MAXDEPTH = 50  
# 14. Instantiate a RandomRegressor object

start = time.time()

regr = RandomForestRegressor(n_estimators=380,       # No of trees in forest

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

# 14.3 Prediction and performance

rf_sparse=regr.predict(X_valid)

print('RMSE:', np.sqrt(mean_squared_error(np.log1p(y_valid), np.log1p(rf_sparse))))

end = time.time()

rf_model_time=(end-start)/60.0

print("Time taken to model: ", rf_model_time , " minutes" ) # 6 minutes
start = time.time()

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.0817, n_estimators=2855,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =42, nthread = -1)

model_xgb.fit(X_train,y_train)

xgb_train_pred = model_xgb.predict(X_valid)

print('RMSE:', np.sqrt(mean_squared_error(np.log1p(y_valid), np.log1p(xgb_train_pred))))

end = time.time()

rf_model_time=(end-start)/60.0

print("Time taken to model: ", rf_model_time , " minutes" )
start = time.time()

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)



model_lgb.fit(X_train,y_train)

lgb_train_pred = model_lgb.predict(X_valid)

print('RMSE:', np.sqrt(mean_squared_error(np.log1p(y_valid), np.log1p(lgb_train_pred))))

end = time.time()

rf_model_time=(end-start)/60.0

print("Time taken to model: ", rf_model_time , " minutes" )
start = time.time()

grad_boost = GradientBoostingRegressor(learning_rate=0.02, loss='huber', max_depth=2, 

                                       max_features='log2', min_samples_leaf=14, min_samples_split=14, 

                                       n_estimators=2850,

                                       random_state=42)





grad_boost.fit(X_train, y_train)

gboost_train_pred = grad_boost.predict(X_valid)

print('RMSE:', np.sqrt(mean_squared_error(np.log1p(y_valid), np.log1p(gboost_train_pred))))

end = time.time()

rf_model_time=(end-start)/60.0

print("Time taken to model: ", rf_model_time , " minutes" )
grad_boost_pred_final = grad_boost.predict(df_test)

## Submission into kaggle

sub = pd.DataFrame()

sub['Id'] = Id

sub['SalePrice'] = grad_boost_pred_final

sub.head()

sub.to_csv('grad_boost_pred_final.csv', index=False)
xgb_train_pred_final = model_xgb.predict(df_test)

## Submission into kaggle

sub = pd.DataFrame()

sub['Id'] = Id

sub['SalePrice'] = xgb_train_pred_final

sub.head()

sub.to_csv('xgb_train_pred_final.csv', index=False)