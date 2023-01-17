# To handle datasets

import numpy as np

import pandas as pd



# For plotting

import matplotlib.pyplot as plt

%matplotlib inline

from scipy.stats import norm, skew



# to divide train and test set

from sklearn.model_selection import train_test_split



# Feature scaling 

from sklearn.preprocessing import MinMaxScaler



# for tree binarisation

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score



# To build the models

from sklearn.linear_model import LinearRegression, Lasso

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

import xgboost as xgb



# To evaluate the models

from sklearn.metrics import mean_squared_error

from math import sqrt



pd.pandas.set_option('Display.max_columns', None)



import warnings

warnings.filterwarnings('ignore')
# Load dataset

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



print(train.shape, test.shape)

train.head()
# let's inspect the type of variables in pandas

train.info()
# we also have an Id variable, that we shoulld not use for predictions:

print('Numbers of house id labels unique in train set:', len(train.Id.unique()))

print('Number of house in train set:', len(train))



print('Numbers of house id labels unique in test set:', len(test.Id.unique()))

print('Number of house in test set:', len(test))

# Find the categorical variables

categorical = [var for var in train.columns if train[var].dtype=='O']

print('These are {} categorical variables in train dataset'.format(len(categorical)))



categorical = [var for var in test.columns if test[var].dtype=='O']

print('These are {} categorical variables in test dataset'.format(len(categorical)))

# make a list of the numerical variables first

numerical = [var for var in train.columns if train[var].dtype!='O']

numerical = [var for var in test.columns if test[var].dtype!='O']



# List of variables that contain year information

year_vars = [ var for var in numerical if 'Yr' in var or'Year' in var]

year_vars = [ var for var in numerical if 'Yr' in var or'Year' in var]

# See the results

train[year_vars].head(), test[year_vars].head()
train.groupby('MoSold')['SalePrice'].median().plot()

plt.title('House price variation in the month')

plt.ylabel('Median house price')
# Let's analyize the values of the discrete variable for train



discrete = []



for var in numerical:

    if len(train[var].unique())<20 and var not in year_vars:

        print(var, 'values:', train[var].unique())

        discrete.append(var)

print()        

print('There are {} discrete variables in train'.format(len(discrete)))





discrete = []



for var in numerical:

    if len(test[var].unique())<20 and var not in year_vars:

        print(var, 'values:', test[var].unique())

        discrete.append(var)

print()        

print('There are {} discrete variables in test'.format(len(discrete)))
# Find the continuoues variables in train set

# Let's let's remeber to skip the Id varibale and the traget variable (SalePrice), which are both numerical



numerical = [var for var in numerical if var not in discrete and var not in ['Id','SalePrice'] and var not in year_vars]

print('There are {} numerical and continuous variables'. format(len(numerical)))

# Train

# Let's Analysis the percentage of missing values for each variable 

for var in train.columns:

    if train[var].isnull().sum()>0:

        print(var, train[var].isnull().mean())

        

# Test

# Let's Analysis the percentage of missing values for each variable 

for var in test.columns:

    if test[var].isnull().sum()>0:

        print(var, test[var].isnull().mean())        

        
# Let's count that how many variable we have with missing information 

vars_with_na = [var for var in train.columns if train[var].isnull().sum()>0]

print('Total variables with missing informations in train:', len(vars_with_na))



vars_with_na = [var for var in test.columns if test[var].isnull().sum()>0]

print('Total variables with missing informations in test:', len(vars_with_na))
# Let's inspeact the type of those variables with a lot of missing information in train

for var in train.columns:

    if train[var].isnull().mean()>0.80:

        print(var, train[var].unique())

# Let's inspeact the type of those variables with a lot of missing information in test

for var in test.columns:

    if test[var].isnull().mean()>0.80:

        print(var, test[var].unique())
# Let's take a look of the numerical variable

numerical
# Let's make boxplots to visualise outliers in the continuous variables

# and histagrams to get an idea of the distribution



for var in numerical:

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)

    fig = train.boxplot(column=var)

    fig.set_title("")

    fig.set_ylabel(var)

    

    plt.subplot(1, 2, 2)

    fig = train[var].hist(bins=20)

    fig.set_ylabel('Number of Houses')

    fig.set_xlabel(var)

    plt.show()   
# Outliers in discrete variables

for var in discrete:

    (train.groupby(var)[var].count() / np.float(len(train))).plot.bar()

    plt.ylabel('Percentage of observations per label')

    plt.title(var)

    plt.show()
no_labels_is = []

for var in categorical:

    no_labels_is.append(len(train[var].unique()))

    

tmp = pd.Series(no_labels_is)

tmp.index = pd.Series(categorical)

tmp.plot.bar(figsize=(12,8))

plt.title('Number of categories in categorical variables')

plt.xlabel('Categorical variables')

plt.ylabel('Number of different categories')
# Let's separate into train and test set

X_train = train.copy()

y_train = train.SalePrice.copy()

X_test = test.copy()

#X_train, X_test, y_train, y_test = train_test_split(data, data.SalePrice, test_size= 0.3, random_state=0)

X_train.shape, X_test.shape
# function to calculate elapsed time



def elapsed_years(df, var):

    # capture difference between year variable and year the house was sold

    df[var] = df['YrSold'] - df[var]

    return df
for var in ['YearBuilt','YearRemodAdd','GarageYrBlt']:

    X_train = elapsed_years(X_train, var)

    X_test = elapsed_years(X_test, var)      
# These are the new creeated variables with elapsed

X_train[['YearBuilt','YearRemodAdd','GarageYrBlt']].head()    
# As per actual dataset

train[['YearBuilt','YearRemodAdd','GarageYrBlt']].head()
# Drop the YrSold variable

X_train.drop('YrSold', axis=1, inplace=True)

X_test.drop('YrSold', axis=1, inplace=True)
# print varibales with missing data

# Keep in mind that we created those new temporal varibales, 

# we are going to treat them as numerical and continuous as well:



# Remove YrSold because it is no longer in our dataset

year_vars.remove('YrSold')



# Examine percentage of missing values

for col in numerical+year_vars:

    if X_train[col].isnull().mean()>0:

        print(col, X_train[col].isnull().mean())
# add variable indicating missingness + median imputation



for df in [X_train, X_test]:

    for var in ['LotFrontage','GarageYrBlt']:

        df[var+"_NA"] = np.where(df[var].isnull(), 1, 0)

        df[var].fillna(X_train[var].median(), inplace=True)

     

    

#median imputation



for df in [X_train, X_test]:

    df['MasVnrArea'].fillna(X_train.MasVnrArea.median(), inplace=True)        
# print variables with missing data

for col in discrete:

    if X_train[col].isnull().mean()>0:

        print(col, X_train[col].isnull().mean())
# print variables with missing data

for col in discrete:

    if X_test[col].isnull().mean()>0:

        print(col, X_test[col].isnull().mean())
# print categorical variables with missing data

for col in categorical:

    if X_train[col].isnull().mean()>0:

        print(col, X_train[col].isnull().mean())
# print variables with missing data

for col in categorical:

    if X_test[col].isnull().mean()>0:

        print(col, X_test[col].isnull().mean())
# add label indicating 'Missing' to categorical variables



for df in [X_train, X_test]:

    for var in categorical:

        df[var].fillna('Missing', inplace=True)
# Check absence of null values in trainset

for var in X_train.columns:

    if X_train[var].isnull().sum()>0:

        print(var, X_train[var].isnull().sum())
# Check absence of null values in trainset

for var in X_test.columns:

    if X_test[var].isnull().sum()>0:

        print(var, X_test[var].isnull().sum())
# Fill NA with median value for those variables that show NA only in the test set

#X_train[var].fillna(data[var].median(), inplace=True)



for var in discrete:

    X_test[var].fillna(X_train[var].median(), inplace=True)



for var in numerical:

    X_test[var].fillna(X_train[var].median(), inplace=True)
def tree_binariser(var):

    score_ls = [] # here I will store the mse



    for tree_depth in [1,2,3,4]:

        # call the model

        tree_model = DecisionTreeRegressor(max_depth=tree_depth)



        # train the model using 3 fold cross validation

        scores = cross_val_score(tree_model, X_train[var].to_frame(), y_train, cv=3, scoring='neg_mean_squared_error')

        score_ls.append(np.mean(scores))



    # find depth with smallest mse

    depth = [1,2,3,4][np.argmin(score_ls)]

    #print(score_ls, np.argmin(score_ls), depth)



    # transform the variable using the tree

    tree_model = DecisionTreeRegressor(max_depth=depth)

    tree_model.fit(X_train[var].to_frame(), X_train.SalePrice)

    X_train[var] = tree_model.predict(X_train[var].to_frame())

    X_test[var] = tree_model.predict(X_test[var].to_frame())   
for var in numerical:

    tree_binariser(var)
# Check absence of null values in trainset

for var in X_test.columns:

    if X_test[var].isnull().sum()>0:

        print(var, X_test[var].isnull().sum())
X_train[numerical].head()
# let's explore how many diffrent buckets we have now among our engineered continuous variables

for var in numerical:

    print(var, len(X_train[var].unique()))
for var in numerical:

    X_train.groupby(var)['SalePrice'].mean().plot.bar()

    plt.title(var)

    plt.ylabel('Mean House Price')

    plt.xlabel('Discretised continuous variable')

    plt.show()
def rare_imputation(variable):

    # find frequent labels / discrete numbers

    temp = X_train.groupby([variable])[variable].count()/np.float(len(X_train))

    frequent_cat = [x for x in temp.loc[temp>0.03].index.values]

    

    X_train[variable] = np.where(X_train[variable].isin(frequent_cat), X_train[variable], 'Rare')

    X_test[variable] = np.where(X_test[variable].isin(frequent_cat), X_test[variable], 'Rare')
# find infrequent labels in categorical variables and replace by Rare



for var in categorical:

    rare_imputation(var)

    

# find infrequent labels in categorical variables and replace by Rare

# remember that we are treating discrete variables as if they were categorical



for var in discrete:

    rare_imputation(var)
X_train[categorical].head()
# Check that we haven't created missing values in the submission dataset



for var in X_train.columns:

    if var !='SalePrice' and X_test[var].isnull().sum()>0:

        training_vars_vars.append(var)
# Let's check that is worked

for var in categorical:

    (X_train.groupby(var)[var].count() / np.float(len(X_train))).plot.bar()

    plt.ylabel('Percentage of observations per label')

    plt.title(var)

    plt.show()
def encode_categorical_variables(var, target):

    # Make label to price dictionary

    ordered_lables = X_train.groupby([var])[target].mean().to_dict()

    

    #encode variables

    X_train[var] = X_train[var].map(ordered_lables)

    X_test[var] = X_test[var].map(ordered_lables)

    

  

# Encode labels in categorical vars

for var in categorical:

    encode_categorical_variables(var, 'SalePrice')

    

#Encode labels in discrete vars

for var in discrete:

    encode_categorical_variables(var, 'SalePrice')       
# sanity check: let's see that we did not introduce NA by accident

for var in X_train.columns:

    if var!='SalePrice' and X_test[var].isnull().sum()>0:

        print(var, X_test[var].isnull().sum())
for var in discrete:

    X_test[var].fillna(X_train[var].median(), inplace=True)



#let's inspect the dataset

X_train.head()    
X_test.isnull().mean()
X_train.describe()
# let's create a list of the training variables

training_vars = [var for var in X_train.columns if var not in ['Id','SalePrice']]



print('Total Number of variables to use for training:', len(training_vars))
training_vars
# Applying a log(1+x) transformation to all skewed numeric features

#numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



# Compute skewness

skewed_feats = X_train[training_vars].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_feats})



skewed_feats_test = X_test[training_vars].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness_test = pd.DataFrame({'Skew' :skewed_feats})





skewness.head(20)
# Check on number of skewed features above 75% threshold

skewness = skewness[abs(skewness) > 0.75]

skewness_test = skewness_test[abs(skewness_test) > 0.75]

print("Total number of features requiring a fix for skewness is: {}".format(skewness.shape[0]))

print("Total number of features requiring a fix for skewness is: {}".format(skewness_test.shape[0]))
from scipy.special import boxcox1p

# Now let's apply the box-cox transformation to correct for skewness

skewed_features = skewness.index

lam = 0.15

for feature in skewed_features:

    X_train[training_vars] = boxcox1p(X_train[training_vars], lam)

    

skewed_features = skewness.index

lam = 0.15

for feature in skewed_features:

    X_test[training_vars] = boxcox1p(X_test[training_vars], lam)    
X_train.isnull().mean().sort_values(), X_test.isnull().mean().sort_values()
for var in year_vars:

    X_train[var].fillna(X_train[var].median(), inplace=True)

    

for var in year_vars:

    X_test[var].fillna(X_train[var].median(), inplace=True)    
X_train.head()
# fit scaler

scaler = MinMaxScaler() # create an instance

scaler.fit(X_train[training_vars]) # fit the scaler to the train set for later use
# Defining two rmse_cv functions

def rmse_cv(model):

    rmse = np.sqrt(-cross_val_score(model, X_train[training_vars], y_train, scoring="neg_mean_squared_error", cv = 10))

    return(rmse)
from sklearn.ensemble import GradientBoostingRegressor

# Initiating Gradient Boosting Regressor

model_gbr = GradientBoostingRegressor(n_estimators=3200, 

                                      learning_rate=0.05,

                                      max_depth=9, 

                                      max_features='sqrt',

                                      min_samples_leaf=15, 

                                      min_samples_split=10, 

                                      loss='huber',

                                      random_state=42)



model_gbr.fit(X_train[training_vars], y_train)

gbr_pred = model_gbr.predict(X_train[training_vars])

print('Gradient Boosting train mse: {}'.format(mean_squared_error(y_train, (gbr_pred))))

print('Gradient Boosting tain rmse: {}'.format(sqrt(mean_squared_error(y_train, (gbr_pred)))))

gbr_pred = model_gbr.predict(X_test[training_vars])
# Initiating XGBRegressor

model_xgb = xgb.XGBRegressor(colsample_bytree=0.2,

                             learning_rate=0.06,

                             max_depth=3,

                             n_estimators=3050)



model_xgb.fit(X_train[training_vars], y_train)

xgb_pred = model_xgb.predict(X_train[training_vars])



print('XGBoost train mse: {}'.format(mean_squared_error(y_train, (xgb_pred))))

print('XGBoost train rmse: {}'.format(sqrt(mean_squared_error(y_train,(xgb_pred)))))

xgb_pred = model_xgb.predict(X_test[training_vars])
import lightgbm as lgb

# Initiating LGBMRegressor model

model_lgb = lgb.LGBMRegressor(objective='regression',

                              num_leaves=4,

                              learning_rate=0.05, 

                              n_estimators=10080,

                              max_bin=9, 

                              bagging_fraction=0.80,

                              bagging_freq=5, 

                              feature_fraction=0.5,

                              feature_fraction_seed=9, 

                              bagging_seed=9,

                              min_data_in_leaf=9, 

                              min_sum_hessian_in_leaf=3)



model_lgb.fit(X_train[training_vars], y_train)

lgb_pred = model_lgb.predict(X_train[training_vars])



print('LightGBM train mse: {}'.format(mean_squared_error(y_train, (lgb_pred))))

print('LightGBM train rmse: {}'.format(sqrt(mean_squared_error(y_train,(lgb_pred)))))

lgb_pred = model_lgb.predict(X_test[training_vars])
# Import the model we are using

from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees

model_rf = RandomForestRegressor(n_estimators = 2000, random_state = 42, bootstrap=True,

                          n_jobs=4, verbose=0, max_features='auto')

# Train the model on training data

model_rf.fit(X_train[training_vars], y_train);



rf_pred = model_rf.predict(X_train[training_vars])



print('RandomForest train mse: {}'.format(mean_squared_error(y_train, (rf_pred))))

print('RandomForest train rmse: {}'.format(sqrt(mean_squared_error(y_train,(rf_pred)))))

rf_pred = model_rf.predict(X_test[training_vars])
# Fit regression model

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor



ada_dt_model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=9),

                          n_estimators=1900, random_state=42)



ada_dt_model.fit(X_train[training_vars], y_train)

ada_dt_pred = ada_dt_model.predict(X_train[training_vars])

print('RandomForest train mse: {}'.format(mean_squared_error(y_train, (ada_dt_pred))))

print('RandomForest train rmse: {}'.format(sqrt(mean_squared_error(y_train,(ada_dt_pred)))))

# Predict

ada_dt_pred = ada_dt_model.predict(X_test[training_vars])
from catboost import Pool, CatBoostRegressor, cv

# Initiating CatBoost Regressor model

model_cat = CatBoostRegressor(iterations=2000,

                              learning_rate=0.10,

                              depth=3,

                              l2_leaf_reg=4,

                              border_count=15,

                              loss_function='RMSE',

                              verbose=200)



# Initiating parameters ready for CatBoost's CV function, which I will use below

params = {'iterations':2000,

          'learning_rate':0.10,

          'depth':3,

          'l2_leaf_reg':4,

          'border_count':15,

          'loss_function':'RMSE',

          'verbose':200}



model_cat.fit(X_train[training_vars], y_train)

cat_pred = model_cat.predict(X_train[training_vars])



print('XGBoost train mse: {}'.format(mean_squared_error(y_train, (cat_pred))))

print('XGBoost train rmse: {}'.format(sqrt(mean_squared_error(y_train,(cat_pred)))))

cat_pred = model_cat.predict(X_test[training_vars])
# Fitting all models with rmse_cv function, apart from CatBoost

cv_gbr = rmse_cv(model_gbr).mean()

cv_xgb = rmse_cv(model_xgb).mean()

cv_lgb = rmse_cv(model_lgb).mean()

cv_rf = rmse_cv(model_rf).mean()

cv_ada_dt = rmse_cv(ada_dt_model).mean()

cv_cat = rmse_cv(model_cat).mean()
# Creating a table of results, ranked highest to lowest

results = pd.DataFrame({

    'Model': ['Gradient Boosting Regressor',

              'XGBoost Regressor',

              'Light Gradient Boosting Regressor',

              'Random Forest Regressor',

              'AdaBoostDecisionTreeRegressor',

              'CatBoost'],

    'Score': [cv_gbr,

              cv_xgb,

              cv_lgb,

              cv_rf,

              cv_ada_dt,

              cv_cat]})



# Build dataframe of values

result = results.sort_values(by='Score', ascending=True).reset_index(drop=True)

result.head()
import seaborn as sns

# Plotting model performance

f, ax = plt.subplots(figsize=(10, 6))

plt.xticks(rotation='90')

sns.barplot(x=result['Model'], y=result['Score'])

plt.xlabel('Models', fontsize=15)

plt.ylabel('Model performance', fontsize=15)

plt.ylim(0.10, 0.116)

plt.title('RMSE', fontsize=15)
# Create stacked model

stacked3 = (gbr_pred + xgb_pred + lgb_pred + rf_pred + ada_dt_pred ) / 5

stacked3
#Save the 'Id' column

#train_ID = train['Id']

test_ID = test['Id']
# Setting up competition submission

sub = pd.DataFrame()

sub['Id'] = test_ID

sub['SalePrice'] = stacked3

sub.to_csv('stached3.csv', index=False)