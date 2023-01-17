#importing the necessary modules

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression,RidgeCV, LassoCV, ElasticNetCV

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

#import lightgbm as lgb

#import xgboost as xgb

#Read the necessay train and test .csv files

train = pd.read_csv("../input/train.csv")

print('The size of the train dataset is {}'.format(train.shape))



test = pd.read_csv("../input/test.csv")

print('The size of the test dataset is {}'.format(test.shape))
#display th first 5 rows of the train and test sets

train.head()
test.head()
#extracting the id columns form train and test datasets

id_train = train['Id']

id_test = test['Id']
#removing them form the train and test sets

train.drop('Id', axis = 1, inplace= True)

test.drop('Id', axis = 1, inplace = True)

#we check to see that they are gone
test.head()

test.shape
# We take a look at the states

train.describe()
#we take a look at the different data types present in the train data

train.dtypes.value_counts()
test.dtypes.value_counts()


#we keep this for when we will be separating DATA_ALL back into train and test 



train_rowsize = train.shape[0]

test_rowsize = test.shape[0]

test_rowsize   
train_rowsize
import warnings

warnings.filterwarnings('ignore')



data_all = pd.concat((train, test))

data_all.drop('SalePrice', axis = 1, inplace = True)

data_all.head()
#we check the size of the new dataframe 

print('The shape of the data_all is:  {} '.format(data_all.shape))
#Here is a list of all the features with Nans and the number of null for each features

null_values = data_all.columns[data_all.isnull().any()]

null_features = data_all[null_values].isnull().sum().sort_values(ascending = False)

missing_data = pd.DataFrame({'No of Nulls' :null_features})

missing_data
%matplotlib inline

sns.set_context('talk')

sns.set_style('ticks')

sns.set_palette('dark')



plt.figure(figsize= (16, 8))

plt.xticks(rotation='90')

ax = plt.axes()

sns.barplot(null_features.index, null_features)

ax.set(xlabel = 'Features', ylabel = 'Number of missing values', title = 'Missing data');

# Correlation between the features and the predictor- SalePrice

predictor = train['SalePrice']

fields = [x for x in train.columns if x != 'SalePrice']

correlations = train[fields].corrwith(predictor)

correlations = correlations.sort_values(ascending = False)

# correlations

corrs = (correlations

            .to_frame()

            .reset_index()

            .rename(columns={'level_0':'feature1',

                                0:'Correlations'}))

corrs
plt.figure(figsize= (16, 8))

ax = correlations.plot(kind = 'bar')

ax.set(ylabel = 'Pearson Correlation', ylim = [-0.2, 1.00]);
# Get the absolute values for sorting

corrs['Abs_correlation'] = corrs.Correlations.abs()

corrs
plt.figure(figsize= (16, 8))

sns.set_context('talk')

sns.set_style('white')

sns.set_palette('dark')



ax = corrs.Abs_correlation.hist(bins= 35)



ax.set(xlabel='Absolute Correlation', ylabel='Frequency');

# Most correlated features wrt the abs_correlations

corrs.sort_values('Correlations', ascending = False).query('Abs_correlation>0.45')
missing_data = ['PoolQC',"MiscFeature","Alley", "Fence", "FireplaceQu", 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',

                  'GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 

                  'BsmtHalfBath', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', "MasVnrType", "MasVnrArea",

                  'MSSubClass']



for x in missing_data:

    data_all[x] = data_all[x].fillna(0)
# null_values_2 = data_all.columns[data_all.isnull().any()]

# null_features_2 = data_all[null_values_2].isnull().sum().sort_values(ascending = False)

# missing_data_2 = pd.DataFrame({'No of Nulls' :null_features_2})

# missing_data_2
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

data_all["LotFrontage"] = data_all.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))

data_all['MSZoning'].value_counts(normalize = True)  

#we see that in the MsZoning 77% of the data is RL. so we replace the Missing values here with RL
data_all['MSZoning'] = data_all['MSZoning'].fillna('RL')

data_all['MSZoning'].isnull().any()    # to see if the MSZoning has any missing values(NaNs)
# Utilities

data_all['Utilities'].value_counts(normalize = True)
data_all['Utilities'] = data_all['Utilities'].fillna('AllPub')
# Functional

data_all['Functional'].value_counts(normalize = True)
data_all['Functional'] = data_all['Functional'].fillna('Typ')
data_all['Electrical'] = data_all['Electrical'].fillna(data_all['Electrical'].mode()[0])

data_all['KitchenQual'] = data_all['KitchenQual'].fillna(data_all['KitchenQual'].mode()[0])

data_all['Exterior1st'] = data_all['Exterior1st'].fillna(data_all['Exterior1st'].mode()[0])

data_all['Exterior2nd'] = data_all['Exterior2nd'].fillna(data_all['Exterior2nd'].mode()[0])

data_all['SaleType'] = data_all['SaleType'].fillna(data_all['SaleType'].mode()[0])
null_values_2 = data_all.columns[data_all.isnull().any()]

null_features_2 = data_all[null_values_2].isnull().sum().sort_values(ascending = False)

missing_data_2 = pd.DataFrame({'No of Nulls' :null_features_2})

missing_data_2



print('|\t\t NO MORE MISSING VALUES REMAINING. \n\n\t\t\t...IMPUTING COMPLETED ...')
data_all.dtypes.value_counts()
train_new = data_all[:train_rowsize]

test_new = data_all[train_rowsize:]

test_new.shape
train_new.dtypes.value_counts()
test_new.dtypes.value_counts()
train_new.head()
#This is the separation of features into numerical and catergorical features, to do

#feature engineering on each class of data.



#isolating all the object/categorical feature and converting them to numeric features



train_numericals = train[train_new.select_dtypes(exclude = ['object']).columns]

test_numericals = test[test_new.select_dtypes(exclude = ['object']).columns]



#takeoutthe salesprice from the numerical features

#train_numericals = train_numericals.drop("SalePrice")

train_categcols = train_new.select_dtypes(include = ['object']).columns

test_categcols = test_new.select_dtypes(include = ['object']).columns



train_categoricals = train[train_categcols]

test_categoricals = test[test_categcols]



# train_numeric = train[numerical_cols]

# test_numeric = test[numerical_cols2]



print("Shape of Train Categoricals features : {}".format(train_categoricals.shape))

print("Shape of Train Numerical features : {}\n".format(train_numericals.shape) )



print("Shape of Test Categoricals features : {}".format(test_categoricals.shape))

print("Shape of Test Numerical features : {}".format(test_numericals.shape) )
# Do the one hot encoding on the categorical features

train_dummies = pd.get_dummies(train_new, columns = train_categcols)

test_dummies = pd.get_dummies(test_new, columns = test_categcols)

#align your test and train data

train_encoded, test_encoded = train_dummies.align(test_dummies, join = 'left', axis = 1)

print('\t\tShape of the new encoded train: {}'.format(train_encoded.shape))

print('\n\t\tShape of the new encoded test: {}'.format(test_encoded.shape))

print('\n\t\t\t....Encoding completed.....')
train_encoded.dtypes.value_counts()
#we check for skewness in the float data



skew_limit = 0.75

skew_vals = train_numericals.skew()



skew_cols = (skew_vals

             .sort_values(ascending=False)

             .to_frame()

             .rename(columns={0:'Skewness'})

            .query('abs(Skewness) > {0}'.format(skew_limit)))



skew_cols 
tester = 'LotArea'

fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(16,5))

#before normalisation

train_new[tester].hist(ax = ax_before)

ax_before.set(title = 'Before nplog1p', ylabel = 'Frequency', xlabel = 'Value')



#After normalisation

train_new[tester].apply(np.log1p).hist(ax = ax_after)

ax_after.set(title = 'After nplog1p', ylabel = 'Frequency', xlabel = 'Value')



fig.suptitle('Field "{}"'.format(tester));
print(skew_cols.index.tolist()) #returns a list of the values
#Log transfrom all the numerical features except the Salepice column

for col in skew_cols.index.tolist():

    train_encoded[col] = np.log1p(train_encoded[col])

    test_encoded[col]  = test_encoded[col].apply(np.log1p)  # same thing

print(test_encoded.dtypes.value_counts())

print ('\n\t\t:) Skewed data Transformation Completed :)')
#plotting the distribution curve for the SalePrice

f, ax = plt.subplots(figsize=(12, 6))

#plt.xticks(rotation='90')

sns.distplot(train['SalePrice']);
predictor = np.log1p(train.SalePrice)

#plotting the distribution curve for the SalePrice

f, ax = plt.subplots(figsize=(8, 6))

sns.distplot(predictor);
from sklearn.model_selection import train_test_split





X_train, X_test, y_train, y_test = train_test_split(train_encoded, predictor, 

                                                    test_size=0.3, random_state=42)



print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
#We creat a function for calculating the mean_squared_erros

from sklearn.metrics import mean_squared_error



def rmse (true_data, predicted_data):

    return np.sqrt(mean_squared_error(true_data, predicted_data))
#now to the most fun part. Feature engineering is over!!!

#i am going to use linear regression, L1 regularization, L2 regularization and ElasticNet(blend of L1 and L2)



#LinearRegression

linearRegression = LinearRegression().fit(X_train, y_train)

prediction1 = linearRegression.predict(X_test)

LR_score = linearRegression.score(X_test, y_test)

LR_rmse = rmse(y_test, prediction1)

print('The scoring and root mean squared error for Linear Regression in percentage\n')

print('\t\tThe score is: ',LR_score*100)

print('\t\tThe rmse is : ',LR_rmse*100)
#choose some values of alpha for cross validation.

alphas = [0.005, 0.05, 0.1, 1, 5, 10, 50, 100]


#ridge

ridgeCV = RidgeCV(alphas=alphas).fit(X_train, y_train)

prediction2 = ridgeCV.predict(X_test)

R_score = ridgeCV.score(X_test, y_test)

R_rmse = rmse(y_test, prediction2)

print('The scoring and root mean squared error for Linear Regression in percentage\n')

print('\tThe parameter used for here was alpha = {}\n'.format(ridgeCV.alpha_))

print('\t\tThe score is: ',R_score*100)

print('\t\tThe rmse is : ',R_rmse*100)
#lasso

lassoCV = LassoCV(alphas=[0.005, 0.001, 0.05, 0.01,1, 5], max_iter=1e2).fit(X_train, y_train)

prediction3 = lassoCV.predict(X_test)

L_score = lassoCV.score(X_test, y_test)

L_rmse = rmse(y_test, prediction3)

print('The scoring and root mean squared error for Linear Regression in percentage\n')

print('\tThe parameter used for here was alpha = {}'.format(lassoCV.alpha_))

print('\n\t\tThe score is: ',L_score*100)

print('\t\tThe rmse is : ',L_rmse*100)
#elasticNetCV

l1_ratios = np.linspace(0.1, 0.9, 9)

elasticnetCV = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, max_iter=1e2).fit(X_train, y_train)

prediction4 = elasticnetCV.predict(X_test)

EN_score = elasticnetCV.score(X_test, y_test)

EN_rmse = rmse(y_test, prediction4)

print('The scoring and root mean squared error for Linear Regression in percentage\n')

print('\tThe parameter used for here was alpha = {} and l1_ratios = {} \n'.format(elasticnetCV.alpha_, elasticnetCV.l1_ratio_))

print('\t\tThe score is: ',EN_score*100)

print('\t\tThe rmse is : ',EN_rmse*100)
randfr = RandomForestRegressor(random_state = 42) #random_state to avoid the result from fluctuating
param_grid = { 

    'n_estimators': [50,250,500],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [2, 4, 6, 8, 10],

}
randfr_cv = GridSearchCV(estimator=randfr, param_grid=param_grid, cv= 5)  #cv = 5 to specify the number of folds(5 in this case)  in a stratified Kfold

randfr = randfr_cv.fit(X_train, y_train)
prediction5 = randfr.predict(X_test)

#print(prediction5.shape)

RF_score = randfr.score(X_test, y_test)

RF_rmse = rmse(y_test, prediction5)

print('The scoring and root mean squared error for Linear Regression in percentage\n')

print('\tThe parameter used for here were = {}\n'.format(randfr_cv.best_params_))

print('\t\tThe score for random forest is: {} '.format(RF_score*100))

print('\t\tThe rmse is: {} '.format (RF_rmse*100))
#putting it lall together



score_vals = [LR_score, R_score, L_score, EN_score, RF_score]

rmse_vals = [LR_rmse, R_rmse, L_rmse, EN_rmse, RF_rmse]

labels = ['Linear', 'Ridge', 'Lasso', 'ElasticNet', 'RandomForest']



rmse_df = pd.Series(score_vals, index=labels).to_frame()

rmse_df.rename(columns={0: 'SCORES'}, inplace=1)

rmse_df['RMSE'] = rmse_vals

rmse_df
rmse_df = rmse_df.sort_values(['RMSE'], ascending=True)

rmse_df
from datetime import datetime



start_xgb = datetime.now()



xgb = XGBRegressor().fit(X_train, y_train)



end_xgb = datetime.now()



xgb_time = end_xgb - start_xgb

print('Duration for XGBoost: {}'.format(xgb_time))
prediction6 = xgb.predict(X_test)

xgb_score = xgb.score(X_test, y_test)

xgb_rmse = rmse(y_test, prediction6)

print('The scoring and root mean squared error for XGBoost in percentage\n')

print('\t\tThe score is: ',xgb_score*100)

print('\t\tThe rmse is : ',xgb_rmse*100)
Adding_xgboost = pd.Series({'SCORES': xgb_score, 'RMSE': xgb_rmse}, name = 'XGBoost')

rmse_df = rmse_df.append(Adding_xgboost)

rmse_df
start_lgbm = datetime.now()



lgb = LGBMRegressor().fit(X_train, y_train)



end_lgbm = datetime.now()



lgbm_time = end_lgbm - start_lgbm

print('Duration for Light GBM: {}'.format(lgbm_time))
prediction7 = lgb.predict(X_test)

lgb_score = lgb.score(X_test, y_test)

lgb_rmse = rmse(y_test, prediction7)

print('The scoring and root mean squared error for light GBM in percentage\n')

print('\t\tThe score is: ',lgb_score*100)

print('\t\tThe rmse is : ',lgb_rmse*100)
Adding_lgbm = pd.Series({'SCORES': lgb_score, 'RMSE': lgb_rmse}, name = 'Light GBM')

rmse_df.append(Adding_lgbm)
print('\t\tComparing the 2 model durations:\n')

print('XGBOOST : {} \t\t LIGHT GBM : {}'.format(xgb_time, lgbm_time))
comparisons = {'Scores': (lgb_score, xgb_score), 'RMSE': (lgb_rmse, xgb_rmse), 'Execution Time' : (lgbm_time, xgb_time)}

comparisons_df = pd.DataFrame(comparisons)
comparisons_df.index= ['LightGBM','XGBOOST'] 

comparisons_df
test_encoded.isnull().sum()
null_values = test_encoded.columns[test_encoded.isnull().any()]

null_features = test_encoded[null_values].isnull().sum().sort_values(ascending = False)

missing = pd.DataFrame({'No of Nulls' :null_features})

missing
test_encoded = test_encoded.fillna(0)
prediction = lassoCV.predict(test_encoded) # WE USE THE BEST RMSE which is tht for Lasso

final_prediction = np.exp(prediction) #undoing the np log we did on the saleprices else the resu


House_submission = pd.DataFrame({'Id': id_test, 'SalePrice': final_prediction})

print(House_submission.shape)

House_submission.to_csv('House_prediction.csv', index = False)

print(House_submission.sample(6))