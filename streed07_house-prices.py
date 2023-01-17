import numpy as np

import pandas as pd

from datetime import datetime

from scipy.stats import skew

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV, Ridge, Lasso

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline

from sklearn.compose import make_column_transformer

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from mlxtend.regressor import StackingCVRegressor

import matplotlib.pyplot as plt

import scipy.stats as stats

from scipy.stats import norm, skew

import sklearn.linear_model as linear_model

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

from lightgbm import LGBMRegressor









import warnings

warnings.filterwarnings('ignore')



#Set pandas options to see all rows/columns in output

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)



import os

print("Primary Directory:")

print(os.listdir("../input/"))

print("\nFiles:")

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Read in core files

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



def quick_look(df, df_name):

    print('{} data contains {} observations across {} columns \n'.format(df_name, df.shape[0], df.shape[1]))

    display(df.head(5))



quick_look(train, 'Train')

quick_look(test, 'Test')
#Show difference in columns between train/test

print('Only difference in columns between train and test is our target variable: {}'.format((set(train.columns).symmetric_difference(set(test.columns)))))



#Save and drop ID columns

train_ID = train['Id']

test_ID = test['Id']



#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

train.reset_index(drop=True, inplace=True)
#Make plots white given the dark background

sns.set_style("whitegrid")



#Establish overall figure size for subplots

plt.figure(figsize=[20,5])



#Plot distribution of SalePrice to observe any non-normality

plt.subplot(1,3,1)

sns.distplot(train['SalePrice'] , fit=norm)

plt.title('Distribution of SalePrice')

plt.ylabel('Frequency')

plt.xlabel('SalePrice')





#Transform SalePrice with log1p to normalize distribution

plt.subplot(1,3,2)

train["SalePrice"] = np.log1p(train["SalePrice"])

sns.distplot(train['SalePrice'] , fit=norm)

plt.title('Distribution of SalePrice after Log Transformation')

plt.ylabel('Frequency')

plt.xlabel('SalePrice')



plt.show()
#Create complete dataframe

complete_df = pd.concat((train.iloc[:,0:-1],test.iloc[:,0:len(test.columns)]))



#Function to show missing, unique, and scarcity of a dataframe

def data_quality(df):

    print('Data Quality for Complete dataframe')

    df_temp = pd.DataFrame(index=df.columns, columns=['Missing_Count', 'Unique_Values', 'Scarcity', 'Datatype'])

    df_temp['Missing_Count'] = df.isnull().sum()

    

    df_temp['Unique_Values'] = [df[i].nunique() for i in df.columns]

    df_temp['Scarcity'] = round(((df.isnull().sum())/(df.shape[0])*100),2)

    df_temp['Datatype'] = [df[i].dtype for i in df.columns]

    return df_temp.head(len(df.columns)).sort_values(by=['Missing_Count'], ascending=False)



#Call our function on train

dq_df = data_quality(complete_df)

display(dq_df)
#Create list of numeric columns and categorical columns to be passed into plotting loop

num_cols = train.select_dtypes(include=np.number).columns.tolist()

cat_cols = train.select_dtypes(exclude=np.number).columns.tolist()



#Read in experimental train copy for plotting below

exp_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')



#Create scatter plots to determine if features have linear relationship with target variable (SalePrice) before our transformation

for col in num_cols:

    temp_df = pd.concat([exp_df['SalePrice'], train[col]], axis=1)

    temp_df.plot.scatter(x=col, y='SalePrice', ylim=(0,exp_df['SalePrice'].max()))
#Show correlation heatmap

complete_df_corr = round(abs(complete_df.corr()), 2)

plt.figure(figsize=(25,20))

sns.heatmap(complete_df_corr, cmap="YlGnBu", linewidths=.5, annot=True)

plt.show()
#Drop GarageCars and GarageYrBlt

complete_df.drop(['GarageCars', 'GarageYrBlt'], axis=1, inplace=True)
#Fill appropriate missing values based on column type and data consistency

missing_cols = [i for i in dq_df[dq_df['Scarcity'] > 0].index if i not in {'GarageCars', 'GarageYrBlt'}]

other_cols = ['SaleType', 'Exterior1st', 'Exterior2nd', 'Functional', 'MSZoning']



#Fill in complete dataframe with imputed values of 0, 'Other', 'None'

for i in missing_cols:

    if complete_df[i].dtypes != 'object':

        complete_df[i] = complete_df[i].fillna(complete_df[i].mean())

    elif i in other_cols:

        complete_df[i] = complete_df[i].fillna('Other')

    else:

        complete_df[i] = complete_df[i].fillna('None')
#Recalculate num_cols

num_cols = [i for i in num_cols if i not in {'GarageCars', 'GarageYrBlt', 'SalePrice', 'Id'}]



#Demonstrate skew in all numerical features, with the exception of numerics we converted above

skewed_data = complete_df[num_cols].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skewed_data[skewed_data > .75]

display(high_skew.head(25))

high_skew_feats = high_skew.index



#Log transform skewed features

complete_df[high_skew_feats] = np.log1p(complete_df[high_skew_feats])
#Create new features in our complete df individually



complete_df['TotalSF']= (complete_df['TotalBsmtSF'] +

                        complete_df['1stFlrSF'] + 

                        complete_df['2ndFlrSF'])



complete_df['TotBathrooms'] = ((complete_df['FullBath']) + 

                               (complete_df['HalfBath'] * .5) + 

                               (complete_df['BsmtFullBath']) + 

                               (complete_df['BsmtHalfBath'] * .5))



complete_df['TotPorchSF'] = (complete_df['OpenPorchSF'] + 

                             complete_df['EnclosedPorch'] +

                              complete_df['3SsnPorch'] + 

                             complete_df['ScreenPorch'] +

                              complete_df['WoodDeckSF'])



complete_df['BedToTotRmsRatio'] = (complete_df['BedroomAbvGr'] / complete_df['TotRmsAbvGrd'])
#Label encode for subset of categorical columns that imply order

cat_cols_convert = ['Street', 'BldgType', 'HouseStyle', 'Electrical', 

                    'KitchenQual', 'Functional', 'FireplaceQu', 'GarageQual', 

                    'GarageCond', 'PavedDrive', 'PoolQC']



#Instantiate label encoder and apply to cat_cols_convert from above

le = LabelEncoder()

complete_df[cat_cols_convert] = complete_df[cat_cols_convert].apply(lambda col: le.fit_transform(col))

print('Converted Encoded Features:\n')

display(complete_df[cat_cols_convert].head(5))



#Robust scaler for numeric columns

num_cols = [i for i in num_cols if i not in {'MSSubClass', 'MoSold', 'YrSold'}]

scaler = RobustScaler()

complete_df[num_cols] = scaler.fit_transform(complete_df[num_cols])

print('\nConverted Scaled Numeric features:')

display(complete_df[num_cols].head(5))



#Get Dummies!

#First update cat_cols to remove those that implied order (cat_cols_convert), then get dummies on all

cat_cols = [i for i in cat_cols if i not in {'Street', 'BldgType', 'HouseStyle', 'Electrical','KitchenQual', 'Functional', 'FireplaceQU', 'GarageQual','GarageCond', 'PavedDrive', 'PoolQC'}]

changed_cols = ['MSSubClass', 'MoSold', 'YrSold']

cat_cols = cat_cols + changed_cols

complete_df[cat_cols] = complete_df[cat_cols].astype('category')

complete_df = pd.get_dummies(complete_df)



#Return complete_df ready for modeling

print('\nModel ready dataframe after label encoding, robust scaling, and one hot encoding:')

display(complete_df.head(10))
#Establish y, X_train, X_test

y = train.SalePrice

X_train = complete_df.iloc[:len(y), :]

X_test = complete_df.iloc[len(y):, :]
#Function to run model

def run_model(model):

    result = np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    return(result)
#Gradient Boosting Regressor

model_gbt = GradientBoostingRegressor()

trees = [10, 50, 100, 200, 300]

cv_gbt = [run_model(GradientBoostingRegressor(n_estimators = tree)).mean() for tree in trees]



#Plot Results

cv_gbt_plot = pd.Series(cv_gbt, index = trees)

cv_gbt_plot.plot(title = "GBT Regressor RMSE v. Alpha")

plt.xlabel("Alpha")

plt.ylabel("RMSE")

plt.show()
#LightGBM Regressor

model_GBM = LGBMRegressor()

trees = [10, 50, 100, 200, 400, 1000]

cv_gbm = [run_model(LGBMRegressor(n_estimators = tree)).mean() for tree in trees]



#Plot Results

cv_gbm_plot = pd.Series(cv_gbm, index = trees)

cv_gbm_plot.plot(title = "LightGBM Regressor RMSE v. Alpha")

plt.xlabel("Alpha")

plt.ylabel("RMSE")

plt.show()
#Ridge Linear Regression

model_ridge = Ridge()

alphas = [0.01, 0.05, 0.1, 0.5, 1, 3, 5, 10, 15, 20, 40, 80]

cv_ridge = [run_model(Ridge(alpha = alpha)).mean() for alpha in alphas]



#Plot Results

cv_ridge_plot = pd.Series(cv_ridge, index = alphas)

cv_ridge_plot.plot(title = "Ridge Regression RMSE v. Alpha")

plt.xlabel("Alpha")

plt.ylabel("RMSE")

plt.xticks(np.arange(0, 80, step=5))

plt.yticks(np.arange(0.11, 0.13, step=.002))

plt.show()
#Lasso Linear Regression

model_lasso = Lasso()

alphas = [1, 0.5, 0.1, 0.01, 0.001]

cv_lasso = [run_model(Lasso(alpha = alpha)).mean() for alpha in alphas]



#Plot Results

cv_lasso_plot = pd.Series(cv_lasso, index = alphas)

cv_lasso_plot.plot(title = "Lasso Regression RMSE v. Alpha")

plt.xlabel("Alpha")

plt.ylabel("RMSE")

plt.xticks(np.arange(0, 1, step=.1))

plt.show()
#Generate Predictions

model_ridge = Ridge(alpha=10).fit(X_train,y)

ridge_preds = np.expm1(model_ridge.predict(X_test))
#Demonstrate distribution of error is normal

predicted = model_ridge.predict(X_train).ravel()

actual = y



# Calculate the error, also called the residual.

residual = actual - predicted



plt.hist(residual)

plt.title('Residual Counts - Normal Distribution')

plt.xlabel('Residual')

plt.ylabel('Count')

plt.show()
##Demonstrate distribution of error is homeoscedastic

plt.scatter(predicted, residual)

plt.xlabel('Predicted')

plt.ylabel('Residual')

plt.axhline(y=0)

plt.title('Residual vs. Predicted - Homeoscedastic')

plt.show()
#Submission



final = pd.DataFrame()

final['Id'] = test_ID

final['SalePrice'] = ridge_preds

final.to_csv('submission.csv',index=False)