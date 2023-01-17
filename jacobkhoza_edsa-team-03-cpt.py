# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline

#set the plots to use the default seaborn style

sns.set()
# Set the option to view all columns

pd.set_option('display.max_columns', None)
train_df = pd.read_csv('../input/train.csv')

train_df.head()
test_df = pd.read_csv('../input/test.csv')

test_df.head()
test_df.head()
df = train_df.append(test_df)
df.head()
df.tail()
print(train_df.info())
## Get a list of columns except for the SalePrice column

columns = list(df.columns.values)

columns.remove('SalePrice')
#get the length of combined dataframe

df_N = df.shape[0]
#get a list of all columns with atleast 10% missing values

missing_val_cols ={}

for col in columns:

    col_empties_N = df[col][df[col].isnull()].shape[0]

    empties_perc = (col_empties_N/df_N) * 100

    if empties_perc > 10:

        missing_val_cols[col] = empties_perc
pd.Series(missing_val_cols).plot.bar()
df['LotFrontage'][df['LotFrontage'].isnull()] = 0
df['Alley'][df['Alley'].isnull()] = 'None'
df['FireplaceQu'][df['FireplaceQu'].isnull()] = 'None'
df.PoolQC[df.PoolQC.isnull()] = 'None'
df['Fence'][df['Fence'].isnull()] = 'None'
df['MiscFeature'][df['MiscFeature'].isnull()] = 'None'
#let us see how the dataframe now looks like

df.head()
df[df.dtypes[df.dtypes == 'int64'].index].fillna(0,inplace=True)
#use forward fill to fill the missing values

df.fillna(method='ffill',inplace=True)
df.tail()
from collections import defaultdict
object_cols = [] #container to store all column names where the data type is "object"

col_dummies = [] #container to store all dataframes created from pandas get_dummies method



for col in df.columns:

    if str(df[col].dtypes) == 'object':

        object_cols.append(col)

        col_dummy = pd.get_dummies(df[['Id',col]],drop_first=True,prefix=col)

        col_dummies.append(col_dummy)  
#drop off all columns where the data

df_clean = df.drop(object_cols,axis=1)

#now append each dataframe from above to the left of our original dataframe

for index,col in enumerate(object_cols):

    df_clean= pd.merge(df_clean,col_dummies[index],on='Id') 

df_clean.set_index('Id',inplace=True)
train_df.plot.scatter(x='OverallQual',y='SalePrice')
plt.figure(figsize=(8,7))

sns.boxplot(y='SalePrice',x='OverallQual',data=train_df)
train_df.plot.scatter(x='GrLivArea',y='SalePrice')
df_clean['OverallQual^2'] = df_clean['OverallQual'] ** 2

df_clean.drop('OverallQual',inplace=True,axis=1)
df_clean.head()
#df_clean = df_clean.drop(df_clean[(df_clean['GrLivArea']>4000)].index)# Summarize features

df_clean['TotalSF'] = df_clean['1stFlrSF'] + df_clean['2ndFlrSF'] + df_clean['TotalBsmtSF']

df_clean = df_clean.drop(['1stFlrSF','2ndFlrSF','TotalBsmtSF'], axis=1)



df_clean['TotalArea'] = df_clean['LotFrontage'] + df_clean['LotArea']

df_clean = df_clean.drop(['LotFrontage','LotArea'], axis=1)



df_clean['BSF'] = df_clean['BsmtFinSF1'] + df_clean['BsmtFinSF2']

df_clean = df_clean.drop(['BsmtFinSF1','BsmtFinSF2'], axis=1)



df_clean['BsmtBath'] = df_clean['BsmtFullBath'] + (0.5 * df_clean['BsmtHalfBath'])

df_clean = df_clean.drop(['BsmtFullBath','BsmtHalfBath'], axis=1)



df_clean['Bath'] = df_clean['FullBath'] + (0.5 * df_clean['HalfBath'])

df_clean = df_clean.drop(['FullBath','HalfBath'], axis=1)
#get the correlation values against each column

train_length = train_df.shape[0]

corrs = df_clean.iloc[:train_length].corr()
#convert them into a dataframe

corrs_df = pd.DataFrame(corrs.SalePrice)
#pick only those whose correlation with the sale price is over 0.5

contenders = corrs_df[corrs_df.SalePrice > 0.5]

contenders.drop('SalePrice',inplace=True)
#reset the index of the dataframe so that we may be able to access the columns names

contenders.reset_index(inplace=True)
#rename the new column for easy acces

contenders.columns = ['Predictor','Correlation']
#show the correlations in descending order

contenders.sort_values('Correlation',ascending=False)
corrs.loc[contenders.Predictor[:-1]][contenders.Predictor[:-1]]
to_plot = corrs.loc[contenders.Predictor][contenders.Predictor]

plt.figure(figsize=(10,10))

g = sns.heatmap(to_plot,annot=True,cmap="RdYlGn")
to_drop = contenders.iloc[[3,8,5]]

to_drop
contenders = contenders.drop([3,8,5])

contenders.sort_values('Correlation',ascending=False)
corrs.loc['SalePrice'][contenders.Predictor].plot.bar()

plt.title('Top 6 Predictors Correlation to SalePrice')

plt.ylabel('Correlation')
#get the length of the original train_df dataframe

train_length = train_df.shape[0]
#make X by taking only the rows from train_df besides the SalePrice column

X = df_clean.iloc[:train_length].drop('SalePrice',axis=1)
X.head()
#make y by taking all the rows from train_df but only the SalePrice column

y = df_clean.iloc[:train_length]['SalePrice']
#use the remaining data for testing the final model

X_validation = df_clean.iloc[train_length:].drop('SalePrice',axis=1)
X_validation.head()
idx_to_drop = X[(X['GrLivArea']>4000)].index

X.drop(X.loc[idx_to_drop].index,inplace=True)

y.drop(y.loc[idx_to_drop].index,inplace=True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=45)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.metrics import mean_squared_error,r2_score ,mean_squared_log_error
def get_model_scores(y_pred,y_test):

    """

        Calculates and returns the mean squared error,r squared score and mean squared log error of the given input

        

        input: y_pred array-like. Prediction values of a dataset

               y_test array-like. Actual values of a dataset

               

       output: tuple in the form of (mean squared error,r squared,mean squared log error) scores

    """

    mse = mean_squared_error(y_test,y_pred)

    r2 = r2_score(y_test,y_pred)

    msle = mean_squared_log_error(y_test,y_pred)

    return mse,r2,msle
def print_model_score(y_pred,y_test):

    """

        Prints the mse, r-squared and msle score of the inputs

    

        input: y_pred array-like items of the predicted values

               y_test array-like items of the actual values   

    """

    test_mse,test_r2,test_log = get_model_scores(y_pred,y_test)

    print('Test MSE: {}'.format(test_mse))

    print('Test R-Squared Score: {}'.format(test_r2))

    print('Test Mean-Squared Log Error: {}'.format(test_log))
def plot_actual_preds(actuals,preds):

    """

        Plots the actual house prices overlayed on the predicted house prices.

        

        Input: actuals array-like values of target variable

               preds array-like predicted values of the target variable

    """

    plt.plot(actuals,linestyle=None,linewidth=0,marker='o',label='Actual Values',alpha=0.5)

    plt.plot(preds,color='red',linestyle=None,linewidth=0,marker='o',

         label='Predictions',alpha=0.2)

    plt.ylabel('Sale Price')

    plt.legend(loc=(1,1))

    plt.show()
def get_scalers(to_scale):

    return scaler.fit_transform(to_scale)
model_performance ={}
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(scaler.fit_transform(X_train[contenders.Predictor]),np.log(y_train))
y_pred = linear_model.predict(scaler.transform(X_test[contenders.Predictor]))
model_performance['Linear'] = get_model_scores(np.exp(y_pred),y_test)

print_model_score(y_test.values,np.exp(y_pred))
plot_actual_preds(y_test.values,np.exp(y_pred))
from sklearn.linear_model import LassoCV
lasso_model = LassoCV(cv=3)
lasso_model.fit(scaler.fit_transform(X_train),np.log(y_train))
y_pred = lasso_model.predict(scaler.transform(X_test))
model_performance['Lasso'] = get_model_scores(np.exp(y_pred),y_test)

print_model_score(np.exp(y_pred),y_test)
from sklearn.linear_model import RidgeCV
ridge_model = RidgeCV(cv=3)
ridge_model.fit(get_scalers(X_train),np.log(y_train))
y_pred = ridge_model.predict(get_scalers(X_test))
model_performance['Ridge'] = get_model_scores(np.exp(y_pred),y_test)

print_model_score(np.exp(y_pred),y_test)
plot_actual_preds(y_test.values,np.exp(y_pred))
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=200,min_samples_leaf=3

                              ,max_features=0.5,warm_start=True,

                              bootstrap=False,random_state=123,

                                  )
rf_model.fit(scaler.fit_transform(X_train),np.log(y_train))
y_pred = rf_model.predict(scaler.fit_transform(X_test))
model_performance['RandomTree'] =  get_model_scores(np.exp(y_pred),y_test)

print_model_score(np.exp(y_pred),y_test)
rf_fi =pd.DataFrame(rf_model.feature_importances_,index=X_train.columns,columns=['Feature Importance'])

rf_fi = rf_fi[rf_fi['Feature Importance'] > 0].sort_values('Feature Importance',ascending=False)*100
#check how much each feature from our selected features, actually contributes

rf_fi.loc[contenders.Predictor].plot.bar()

plt.title('Top 6 Predictors')

plt.ylabel('Contribution (%)')
rf_fi.loc[contenders.Predictor].sum().plot.bar()

plt.title('Top 6 Predictors Total Contribution')

plt.ylabel('Contribution (%)')
rf_fi.head(13).plot.bar()

plt.title('Top 16 Predictors')

plt.ylabel('Total Contribution (%)')
plot_actual_preds(y_test.values,np.exp(y_pred))
from sklearn.ensemble import GradientBoostingRegressor
xgb_model = GradientBoostingRegressor(alpha=0.95, criterion='friedman_mse',

             learning_rate=0.01, loss='huber',max_features='sqrt'

             ,min_samples_leaf=10,min_samples_split=10,n_estimators=3000,

             random_state=None, subsample=0.4)
xgb_model.fit(scaler.fit_transform(X_train),np.log(y_train))
y_pred = xgb_model.predict(scaler.fit_transform(X_test))
model_performance['XGB'] = get_model_scores(np.exp(y_pred),y_test)

print_model_score(np.exp(y_pred),y_test)
plot_actual_preds(y_test.values,np.exp(y_pred))
def blended_prediction(X):

    return ((0.45 * xgb_model.predict(X)) + \

            (0.25 * lasso_model.predict(X)) + \

            (0.25 * ridge_model.predict(X)) +\

            (0.05 * rf_model.predict(X))

            )
y_pred = blended_prediction(scaler.fit_transform(X_test))
model_performance['Stacked'] =get_model_scores(np.exp(y_pred),y_test)

print_model_score(np.exp(y_pred),y_test)
plot_actual_preds(y_test.values,np.exp(y_pred))
mp_array_dic =  {model:np.array(perf) for model,perf in model_performance.items()}

perf_df = pd.DataFrame(mp_array_dic).transpose()

perf_df.columns = ['MSE','R-Squared','MSLE']
perf_df[['MSE']].plot.bar(legend=None)

plt.title('Mean Sqaured Error per Model')

plt.ylabel("MSE (100 Million)")
perf_df[['R-Squared']].plot.bar(legend=None)

plt.title('R-Sqaured Score per Model')

plt.ylabel("Score")
perf_df[['MSLE']].plot.bar(legend=None)

plt.title('Mean Squared Log Error per Model')

plt.ylabel("Mean Squared Log Error")
y_pred = blended_prediction(scaler.fit_transform(X_validation))

validation_df = X_validation.copy()

validation_df['SalePrice'] = np.exp(y_pred)

validation_df['SalePrice'].tail()
validation_df[['SalePrice']].tail()
validation_df[['SalePrice']].to_csv('submission.csv')