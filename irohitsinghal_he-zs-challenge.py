# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Training data
train = pd.read_csv('../input/yds_train2018.csv')
train=train.groupby(['Year', 'Month', 'Product_ID', 'Country']).agg({
                                              'Sales':'sum',
                                                'S_No':'count',
                                             })
train.rename(columns={'S_No': 'Training Horizon'}, inplace=True)
train.reset_index(inplace=True)
train[['Year', 'Month', 'Product_ID']]=train[['Year', 'Month', 'Product_ID']].astype(int)
train.head()
#Testing data
test = pd.read_csv('../input/yds_test2018.csv')
final=test.copy()
test=test.groupby(['Year', 'Month', 'Product_ID', 'Country']).agg({
                                                'S_No':'count',
                                             })
test.rename(columns={'S_No': 'Training Horizon'}, inplace=True)
test.reset_index(inplace=True)
test[['Year', 'Month', 'Product_ID']]=test[['Year', 'Month', 'Product_ID']].astype(int)
test.head()
test[test['Training Horizon']>1].head()
import dateutil

holidays = pd.read_excel('../input/holidays.xlsx')
holidays['Date']=holidays['Date'].apply(dateutil.parser.parse, dayfirst=False)
holidays['Month']=holidays['Date'].map(lambda x: x.month)
holidays['Year']=holidays['Date'].map(lambda x: x.year)
holidays=holidays.groupby(['Year', 'Month', 'Country']).agg({
                                                'Holiday':'count',
                                             })
holidays.reset_index(inplace=True)

train = train.merge(holidays, on=['Year', 'Month', 'Country'], how='left')
test = test.merge(holidays, on=['Year', 'Month', 'Country'], how='left')

train.head()
test.head()
promotional = pd.read_csv('../input/promotional_expense.csv')
promotional.rename(columns={'Product_Type': 'Product_ID'}, inplace=True)

test = test.merge(promotional, on=['Year', 'Month', 'Product_ID', 'Country'], how='left')
train = train.merge(promotional, on=['Year', 'Month', 'Product_ID', 'Country'], how='left')

test.head()
train.head()
# #Transforming training data
# train['ToM']=train['Month']=train['Month'].map('{:0>2d}'.format)
# train['ToY']=train['Year']=train['Year'].map('{:0>2d}'.format)
# stats=train.groupby(['Country', 'Product_ID']).agg({'Year':'first',
#                                               'Month':'first',
#                                               'ToY':'last',
#                                               'ToM':'last',
#                                               'Sales':'sum',
#                                                 'S_No':'count',
#                                              })
# stats['From']=stats['Year'].astype(str)+stats['Month'].astype(str)
# stats['To']=stats['ToY'].astype(str)+stats['ToM'].astype(str)
# stats.drop(columns=['Year', 'Month', 'ToY', 'ToM'], inplace=True)
# stats.rename(columns={'S_No': 'Training Horizon'}, inplace=True)
# stats.reset_index(inplace=True)
# stats.head(10)
# #Transforming testing data
# test['ToM']=test['Month']=test['Month'].map('{:0>2d}'.format)
# test['ToY']=test['Year']=test['Year'].map('{:0>2d}'.format)
# stats=test.groupby(['Country', 'Product_ID']).agg({'Year':'first',
#                                               'Month':'first',
#                                               'ToY':'last',
#                                               'ToM':'last',
#                                               'S_No':'count',
#                                              })
# stats['From']=stats['Year'].astype(str)+stats['Month'].astype(str)
# stats['To']=stats['ToY'].astype(str)+stats['ToM'].astype(str)
# stats.drop(columns=['Year', 'Month', 'ToY', 'ToM'], inplace=True)
# stats.rename(columns={'S_No': 'Training Horizon'}, inplace=True)
# stats.reset_index(inplace=True)
# stats.head(15)
# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
#missing_values_table(train)
missing_values_table(test)
train[train.Expense_Price.isnull()]['Sales'].plot.hist()
train[~train.Expense_Price.isnull()]['Sales'].plot.hist()
train['Expense_Price_flag']= train.Expense_Price.isnull()
train['Holiday_flag']= train.Holiday.isnull()
test['Expense_Price_flag']= test.Expense_Price.isnull()
test['Holiday_flag']= test.Holiday.isnull()
train.head()
# one-hot encoding of categorical variables
train = pd.get_dummies(train)
test = pd.get_dummies(test)

print('Training Features shape: ', train.shape)
print('Testing Features shape: ', test.shape)
# from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

train_labels = train['Sales'].astype(int)
# Drop the target from the training data
train_data = train.drop(columns = ['Sales'])
test_data = test.drop(columns = [])
    
# Feature names
features = list(train_data.columns)

# Median imputation of missing values
imputer = SimpleImputer(strategy = 'median')

# Scale each feature to 0-1
# scaler = MinMaxScaler(feature_range = (0, 1))

# Fit on the training data
# imputer.fit(train)

# Transform both training and testing data
train_data = imputer.fit_transform(train_data)
test_data = imputer.transform(test_data)

# Repeat with the scaler
# scaler.fit(train)
# train = scaler.transform(train)
# test = scaler.transform(test)

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor

# Make the random forest classifier
clf = RandomForestClassifier(n_estimators = 400, random_state = 50, verbose = 1, n_jobs = -1)

# # Make the model with the specified regularization parameter
# clf = LogisticRegression(C = 0.0001)

# #Use XGBooster
# clf = XGBRegressor(n_estimators=100, learning_rate=0.1, n_jobs=-1)
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    train_data, train_labels, test_size=0)
# Train on the training data
# clf.fit(X_train, y_train, early_stopping_rounds=5, 
#              eval_set=[(X_test, y_test)], verbose=True)
clf.fit(X_train, y_train)
# print(clf.feature_importances_)
#clf.score(X_test, y_test)

# plt.figure(figsize = (8, 6))

# # Heatmap of correlations
# sns.heatmap(train.corr(), cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
# plt.title('Correlation Heatmap');
train.Sales.describe()
final['Sales']=clf.predict(test_data)
final[['S_No', 'Year', 'Month', 'Product_ID', 'Country', 'Sales']].to_csv('submission.csv', index=False)
final.head()
