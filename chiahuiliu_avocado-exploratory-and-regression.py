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
import os
# print current file directory
print(os.listdir("../input"))

import matplotlib.pyplot as plt
# read csv
# In pandas, you can call function read_csv() to read the csv file with given parameters
# such as the path of the file, and the encoding, 
# if the file is large, you can add parameter "low_memory=False" to read large csv file
df = pd.read_csv("../input/avocado.csv", encoding='utf-8')
#########################
# show some random datapoints
# in pandas, once you put all the file into a dataframe (df in my case), 
# you are able to use function sample(n) to extract n random samples from the dataframe
# if the n is not given, the default value of n is 1
#########################
df.sample(3)
######################
# you can also use .dtypes to see the data types of each column in the dataframe
# int: Integer
# object: String
# float: Float
######################
df.dtypes
######################
# you can also use .info() to see if there are any null value
######################
df.info()
######################
# use describe() to get statistical analysis of the data
######################
df.describe()
df["type"].value_counts()
df["Date"].value_counts()
df["region"].value_counts()

# convert data type of the column "Date"
df["Date"] = pd.to_datetime(df["Date"])
# sample to data points to check the content and the data type
df["Date"].sample(2)
# import packages 
import seaborn as sns
import matplotlib.pyplot as plt
# set the size of the figure
plt.figure(figsize=(16,8))
# set the title
plt.title("Distribution of the Average Price")
# plot the distribution
ax = sns.distplot(df["AveragePrice"])
# set the size of the figure
plt.figure(figsize=(16,8))
# set the title
plt.title("BoxPlot of AveragePrice")
# plot the boxplot
ax = sns.boxplot(df["AveragePrice"])
# set the size of the figure
plt.figure(figsize=(16,8))
# set the title
plt.title("Type v.s. AveragePrice")
# plot Type v.s. AveragePrice
ax = sns.boxplot(y="type", x="AveragePrice", data=df, palette = 'pink')
# conventional avocado X regions X Year
# filter out all conventional avocado (type = conventional)
conventional_avo = df[df["type"].isin(['conventional'])]
# sort by average price
conventional_avo = conventional_avo.sort_values(by='AveragePrice')
# plot
ax = sns.factorplot('AveragePrice','region',data=conventional_avo,
                   hue='year', # category
                   height=13,
                   aspect=0.8,
                   palette='muted',
                   join=False)
# organic avocado X regions X Year
# filter out all organic avocado (type = organic)
organic_avo = df[df["type"].isin(['organic'])]
# sort by average price
organic_avo = organic_avo.sort_values(by='AveragePrice')
# plot
ax = sns.factorplot('AveragePrice','region',data=organic_avo,
                   hue='year', # category
                   height=13,
                   aspect=0.8,
                   palette='muted',
                   join=False)
# plt date vs. AveragePrice
# set the size of the figure
plt.figure(figsize=(16,8))
# set the title
plt.title("Date v.s. AveragePrice")

ax = sns.tsplot(data=df, time="Date", unit="region",condition="type", value="AveragePrice")

# Non-numerical data conversion
# Encode type into dummy variables

# convert type into dummies by separating it into 2 other columns: organic and conventional
dummy_type = pd.get_dummies(df['type'])
# print sample
dummy_type.sample(2)
# concat
df = pd.concat([df, dummy_type], axis=1)
print(df.sample(2))

import matplotlib.pyplot as plt
# quick chack of column "region"
region_dict = dict(df["region"].value_counts())
y_pos = np.arange(len(region_dict))
plt.figure(figsize=(16,18))
plt.barh(y_pos, list(region_dict.values()), align='center', alpha=0.5)
plt.yticks(y_pos, region_dict.keys())
plt.xlabel('Counts')
plt.title('Region Distribution')
len(region_dict)
# covert region to categorical data
df['region'] = df['region'].astype('category')
df.dtypes
df['region'] = df['region'].cat.codes
df['region'].sample(3)
df['Date_Q'] = df['Date'].apply(lambda x: x.quarter)
df['Date_Q'].value_counts()
# plot correlation martix
# set the size of the figure
plt.figure(figsize=(22,12))
# set the title
plt.title("Correlation Matrix")

coe_col = ['AveragePrice', 'Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 
           'year', 'organic', 'conventional', 'Date_Q', 'region']
cm = np.corrcoef(df[coe_col].values.T)
sns.set(font_scale = 1.7)
ax = sns.heatmap(cm,cbar = True, annot = True,square = True, fmt = '.2f', annot_kws = {'size':15}, yticklabels = coe_col, 
                 xticklabels = coe_col)
df.columns
# import packages
from sklearn.model_selection import train_test_split
# split the dataframe to X and Y
X_columns = ['Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 'conventional', 'organic', 
             'Date_Q', 'year', 'region']
X = df[X_columns]
Y = df['AveragePrice']
# check X and Y shape
print('X Shape:', X.shape)
print('Y Shape:', Y.shape)
# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=2018)
print('X_train Shape:', X_train.shape)
print('X_test Shape:', X_test.shape)
print('y_train Shape:', y_train.shape)
print('y_test Shape:', y_test.shape)
# import packages
# for linear regression
import statsmodels.api as sm
from sklearn.metrics import explained_variance_score
# built regression function
model = sm.OLS(y_train, X_train)
res = model.fit()
print(res.summary())
from sklearn.feature_selection import mutual_info_regression
dependencies = mutual_info_regression(X_train, y_train)
column_list = list(X_train.columns)
print('Mean among dependencies of X v.s. Y', np.mean(dependencies))
for i in range(len(dependencies)):
    if dependencies[i] > np.mean(dependencies):
        print('* ', column_list[i], dependencies[i])
    else:
        print(column_list[i], dependencies[i])
X_train.columns
selected_features = ['Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'conventional', 'organic']
X_train_sel = X_train[selected_features]
model_2 = sm.OLS(y_train, X_train_sel)
res_2 = model_2.fit()
print(res_2.summary())
strong_relation_features = ['conventional', 'organic', 'Date_Q', 'year']
X_train_strong = X_train[strong_relation_features]
model_3 = sm.OLS(y_train, X_train_strong)
res_3 = model_3.fit()
print(res_3.summary())
from sklearn.feature_selection import f_regression
f_reg = f_regression(X_train, y_train)
column_list = list(X_train.columns)
print('Mean of F-Regression', np.mean(f_reg[0]))
print('Mean of F-Regression p-value', np.mean(f_reg[1]))
for i in range(len(column_list)):
    print(column_list[i],'\t', f_reg[0][i],'\t', f_reg[1][i])
sorted(f_reg[1])
fre_features = ['conventional', 'organic', '4046', 'Total Volume']
X_train_fre = X_train[fre_features]

model_4 = sm.OLS(y_train, X_train_fre)
res_4 = model_4.fit()
print(res_4.summary())
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_trian_new = scaler.fit_transform(X_train)
model_5 = sm.OLS(y_train, X_trian_new)
res_5 = model_5.fit()
print(res_5.summary())
import xgboost
# XGBoost Regressor
xgb = xgboost.XGBRegressor(n_estimators=500, learning_rate=0.1, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=8)
# fit data
xgb.fit(X_train,y_train)
predictions = xgb.predict(X_test)
print(explained_variance_score(predictions,y_test))
# Calculate R-squared
residuals = y_test - predictions
RMSE = np.sqrt(np.mean(residuals**2))
y_test_mean = np.mean(y_test)
tss =  np.sum((y_test - y_test_mean)**2 ) # total sum of square
rss =  np.sum(residuals**2) # sum of residuals
rsq  =  1 - (rss/tss)
print('R^2 of XGBoost', rsq)
