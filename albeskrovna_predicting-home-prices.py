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
# importing other libraries I will be using
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
#importing the data file
data = pd.read_csv("../input/train.csv")
data.shape
data.head()
data.columns
df=data[['LotArea', 'Utilities', 'Neighborhood', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 
        'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'GrLivArea', 'FullBath', 'HalfBath',
        'BedroomAbvGr', 'KitchenAbvGr', 'MoSold', 'YrSold', 'SalePrice']]
df.head()
# Renaming the columns 
df.columns=['LotArea', 'Utilities', 'NBHD', 'BldgType', 'HouseStyle', 'OverallQlt', 'OverallCond', 'YearBuilt', 'YearRemod',\
           'RoofStyle', 'RoofMatl', 'LivingArea', 'FullBath', 'HalfBath', 'Bedroom', 'Kitchen', 'MonthSold', 'YearSold',\
           'SalePrice']
df.info()
df['YearSold'].unique()
df['LotArea'].describe()
_=plt.figure(figsize = (10,7))
_=plt.boxplot(df['LotArea'],0, 'gD')
_=plt.title('Lot Area Boxplot')
len(df[df['LotArea']>100000])/len(df)
len(df[df['LotArea']>50000])/len(df)
# dropping everything above 50,000
df = df[df['LotArea']<50000]
df.shape
df['Utilities'].value_counts()
df.drop('Utilities', inplace=True, axis=1)
df['LivingArea'].describe()
_=plt.figure(figsize=(10,7))
_=plt.boxplot(df['LivingArea'], 0, 'gD')
_=plt.title('Living Area Boxplot')
len(df[df['LivingArea']>4000])
len(df[df['LivingArea']>3000])
df = df[df['LivingArea']<3000]
_=plt.figure(figsize=(10,7))
_=plt.boxplot(df['Bedroom'], 0, 'gD')
_=plt.title('Bedroom Boxplot')
# checking the perncentage of observations with 6 bedrooms against total observations number
len(df[df['Bedroom'] > 5])/len(df)
# dropping all observations with 6 bedrooms
df = df[df['Bedroom'] < 6]
_=plt.figure(figsize=(10,7))
_=plt.boxplot(df['Bedroom'], 0, 'gD')
_=plt.title('Bedroom Boxplot')
sns.countplot(y = "NBHD", data = df)
salescount = pd.DataFrame(df['NBHD'].value_counts())
salescount.columns = ['Total Number of Sales']
salescount.head(5)
df2=pd.DataFrame(pd.pivot_table(df, index='YearSold', values='SalePrice'))
df2.reset_index(level=0, inplace=True)
df2.columns=['years', 'avprices']
df2
years = ('2006', '2007', '2008', '2009', '2010')
y_pos = np.arange(len(years))
perf = df2['avprices']
  
plt.bar(y_pos, perf, align='center', alpha=0.5)
plt.xticks(y_pos, years)
plt.ylabel('Average Price')
plt.title('Average Prices Distribution by Year')
plt.show()
r=pd.DataFrame(pd.pivot_table(df, index='YearSold', values='SalePrice', aggfunc='sum'))
r.reset_index(level=0, inplace=True)
r.columns=['yrs', 'totalsum']
r
years = ('2006', '2007', '2008', '2009', '2010')
y_pos = np.arange(len(years))
performance = r['totalsum']

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, years)
plt.ylabel('Total Sales')
plt.title('Total Sales Distribution by Year')
plt.show()
v=pd.DataFrame(df['YearSold'].value_counts())
v.reset_index(level=0, inplace=True)
v.columns=['year', 'price']
v
years = ('2006', '2007', '2008', '2009', '2010')
y_pos = np.arange(len(years))
performance = v['price']
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, years)
plt.ylabel('Number of Sales Closed')
plt.title('Number of Sales Closed by Year')
plt.show()
sns.heatmap(df.corr())
df.head()
new_df = pd.get_dummies(data=df, columns=['NBHD','BldgType','HouseStyle','YearBuilt',
                                          'YearRemod','RoofStyle','RoofMatl','MonthSold','YearSold'],drop_first=True)
new_df.columns
new_df.shape
new_df.head()
# forming the matrix of features X
X=new_df[[i for i in list(new_df.columns) if i != 'SalePrice']]
X.shape
y=df['SalePrice']
# splitting the data into the training and the test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Length of X_train: ' + str(len(X_train)))
print('Length of y_train: ' + str(len(y_train)))
print
print('Length of X_test: ' + str(len(X_test)))
print('Length of y_test: ' + str(len(y_test)))
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
plt.scatter(y_pred, y_test)
plt.plot([0,600000], [0, 600000])
plt.title('Linear Regression: Predicted vs True Prices')
plt.xlabel('Predicted')
plt.ylabel('True')
from sklearn.model_selection import cross_val_score
cv = cross_val_score(regressor, X_train, y_train)
cv
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
r2