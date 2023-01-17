# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import sklearn
df_org = pd.read_csv('/kaggle/input/melbourne-housing-market/Melbourne_housing_FULL.csv')
df = pd.read_csv('/kaggle/input/melbourne-housing-market/MELBOURNE_HOUSE_PRICES_LESS.csv')
df_org.head(3)
df.head(3)
df_org.columns
extra_col = []
for col in df_org.columns:
    if col not in df.columns:
        extra_col.append(col)
print(extra_col)
print(df_org.shape)
print(df_org.dtypes)

df_org.info()
df_org.describe().T
print(df_org.select_dtypes(['object']).columns)
# convert objects to categorical variables
obj_cats = ['Suburb', 'Address', 'Type', 'Method', 'SellerG', 'CouncilArea',
       'Regionname']
for col in obj_cats:
    df_org[col] = df_org[col].astype('category')
# convert date from object to date format
# after converting the date to category, it would not change in the datetime
df_org['Date'] = pd.to_datetime(df_org['Date'])
# converting postcode 'numeric variable' to categorical
df_org['Postcode'] = df_org['Postcode'].astype('category')
# examine rooms vs bedroom2
df_org['Room v Bedroom2'] = df_org['Rooms'] - df_org['Bedroom2']
df_org
df_org['Room v Bedroom2'].value_counts()
# Drop columns
df_org = df_org.drop(['Room v Bedroom2', 'Bedroom2'], axis = 1)
# Add age variable
df_org['Age'] = 2017 - df_org['YearBuilt']

# identify historic homes
df_org['Historic'] = np.where(df_org['Age'] >= 50, 'Historic', 'Contemporary')

#convert to category
df_org['Historic'] = df_org['Historic'].astype('category')
df_org.Historic.value_counts()
# Visualize the missing values

fig, ax = plt.subplots(figsize = (15,7))
sns.set(font_scale = 1.2)
sns.heatmap(df_org.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')

plt.show()
# Count of missing value
df_org.isnull().sum()
# percentage of missing values
df_org.isnull().sum()/len(df_org)*100
print(df_org.shape)
# to remove rows missing data in a specific column
# as the yearbuilt column has some missing value hence it would miscalculate the historic col
df_org = df_org[pd.notnull(df_org['YearBuilt'])]
print(df_org.shape)
# drop all rows having null vlaues
df_org = df_org.dropna()
print(df_org.shape)
plt.figure(figsize = (12, 7))
sns.heatmap(df_org.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
df_org.describe().T
df_org[df_org['Age']>800]
df_org[df_org['BuildingArea'] == 0]
df_org[df_org['Landsize'] == 0]
# Confirm removal
df_org[df_org['BuildingArea'] == 0]
df_org.describe().T
plt.figure(figsize=(12,5))
sns.distplot(df_org['Price'], kde = False)
#columns having null values
#null_col =df_org.columns[df_org.isnull().any()].tolist()
#null_col
#categorical features
df_org.select_dtypes(['category']).columns
df_org['Regionname'].value_counts()
# Abbreviate Regionname categories
df_org['Regionname'] = df_org['Regionname'].map({'Southern Metropolitan' : 'S Metro',
                                                'Northern Metropolitan' : 'N Metro',
                                                'Western Metropolitan': 'W Metro', 
                                                'Eastern Metropolitan': 'E Metro', 
                                                'South-Eastern Metropolitan': 'SE Metro',
                                                'Northern Victoria': 'N Vic',
                                                'Eastern Victoria': 'E Vic',
                                                'Western Victoria' : 'W Vic'})
df_org['Regionname'].value_counts()
# Subplot of categorical features vs price
sns.set_style('whitegrid')
f, ax = plt.subplots(2,2, figsize = (15,15))

# plot [0,0]
sns.boxplot(data = df_org, x = 'Type', y = 'Price', ax= ax[0,0])
ax[0,0].set_xlabel('Type')
ax[0,0].set_ylabel('Price')
ax[0,0].set_title('Type vs Price')

# Plot[0,1]
sns.boxplot(data = df_org, x = 'Method', y= 'Price', ax = ax[0,1])
ax[0,1].set_xlabel('Method')
ax[0,1].set_title('Method vs Price')

# Plot[1,0]
sns.boxplot(data = df_org, x = 'Regionname', y= 'Price', ax = ax[1,0])
ax[1,0].set_xlabel('Regionname')
ax[1,0].set_title('Region Name vs Price')

# Plot[1,1]
sns.boxplot(data = df_org, x = 'Historic', y= 'Price', ax = ax[1,1])
ax[1,1].set_xlabel('Historic')
ax[1,1].set_title('Historic vs Price')

plt.show()
# Identify numeric features
df_org.select_dtypes(['float64', 'int64']).columns
# subplots of numeric features vs price
sns.set_style('whitegrid')
fig, ax = plt.subplots(4,2, figsize =(30,40))

#plot[0,0]
ax[0,0].scatter(x = 'Rooms', y = 'Price', data = df_org, edgecolor = 'b')
ax[0,0].set_xlabel('Rooms')
ax[0,0].set_ylabel('Price')
ax[0,0].set_title('Rooms vs Price')

#Plot [0,1]
ax[0,1].scatter(x = 'Distance', y = 'Price', data = df_org, edgecolor = 'b')
ax[0,1].set_xlabel('Distance')
ax[0,1].set_ylabel('Price')
ax[0,1].set_title('Distance vs Price')

#Plot [1,0]
ax[1,0].scatter(x = 'Bathroom', y = 'Price', data = df_org, edgecolor = 'b')
ax[1,0].set_xlabel('Bathroom')
ax[1,0].set_ylabel('Price')
ax[1,0].set_title('Bathroom vs Price')

#Plot [1,1]
ax[1,1].scatter(x = 'Car', y = 'Price', data = df_org, edgecolor = 'b')
ax[1,1].set_xlabel('Car')
ax[1,1].set_ylabel('Price')
ax[1,1].set_title('Car vs Price')

#Plot [2,0]
ax[2,0].scatter(x = 'Landsize', y = 'Price', data = df_org, edgecolor = 'b')
ax[2,0].set_xlabel('Landsize')
ax[2,0].set_ylabel('Price')
ax[2,0].set_title('Landsize vs Price')

#Plot [2,1]
ax[2,1].scatter(x = 'BuildingArea', y = 'Price', data = df_org, edgecolor = 'b')
ax[2,1].set_xlabel('BuildingArea')
ax[2,1].set_ylabel('Price')
ax[2,1].set_title('BuildingArea vs Price')

#Plot [3,0]
ax[3,0].scatter(x = 'Age', y = 'Price', data = df_org, edgecolor = 'b')
ax[3,0].set_xlabel('Age')
ax[3,0].set_ylabel('Price')
ax[3,0].set_title('Age vs Price')

#Plot [3,1]
ax[3,1].scatter(x = 'Propertycount', y = 'Price', data = df_org, edgecolor = 'b')
ax[3,1].set_xlabel('Propertycount')
ax[3,1].set_ylabel('Price')
ax[3,1].set_title('Propertycount vs Price')

plt.show()

plt.figure(figsize=(12,7))
sns.boxplot(x= 'Rooms', y = 'Price', data = df_org)
plt.figure(figsize = (10,6))
sns.heatmap(df_org.corr(), cmap = 'coolwarm', linewidth = 1, annot = True, annot_kws = {'size':9})
plt.title('Variable Correlation')
# Identify numeric features
df_org.select_dtypes(['float64', 'int64']).columns
# Split test and train
X = df_org[['Rooms', 'Distance', 'Bathroom', 'Car', 'Landsize',
       'BuildingArea', 'Propertycount','Age']]
y = df_org['Price']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .20, random_state= 0)
# model fitting and prediction
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
#predictions
y_pred = lm.predict(X_test)
from sklearn import metrics
print('MAE: ', metrics.mean_absolute_error(y_test, y_pred))
print('MSE: ', metrics.mean_squared_error(y_test, y_pred))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# Calulate R squared
print('R^2 =', metrics.explained_variance_score(y_test, y_pred))
# Actual vs predictions scatter
plt.scatter(y_test, y_pred)
# Histogram of the distribution of residuals
sns.distplot((y_test - y_pred))
cdf = pd.DataFrame(data = lm.coef_, index = X.columns, columns = ['Coefficients'])
cdf