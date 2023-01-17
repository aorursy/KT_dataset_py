#we need to some modeule for handling and visualizing result of the data 

import pandas as pd   #pandas for data manipulation and analysis

import numpy as np   # numpy used for handling numerical data 

import matplotlib.pyplot as plt  # this is used for visualizing the data

import statistics     # it is used for statistical functions

import seaborn as sns

%matplotlib inline

from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error
#import datafile using pandas module

train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

#test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train_data.head()
train_data.shape
train_data.info()        # here we can get that is there any null value and how much
plt.figure(figsize=(15,15))

sns.distplot(train_data['SalePrice'],bins=100) 
column_names = train_data.columns   # we need list of columns for preprocessing so we can generate here

print(column_names)
# there are lots of null values in some columns so we directly remove that column fron both of data files

train_data.drop(labels=['Alley','FireplaceQu','PoolQC','Fence','MiscFeature',], axis=1,inplace=True)

#test_data.drop(labels=['Alley','FireplaceQu','PoolQC','Fence','MiscFeature',], axis=1,inplace=True)

column_names = train_data.columns   # we need list of columns for preprocessing so we can generate here

print(column_names)
# Remaining null values we can replace by median and mode which is depnd on the column type



for c in range(1,len(column_names)-1):                                       # we create loop for change ing null values

    col = column_names[c]                                                    # in train and test data set

    if train_data[col].dtype == 'object':                                    # with the using of if else statement

                                                                             # objects replace by mode and other  

        train_data[col].fillna(train_data[col].dropna().max(),inplace =True) # replace by mean

        #test_data[col].fillna(test_data[col].dropna().max(),inplace =True)



    else:

        train_data[col] = train_data[col].fillna(train_data[col].mean())

        #test_data[col] = test_data[col].fillna(test_data[col].mean())

#here we can distribute columns by categorical and numerical

column_names = train_data.columns

categorical_col = []

numeric_col = []

for c in range(1,len(column_names)-1):                                       # we create loop for change ing null values

    col = column_names[c]                                                    # in train and test data set

    if train_data[col].dtype == 'object': 

        categorical_col.append(col)                                         # with the using of if else statement

                                                                                 # objects replace by mode and other  

    else:

        numeric_col.append(col)

print('categorical_col:\t', categorical_col,'\n')

print('numeric_col:\t',numeric_col )
plt.figure(figsize=(15,15))

sns.heatmap(train_data[numeric_col].corr(),cmap='coolwarm',annot=True)
plt.figure(figsize=(10,10))

sns.stripplot(x="SaleCondition", y="SalePrice", data=train_data,jitter=True,hue='SaleType',palette='Set1')
plt.figure(figsize=(15,15))

sns.countplot(x='GarageType',data=train_data)
'''

# Then you map to the grid

g = sns.PairGrid(train_data[numeric_col])

g.map(plt.scatter)

'''
g = sns.PairGrid(train_data[numeric_col[:10]])

g.map(plt.scatter)
#in most of the cases model not handle categorical data 

# in python there is best module to handle the data

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()


column = train_data.columns                        

for c in range(1,len(column)-1):                     

    col = column[c]

    if train_data[col].dtype == 'object':

        train_data[col] = le.fit_transform(train_data[col])

        #test_data[col] = le.fit_transform(test_data[col])

    else:

        pass
X = train_data.iloc[:,1:-1]

y = train_data.iloc[:,-1]

#test = test_data.iloc[:,1:].values
# We can split data to model evaluation

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.25, random_state = 100)   # here we can split into train and test 
from sklearn.ensemble import GradientBoostingRegressor

reg = GradientBoostingRegressor(n_estimators=2000,loss='ls')

reg.fit(X_train, y_train)

y_pred= reg.predict(X_test)

print("r2_score",r2_score(y_test,y_pred))

print('mean abs error', mean_absolute_error(y_test,y_pred))
plt.figure(figsize=(20,12))

plt.plot(list(y_test), color = 'red', label = 'Actual Price')

plt.plot(y_pred, color = 'green', label = 'predicted Price')

plt.title('GB Regression')

plt.ylim(0,1000000)

plt.xlabel('Id')

plt.ylabel('Price')

plt.legend()

plt.show()
