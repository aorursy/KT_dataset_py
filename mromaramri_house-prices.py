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







# Writtin by Omar 



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
# DO NOT TOUCH

# Writtin by Omar 

# load train dataset



train_data = pd.read_csv('../input/train.csv')
# DO NOT TOUCH

# Writtin by Omar 

# load test dataset



test_data = pd.read_csv('../input/test.csv')
# DO NOT TOUCH

# Writtin by Omar 

# Create dataframe for train dataset



train = pd.DataFrame(train_data)
# DO NOT TOUCH

# Writtin by Omar 

# Create dataframe for test dataset



test = pd.DataFrame(test_data)
# Writtin by Omar 

# check shape 



train.shape
# Writtin by Omar 

# chack shape



test.shape
# DO NOT TOUCH

# Writtin by Omar 

# concat -> Join the two dataframes togather 



df = pd.concat([train, test])
# Writtin by Omar 

# View dataframe



df.head()
# Writtin by Omar 

# chack shape



df.shape
# Writtin by Omar 

# chack tail to see NaN valuse at SalePrice (targat) 

# To chack if the test_data at the end of the new dataframe 



df.tail()
# Writtin by Omar 

# These are the most important faetures that we have choosn beased on our domain knewledge 



# ['OverallQual', 'GrLivArea', 'GarageCars',

#                     'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd',

#                     'YearBuilt', 'YearRemodAdd',

#                     'Fireplaces', 'BsmtFinSF1', 'SalePrice']
# checking the data types of our favoret features, you can r use it at any stage of your code, copy and paste.



print('OverallQual is type: ', df.OverallQual.dtypes)

print('GrLivArea is type: ', df.GrLivArea.dtypes)

print('GarageCars is type: ', df.GarageCars.dtypes) # couldn't be converted to int because of the nan in the test data,

# it's all int exept the nan, and if it wasn't, we can change the datatype 'in both dataframes'after splitting.

print('TotalBsmtSF is type: ', df.TotalBsmtSF.dtypes)

print('1stFlrSF is type: ', df['1stFlrSF'].dtypes)

print('FullBath is type: ', df.FullBath.dtypes)

print('TotRmsAbvGrd is type: ', df.TotRmsAbvGrd.dtypes)

print('Fireplaces is type: ', df.Fireplaces.dtypes)

print('BsmtFinSF1 is type: ', df.BsmtFinSF1.dtypes)

print('SalePrice is type: ', df.SalePrice.dtypes)
df[df.TotalBsmtSF == 0.0].count()
# checking the nan values in our 11 features that is suppose to be int and is not int.

df[df.TotalBsmtSF.isna()]

# this nan is in the test data we will deal with it after spliting the 2 dataframes
df[df['BsmtFinSF1'].isna()]

# one nan in the test data set, we will deal with it after spliting the 2 dataframes
# DO NOT TOUCH

# splited the train data

train_splited = df[df.SalePrice.notnull()]
# sns.pairplot(train[['SalePrice']])

# DO NOT TOUCH

# splited the test data, Now, any modefication you do in one, do on the other.

test_splited = df[df.SalePrice.isna()].copy()
#A7

# all modeling above is a trail, we will apply it to 'train_splited' after cleaning it.

# let's check the data types in train_splited & test_splited



print('train_splited OverallQual is type: ', train_splited.OverallQual.dtypes)

print('train_splited GrLivArea is type: ', train_splited.GrLivArea.dtypes)

print('train_splited GarageCars is type: ', train_splited.GarageCars.dtypes)

print('train_splited TotalBsmtSF is type: ', train_splited.TotalBsmtSF.dtypes)

print('train_splited 1stFlrSF is type: ', train_splited['1stFlrSF'].dtypes)

print('train_splited FullBath is type: ', train_splited.FullBath.dtypes)

print('train_splited TotRmsAbvGrd is type: ', train_splited.TotRmsAbvGrd.dtypes)

print('train_splited Fireplaces is type: ', train_splited.Fireplaces.dtypes)

print('train_splited BsmtFinSF1 is type: ', train_splited.BsmtFinSF1.dtypes)

print('train_splited SalePrice is type: ', train_splited.SalePrice.dtypes)
# DO NOT TOUCH

# let's fix the data type in GarageCars in both train_splited & test_splited.





train_splited[train_splited.GarageCars.isna()] #no NaNs, let's convert to int.

train_splited = train_splited.astype({"GarageCars": int})

train_splited.GarageCars.dtypes #Done
print('test_splited OverallQual is type: ', test_splited.OverallQual.dtypes)

print('test_splited GrLivArea is type: ', test_splited.GrLivArea.dtypes)

print('test_splited GarageCars is type: ', test_splited.GarageCars.dtypes)

print('test_splited TotalBsmtSF is type: ', test_splited.TotalBsmtSF.dtypes)

print('test_splited 1stFlrSF is type: ', test_splited['1stFlrSF'].dtypes)

print('test_splited FullBath is type: ', test_splited.FullBath.dtypes)

print('test_splited TotRmsAbvGrd is type: ', test_splited.TotRmsAbvGrd.dtypes)

print('test_splited Fireplaces is type: ', test_splited.Fireplaces.dtypes)

print('test_splited BsmtFinSF1 is type: ', test_splited.BsmtFinSF1.dtypes)

print('test_splited SalePrice is type: ', test_splited.SalePrice.dtypes)
# DO NOT TOUCH

test_splited[test_splited.GarageCars.isna()] #one NaN.

test_splited['GarageCars'].fillna(0, inplace=True) #replicing the only NaN with Zero.

test_splited[test_splited.GarageCars.isna()] ##no NaNs, let's convert to int.

test_splited = test_splited.astype({"GarageCars": int})

test_splited.GarageCars.dtypes #Done
# DO NOT TOUCH

# let's fix the data type in TotalBsmtSF in both train_splited & test_splited.

train_splited[train_splited.TotalBsmtSF.isna()] #no NaNs, let's convert to int.

train_splited = train_splited.astype({"TotalBsmtSF": int})

train_splited.TotalBsmtSF.dtypes #Done
# DO NOT TOUCH

test_splited[test_splited.TotalBsmtSF.isna()] #one NaN.

test_splited['TotalBsmtSF'].fillna(0, inplace=True) #replicing the only NaN with Zero.

test_splited[test_splited.TotalBsmtSF.isna()] #no NaNs, let's convert to int.

test_splited = test_splited.astype({"TotalBsmtSF": int})

test_splited.TotalBsmtSF.dtypes #Done
# DO NOT TOUCH

# let's fix the data type in BsmtFinSF1 in both train_splited & test_splited.



train_splited[train_splited.BsmtFinSF1.isna()] #no NaNs, let's convert to int.

train_splited = train_splited.astype({"BsmtFinSF1": int})

train_splited.BsmtFinSF1.dtypes #Done
# DO NOT TOUCH

test_splited[test_splited.BsmtFinSF1.isna()] #one NaN.

test_splited['BsmtFinSF1'].fillna(0, inplace=True) #replicing the only NaN with Zero.

test_splited[test_splited.BsmtFinSF1.isna()] #no NaNs, let's convert to int.

test_splited = test_splited.astype({"BsmtFinSF1": int})

test_splited.BsmtFinSF1.dtypes #Done
# DO NOT TOUCH

# in SalePrice all prices are integers, no need for the float.

train_splited[train_splited.SalePrice.isna()] #no NaNs, let's convert to int.

train_splited = train_splited.astype({"SalePrice": int})

train_splited.SalePrice.dtypes #Done
# Now let's check our data type after fixing them.

print('train_splited OverallQual is type: ', train_splited.OverallQual.dtypes)

print('train_splited GrLivArea is type: ', train_splited.GrLivArea.dtypes)

print('train_splited GarageCars is type: ', train_splited.GarageCars.dtypes)

print('train_splited TotalBsmtSF is type: ', train_splited.TotalBsmtSF.dtypes)

print('train_splited 1stFlrSF is type: ', train_splited['1stFlrSF'].dtypes)

print('train_splited FullBath is type: ', train_splited.FullBath.dtypes)

print('train_splited TotRmsAbvGrd is type: ', train_splited.TotRmsAbvGrd.dtypes)

print('train_splited Fireplaces is type: ', train_splited.Fireplaces.dtypes)

print('train_splited BsmtFinSF1 is type: ', train_splited.BsmtFinSF1.dtypes)

print('train_splited SalePrice is type: ', train_splited.SalePrice.dtypes)

# All int, No nans.
test_splited.SalePrice.unique() # All nans, great!
# DO NOT TOUCH

# in the test_splited we have unwanted colume SalePrice, let's drop it.

test_splited.drop(['SalePrice'], axis=1, inplace=True)

# SalePrice is gone from test_splited.
print('test_splited  OverallQual is type: ', test_splited.OverallQual.dtypes)

print('test_splited GrLivArea is type: ', test_splited.GrLivArea.dtypes)

print('test_splited GarageCars is type: ', test_splited.GarageCars.dtypes)

print('test_splited TotalBsmtSF is type: ', test_splited.TotalBsmtSF.dtypes)

print('test_splited 1stFlrSF is type: ', test_splited['1stFlrSF'].dtypes)

print('test_splited FullBath is type: ', test_splited.FullBath.dtypes)

print('test_splited TotRmsAbvGrd is type: ', test_splited.TotRmsAbvGrd.dtypes)

print('test_splited Fireplaces is type: ', test_splited.Fireplaces.dtypes)

print('test_splited BsmtFinSF1 is type: ', test_splited.BsmtFinSF1.dtypes)

# test_splited all type int.
# all data types are good in both dataframes, thursday we want to investigate the values of each column

# and if we are satisfied with the range and the values itself we will start building our actual model.
train= train_splited
#عبدالرحمن 

## Training the Model ##

# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd









X = train[['OverallQual', 'GrLivArea', 'GarageCars',

                     'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd',

                     'YearBuilt', 'YearRemodAdd',

                     'Fireplaces', 'BsmtFinSF1']]   # 11 features (columns) wree chosen



y = train.SalePrice

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)



# Feature Scaling

# from sklearn.preprocessing import StandardScaler

# sc_X = StandardScaler()

# X_train = sc_X.fit_transform(X_train)

# X_test = sc_X.transform(X_test)

# sc_y = StandardScaler()

# y_train = sc_y.fit_transform(y_train.reshape(-1,1))



#A2

## Feature Scaling ##

import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()



# X = train[['OverallQual', 'GrLivArea', 'GarageCars',

#                      'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd',

#                      'YearBuilt', 'YearRemodAdd',

#                      'Fireplaces', 'BsmtFinSF1']]   # 11 features (columns) wree chosen



X[['OverallQual', 'GrLivArea', 'GarageCars',

                     'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd',

                     'YearBuilt', 'YearRemodAdd',

                     'Fireplaces', 'BsmtFinSF1']] = scale.fit_transform(X[['OverallQual', 'GrLivArea', 'GarageCars',

                     'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd',

                     'YearBuilt', 'YearRemodAdd',

                     'Fireplaces', 'BsmtFinSF1']].as_matrix())

y = train.SalePrice



print (X)



est = sm.OLS(y, X).fit()



# est.summary()
#A3

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit (X_train, y_train)

#A4

lr.score(X_train, y_train)

print ('lr.score for train :', lr.score(X_train, y_train))



lr.score(X_test, y_test)

print ('lr.score for test :', lr.score(X_test, y_test))
#A5

coeff_df = pd.DataFrame(lr.coef_,X.columns,columns=['Coefficient'])

coeff_df
#O6

from sklearn.model_selection import KFold, cross_val_score , cross_val_predict

kf = KFold(n_splits=4, shuffle=True)



np.mean(cross_val_score(lr, X, y, cv=kf))

# cross_val_score(lr, X, y, cv=kf)

predictions = lr.predict(X_test)
#A8

## Evaluating the Model ##



from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
#A9

# Create a scatterplot of the real test values versus the predicted values:

plt.scatter(y_test,predictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
#A10

# Plot a histogram of the residuals

sns.distplot((y_test-predictions),bins=50);
# Now its time to apply the MVLR model on the testing data

# test.head()

test.shape
############

# Now we need to evaluate the performance of the model on the test data

test_splited.Id
## Predicting Test Data

from sklearn.linear_model import LinearRegression

X_testdata = test_splited[['OverallQual', 'GrLivArea', 'GarageCars',

                     'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd',

                     'YearBuilt', 'YearRemodAdd',

                     'Fireplaces', 'BsmtFinSF1']]

lr.predict(X_testdata)
predicted_Saleprice = lr.predict(X_testdata)

predicted_Saleprice


df_subm_test = test_splited[['Id']].copy()

df_subm_test['predicted_Saleprice'] = predicted_Saleprice
df_subm_test.head()
df_subm_test.to_csv('df_subm_test.csv', index=False)
from IPython.display import HTML

import base64



def create_download_link( df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = f'<a target="_blank">{title}</a>'

    return HTML(html)



create_download_link(df_subm_test, filename='df_subm_test')
plt.scatter(y_test,predictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
# log model enhancing 