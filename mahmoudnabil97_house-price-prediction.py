import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.style as style

import seaborn as sns

import scipy as stats
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.head()
train.describe().T
train.info()
corr= train.corr()['SalePrice'].sort_values(ascending=False)

corr
def missing_val(df):

  miss=df.isnull().sum().sort_values(ascending=False) 

  total_miss = miss[miss != 0]

  percent = round(total_miss / len(df)*100,2)

  return pd.concat((total_miss , percent) , axis=1 , keys=['Total Miss' , 'Percentage'])
missing_val(train)
missing_val(test)
figure = plt.figure(figsize=(15,8))

plt.subplot(1,3,1)

sns.distplot(train['SalePrice'])



from scipy import stats

plt.subplot(1,3,2)

stats.probplot(train['SalePrice'] , plot=plt)



plt.subplot(1,3,3)

sns.boxplot(train['SalePrice'] ,orient='v')
print('Skewness of Saleprice is :-   ' , train['SalePrice'].skew()) 

print('kurtoises of Saleprice is :   ' , train['SalePrice'].kurt())
train['SalePrice'] = np.log1p(train['SalePrice'])



print('Skewness of Saleprice is :-   ' , train['SalePrice'].skew()) 

print('kurtoises of Saleprice is :   ' , train['SalePrice'].kurt())
fg = plt.figure(figsize=(15,8))

plt.subplot(1,3,1)

sns.distplot(train['SalePrice'])





plt.subplot(1,3,2)

stats.probplot(train['SalePrice'] , plot=plt)



plt.subplot(1,3,3)

sns.boxplot(train['SalePrice'], orient='v')
fg = plt.figure(figsize=(18,12))

sns.heatmap(train.corr())
# EXplore correlation between features

corr = train.corr()['SalePrice'].sort_values(ascending =False)[:10]

print(corr)
st_corr=['OverallQual' , 'GrLivArea' ,'GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','YearBuilt','YearRemodAdd']

print(st_corr)

for i in st_corr:

  plt.subplots(figsize=(12,8))

  sns.scatterplot(x = train[i] , y=train['SalePrice'])

#sns.scatterplot(x = train['OverallQual'] , y=train['SalePrice'])
numeric_data = train.select_dtypes(include = np.number).drop(['SalePrice'] , axis =1)

items = numeric_data.loc[ : , ['OverallQual','GrLivArea','GarageCars','TotalBsmtSF',

                              'FullBath','YearBuilt','YearRemodAdd'  ]]
 #visualize these itemes using Boxplot

fig = plt.figure(figsize=(12,12))

for col in range(len(items.columns)) : 

    fig.add_subplot(3 , 3 , col+1)

    sns.boxplot(y=items.iloc[: , col])

plt.show()



    # Visualize these items using multivariate analysis (SCatter plot)   

fig = plt.figure(figsize=(16,12))

for col in range(len(items.columns)):

    fig.add_subplot(3,3,col+1)

    sns.scatterplot(items.iloc[ : , col] , train['SalePrice'])

plt.show()
  # Using Z-Score to identify outliers

from scipy import stats

z= np.abs(stats.zscore(items))

print(z)

threshold = 4

print(np.where(z > threshold))

    # Remove outlier using z-score

train.shape

train = train[(z < threshold).all(axis=1)]

train.shape
train.corr()['SalePrice'].sort_values(ascending=False)[:10]    # note that correlation of thesee features still strong
dataset = pd.concat((train,test) , sort = False).reset_index(drop=True)

dataset = dataset.drop(columns =['SalePrice'] , axis =1 )
missing_val(dataset)
missing_val(dataset)
dataset['totalSF'] =( dataset['TotalBsmtSF'] + dataset['1stFlrSF'] + dataset['2ndFlrSF']  )

dataset['total_bathrooms'] = (dataset['BsmtFullBath'] + 0.5*dataset['BsmtHalfBath'] + dataset['FullBath'] + 0.5*dataset['HalfBath'])

dataset['ageHouse'] = (dataset['YrSold'] - dataset['YearBuilt'] )

dataset.drop(['Id','Utilities','PoolQC','MiscFeature','Alley','Fence','GarageYrBlt'] , axis=1 , inplace=True)

dataset.drop(['TotalBsmtSF' , '1stFlrSF' ,'2ndFlrSF'] , axis = 1 , inplace =True)

dataset.drop(['BsmtFullBath' , 'BsmtHalfBath' , 'FullBath' , 'HalfBath'] , axis=1 , inplace=True)

miss_mode =  ['MasVnrArea' , 'Electrical' , 'MSZoning' , 'SaleType','Exterior1st','Exterior2nd','KitchenQual']

for col in miss_mode:

    dataset[col]  = dataset[col].fillna(dataset[col].mode()[0])

    

missing_feat = ['GarageType','GarageCond','GarageQual','GarageFinish',

                'BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','BsmtQual',

                'FireplaceQu','MasVnrType']

for col in missing_feat:

    dataset[col]=dataset[col].fillna('None')



dataset['Functional'] = dataset['Functional'].fillna('Typ')

dataset['LotFrontage'] = dataset['LotFrontage'].fillna(dataset['LotFrontage'].median())



miss_zero = ['total_bathrooms','totalSF','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','GarageArea','GarageCars' ]

for col in miss_zero:

    dataset[col] = dataset[col].fillna(0)





#check on misssing values

missing_val(dataset)
dataset.shape
dataset.dropna(inplace=True)
dataset.shape
missing_val(dataset)
dataset['totalSF'].describe().T
dataset['ageHouse'].describe().T
neg_value = dataset[dataset['ageHouse'] < 0 ]

neg_value
dataset.loc[dataset['YrSold'] < dataset['YearBuilt'], 'YrSold'] = 2009

dataset['ageHouse'] = (dataset['YrSold'] - dataset['YearBuilt'] )

dataset['ageHouse'].describe()
dataset['MSSubClass']   = dataset['MSSubClass'].astype(str)

#check for duplicate rows 

duplicate= train[train.duplicated()]

print(duplicate) # there is no duplicate rows

dataset.shape
final_features = pd.get_dummies(dataset).reset_index(drop=True)

print(final_features.shape)

final_features.head()
final_features =final_features.loc[:,~final_features.columns.duplicated()]
final_features.shape
y= train['SalePrice']

X = final_features.iloc[: len(y) , :]

df_test  = final_features.iloc[len(y): , :]
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

sc_y = StandardScaler()

X = sc_x.fit_transform(X)

y = sc_y.fit_transform(np.array(y).reshape(-1,1))
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X , y)

y_pred = lr.predict(df_test)

print(y_pred)
from sklearn.model_selection import KFold , cross_val_score

#lr = LinearRegression()

cv = KFold(shuffle= True , random_state=2 , n_splits=10)

scores = cross_val_score(lr , X , y , cv =cv ,scoring = 'neg_mean_absolute_error' )

print(scores.mean())
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_absolute_error, mean_squared_error

ridge = Ridge(alpha = 400)

ridge.fit(X , y)

test_pred = ridge.predict(df_test)

print(test_pred)
import pickle

filename = 'Ridge_model.pkl'

pickle.dump(ridge , open(filename , 'wb') )
## create simple submission file 

pred = pd.DataFrame(test_pred)

sample_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

final_data = pd.concat([sample_df['Id'] , pred] , axis=1)

final_data.columns=['Id' , 'SalePrice']

final_data.to_csv('Ridge_model.csv' , index=False)
from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error , mean_absolute_error

lasso = Lasso(alpha = 0.001 )

lasso.fit(X , y)

test_pred = lasso.predict(df_test)

print(test_pred)
#save model 

import pickle

filename = 'Lasso_model.pkl'

pickle.dump(lasso , open(filename , 'wb') )
## create simple submission file 

pred = pd.DataFrame(test_pred)

sample_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

final_data = pd.concat([sample_df['Id'] , pred] , axis=1)

final_data.columns=['Id' , 'SalePrice']

final_data.to_csv('Lasso_model.csv' , index=False)
from sklearn.linear_model import ElasticNet

elastic = ElasticNet(alpha =0.0001 ,normalize= True)

elastic.fit(X , y)

test_pred = elastic.predict(df_test)

print(test_pred)
#save model 

import pickle

filename = 'Lasso_model.pkl'

pickle.dump(elastic , open(filename , 'wb') )
## create simple submission file 

pred = pd.DataFrame(test_pred)

sample_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

final_data = pd.concat([sample_df['Id'] , pred] , axis=1)

final_data.columns=['Id' , 'SalePrice']

final_data.to_csv('Elastic_model.csv' , index=False)
X.shape
from keras import backend as K

def root_mean_squared_error(y_true, y_pred):

        return K.sqrt(K.mean(K.square(y_pred - y_true)))
import keras

from keras.models import Sequential 

from keras.layers import Dense

from keras.layers import LeakyReLU,PReLU,ELU

from keras.layers import Dropout



# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 100, kernel_initializer='he_uniform',activation='relu',input_dim = 292))



# Adding the second hidden layer

classifier.add(Dense(units = 50, kernel_initializer = 'he_uniform',activation='relu'))



# Adding the third hidden layer

classifier.add(Dense(units = 25, kernel_initializer = 'he_uniform',activation='relu'))



# Adding the forth hidden layer

classifier.add(Dense(units = 50, kernel_initializer = 'he_uniform',activation='relu'))



# Adding the fifth hidden layer

classifier.add(Dense(units = 25, kernel_initializer = 'he_uniform',activation='relu'))



# Adding the output layer

classifier.add(Dense(units = 1, kernel_initializer = 'he_uniform'))



# Compiling the ANN

classifier.compile(loss=root_mean_squared_error, optimizer='Adamax')



# Fitting the ANN to the Training set

model_history=classifier.fit(X, y,validation_split=0.20,epochs=1000, batch_size = 10)





test_pred = classifier.predict(df_test)

print(test_pred)


## create simple submission file 

pred = pd.DataFrame(test_pred)

sample_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

final_data = pd.concat([sample_df['Id'] , pred] , axis=1)

final_data.columns=['Id' , 'SalePrice']

final_data.to_csv('ANN_model.csv' , index=False)



from sklearn.svm import SVR

svr  = SVR(kernel = 'linear')

svr.fit(X , y )

y_pred = svr.predict(df_test)

print(y_pred)



## create simple submission file 

pred = pd.DataFrame(test_pred)

sample_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

final_data = pd.concat([sample_df['Id'] , pred] , axis=1)

final_data.columns=['Id' , 'SalePrice']

final_data.to_csv('SVR_model.csv' , index=False)
