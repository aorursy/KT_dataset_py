import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from keras.layers import Activation,Dropout,Dense

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import time

from sklearn.preprocessing import StandardScaler,MinMaxScaler

import seaborn as sns

from keras.models import Sequential

from sklearn.metrics import mean_squared_error

import math

from keras.callbacks import EarlyStopping
df=pd.read_csv('../input/train.csv',delimiter=',')

df.head(5)

test=pd.read_csv('../input/test.csv',delimiter=',')

df.isnull().all()
df.columns
df.corr()
fig, ax = plt.subplots(figsize=(25,25))

sns.heatmap(df.corr(),annot=True, linewidths=.5,ax=ax)
df.shape

# Function to transform housing data

def transform_house(data):

    # List of categorical, non-numeric variables

    dummy_list = ['MSSubClass', # though numeric in original data, it is categorical

                  'MSZoning', 

                  'Street', 

                  'Alley', 

                  'LotShape', 

                  'LandContour', 

                  'Utilities',

                  'LotConfig',

                  'LandSlope',

                  'Neighborhood',

                  'Condition1',

                  'Condition2',

                  'BldgType',

                  'HouseStyle',

                  'RoofStyle',

                  'RoofMatl',

                  'Exterior1st',

                  'Exterior2nd',

                  'MasVnrType', # Must be used if we use MasVnrArea

                  'ExterQual',

                  'ExterCond',

                  'Foundation',

                  'BsmtQual',

                  'BsmtCond',

                  'BsmtExposure',

                  'BsmtFinType1', # Must be used if we use BsmtFinSF1

                  'BsmtFinType2', # Must be used if we use BsmtFinSF2

                  'Heating',

                  'HeatingQC',

                  'CentralAir',

                  'Electrical',

                  'KitchenQual',

                  'Functional',

                  'FireplaceQu',

                  'GarageType',

                  'GarageFinish',

                  'GarageQual',

                  'GarageCond',

                  'PavedDrive',

                  'PoolQC',

                  'Fence',

                  'MiscFeature',

                  'SaleType',

                  'SaleCondition',

                  'MoSold',

                  #'YrSold', we think we should keep year sold as numeric not categorical

                 ]

    

    # create dummy variables

    for var in dummy_list:

        data = pd.concat([data, pd.get_dummies(data[var], drop_first=True, prefix=var)], axis=1)



    # drop dummy tables

    data = data.drop(dummy_list, axis=1)

    

    # fill nan with 0

    data = data.fillna(0)

        

    return data.copy()
label=df['SalePrice']

df=df.drop(['SalePrice'],axis=1)

label.shape
df=pd.concat([df,test])

df.shape
df.head()
train = transform_house(df)

train['TotalSquareFootage'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
test=train.iloc[1460:,:]

test.shape

train=train.iloc[:1460,:]
sc=StandardScaler()

sc.fit_transform(train)
print(test.shape,train.shape)
X_train,X_test,Y_train,Y_test=train_test_split(train,label,test_size=0.1)

X_train.shape


model=Sequential()

model.add(Dense(270,input_dim=271))

model.add(Dropout(0.25))

model.add(Dense(120))

model.add(Dropout(0.2))

model.add(Dense(50))

model.add(Dropout(0.1))

model.add(Dense(10))

model.add(Dense(output_dim=1,activation='linear'))

model.compile(loss='mean_squared_logarithmic_error', optimizer='Adam')

model.summary()

print(X_train.shape)

model.fit(X_train,Y_train,validation_split=0.3,epochs=2000,batch_size=20,verbose=1)



trainPredict = model.predict(X_train)

testPredict = model.predict(X_test)

trainPredict=np.squeeze(trainPredict,axis=1)

testPredict=np.squeeze(testPredict,axis=1)





trainScore=0

testScore=0

for i in range(len(Y_train)):

    Y_train.iloc[i]=math.log(Y_train.iloc[i])

for i in range(len(Y_test)):   

    Y_test.iloc[i]=math.log(Y_test.iloc[i])

for i in range(len(testPredict)):

    testPredict[i]=math.log(testPredict[i])

for i in range(len(trainPredict)):

    trainPredict[i]=math.log(trainPredict[i])



for i in range(1314):

    trainScore = trainScore+ (((Y_train.iloc[i]-trainPredict[i]))*((Y_train.iloc[i]-trainPredict[i])))

trainScore=trainScore/1314

trainScore=math.sqrt(trainScore)

print('Train Score: %.2f log over RMSE' % (trainScore))

for i in range(len(testPredict)):                                                     

    testScore += (((Y_test.iloc[i]-testPredict[i]))*((Y_test.iloc[i]-testPredict[i])))

testScore=testScore/146

testScore=math.sqrt(testScore)

print('Test Score: %.2f log over RMSE' % (testScore))
c=model.predict(test)

c=np.squeeze(c,axis=1)

d=np.arange(0,1459,1)+1461

#print(d.shape,c.shape)

ind=np.arange(0,1459,1)

g=pd.DataFrame({'Id':d,'SalePrice':c},index=ind)

g.to_csv("sales_price.csv",index=False)


