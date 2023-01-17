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
house=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
house.info()
x=house.drop(["LotFrontage","Alley","MasVnrType","MasVnrArea","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","Electrical","FireplaceQu","GarageType","GarageYrBlt","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature"],axis=1)
x.info()
cat_col = x.select_dtypes(include=['object']).columns

dummies = pd.get_dummies(x[cat_col],drop_first=True)

without_dummies = x.drop(cat_col,axis=1)
data = pd.concat([dummies,without_dummies],axis=1)
data.info()
y=data["SalePrice"]

x=data.drop(["SalePrice"],axis=1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,

                                  test_size=0.3,random_state=1)

input_dim = len(data.columns) - 1

from keras.models import Sequential

from keras.layers import Dense

from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder

from keras.utils.np_utils import to_categorical

from sklearn.utils import shuffle

model = Sequential()

model.add(Dense(190, input_dim = input_dim , activation = 'relu'))

model.add(Dense(10000, activation = 'relu'))

model.add(Dense(10000, activation = 'relu'))

model.add(Dense(10000, activation = 'relu'))

model.add(Dense(1, activation = 'linear'))

model.compile(loss = 'mean_absolute_error' , optimizer = 'adam' , metrics = ['mean_absolute_error'] )

model.fit(x_train, y_train, epochs = 10, batch_size = 2)

scores = model.evaluate(x_test, y_test)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

print(test.columns)
x=test.drop(["LotFrontage","Alley","MasVnrType","MasVnrArea","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","Electrical","FireplaceQu","GarageType","GarageYrBlt","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature"],axis=1)
cat_col = x.select_dtypes(include=['object']).columns

dummies = pd.get_dummies(x[cat_col],drop_first=True)

without_dummies = x.drop(cat_col,axis=1)
testdata = pd.concat([dummies,without_dummies],axis=1)

data=data.drop(["SalePrice"],axis=1)

train_columns = data.columns

test_columns  = testdata.columns

print(len(train_columns))

print(len(test_columns))

for traincolumn in train_columns:

    if traincolumn not in test_columns:

        testdata[str(traincolumn)]=0

testdata['Id']
testdata['SalePrice']=model.predict(testdata)

print(testdata.columns)

submit=testdata.loc[:,['Id','SalePrice']]

ind=pd.isnull(submit['SalePrice'])

submit.loc[ind,'SalePrice']=np.mean(submit['SalePrice'])

submit.to_csv("mysubmission.csv",index=False)