import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
#importing train and test datasets



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
#Removing SalePrice from train set to merge the two datasets together for EDA = train2



train2 = train.drop("SalePrice",axis=1)
train2.head()
#Adding another column to differentiate train and test datasets

train2['DataSet'] = 'Train'

test['DataSet'] = 'Test'
#Merging the 2 datasets together = eda



eda = [train2,test]

eda = pd.concat(eda)
eda.count().plot(kind = 'barh',figsize=(20,20))
#Verify NaN values



#There are a few variables that doesn't have almost any value in the dataset, therefore we will remove them

#because there's no possibility of them being imported to explain the SalePrice, removing PoolQC, MiscFeature,

#Alley, Fence, FirePlaceQu.



eda.count().sort_values()
eda2 = eda.drop(['PoolQC', 'MiscFeature','Alley', 'Fence', 'FireplaceQu'],axis=1)
eda2.describe()
print(len(eda.columns))

print(len(eda2.columns))
#Now we have to treat the rest of the dataframe, we still have NaN values in the other columns
eda2.columns[eda2.isna().any()].tolist()
#Now we have to understand the meaning of each variable to see if we replace the NaN values with 

#'None' 'Zero' or 'Mode'
eda2['MSZoning'] = eda2['MSZoning'].fillna(eda2['MSZoning'].mode()[0])

eda2['LotFrontage'] = eda2['LotFrontage'].fillna(0)

eda2['Utilities'] = eda2['Utilities'].fillna(eda2['Utilities'].mode()[0])

eda2['Exterior1st'] = eda2['Exterior1st'].fillna(eda2['Exterior1st'].mode()[0])

eda2['Exterior2nd'] = eda2['Exterior2nd'].fillna('None')

eda2['MasVnrType'] = eda2['MasVnrType'].fillna('None')

eda2['MasVnrArea'] = eda2['MasVnrArea'].fillna(0)

eda2['BsmtQual'] = eda2['BsmtQual'].fillna('None')

eda2['BsmtCond'] = eda2['BsmtCond'].fillna('None')

eda2['BsmtExposure'] = eda2['BsmtExposure'].fillna('None')

eda2['BsmtFinType1'] = eda2['BsmtFinType1'].fillna('None')

eda2['BsmtFinSF1'] = eda2['BsmtFinSF1'].fillna(0)

eda2['BsmtFinType2'] = eda2['BsmtFinType2'].fillna('None')

eda2['BsmtFinSF2'] = eda2['BsmtFinSF2'].fillna(0)

eda2['BsmtUnfSF'] = eda2['BsmtUnfSF'].fillna(0)

eda2['TotalBsmtSF'] = eda2['TotalBsmtSF'].fillna(0)

eda2['Electrical'] = eda2['Electrical'].fillna(eda2['Electrical'].mode()[0])

eda2['BsmtFullBath'] = eda2['BsmtFullBath'].fillna(0)

eda2['BsmtHalfBath'] = eda2['BsmtHalfBath'].fillna(0)

eda2['KitchenQual'] = eda2['KitchenQual'].fillna(eda2['KitchenQual'].mode()[0])

eda2['Functional'] = eda2['Functional'].fillna(eda2['Functional'].mode()[0])

eda2['GarageType'] = eda2['GarageType'].fillna('None')

eda2['GarageYrBlt'] = eda2['GarageYrBlt'].fillna('None')

eda2['GarageFinish'] = eda2['GarageFinish'].fillna('None')

eda2['GarageCars'] = eda2['GarageCars'].fillna(0)

eda2['GarageArea'] = eda2['GarageArea'].fillna(0)

eda2['GarageQual'] = eda2['GarageQual'].fillna('None')

eda2['GarageCond'] = eda2['GarageCond'].fillna('None')

eda2['SaleType'] = eda2['SaleType'].fillna(eda2['SaleType'].mode()[0])





#Checking if all NaN values were filled

eda2.columns[eda2.isna().any()].tolist()
#Transforming categorical variables into numeric for model usage = eda3



eda3 = pd.get_dummies(eda2)
eda3.head()
#Now we have to separate train and test sets again for model training



train2 = eda3['DataSet_Train']==1

test2 = eda3['DataSet_Test']==1

train3 = eda3[train2]

test3 = eda3[test2]

train3 = train3.drop(['DataSet_Train','DataSet_Test'],axis=1)

test3 = test3.drop(['DataSet_Train','DataSet_Test'],axis=1)
from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
#selecting the variables to train model



x = train3

y = train['SalePrice']





#Since the Sale Price is too high we are going to scale it to model.

y2 =scaler.fit_transform(np.array(y).reshape(-1,1))

y2

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y2, test_size=0.33)
#We are gonna use Lasso Regression



from sklearn.linear_model import Lasso

model = Lasso(alpha = 0.001,max_iter=10e4)
#Using Lasso model 



model.fit(X_train,y_train)
#Calculating R²



model.score(X_train,y_train)
#to evaluate the model

resulttraining = model.predict(X_test)
from sklearn import metrics

mrse = np.sqrt(metrics.mean_squared_error(y_test,resulttraining))

mae = metrics.mean_absolute_error(y_test,resulttraining)



ytestreal = scaler.inverse_transform(y_test)

ypredictedreal = scaler.inverse_transform(resulttraining)



mrsereal = np.sqrt(metrics.mean_squared_error(ytestreal,ypredictedreal))

maereal = metrics.mean_absolute_error(ytestreal,ypredictedreal)





print(mse)

print(mae)

print(mrsereal)

print(maereal)
#Now Applying model to test set
xtest = test3

testresult = model.predict(xtest)

testresult2 = scaler.inverse_transform(testresult)
submission = pd.DataFrame()

testresult2
submission['Id'] = np.array(xtest['Id'])

submission['SalePrice'] = np.array(testresult2)



submission.head()
#Exporting to CSV



submission.to_csv('Submission_PedroAlvesLassorev3.csv',index=False)