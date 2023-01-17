import math

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split



test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')
train.plot(x='GrLivArea', y='SalePrice', style='o')
train = train[train.GrLivArea < 4000]

train.plot(x='GrLivArea', y='SalePrice', style='o')
#split the data using test-train split from sklearn.

X = train['GrLivArea']

d = train['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, d, test_size=0.33, random_state=1)



#Create the X matrix for y = W_0 + W_1*X where y is sale price and X is sq footage

X_1stOrder = np.row_stack((np.ones(X_train.size),X_train.values))

X_1stOrder_pinv = np.linalg.pinv(X_1stOrder)

W_1stOrder = np.dot(np.transpose(X_1stOrder_pinv),y_train)



#Add a row of ones to the test set to allow the matrix multiplication

X_test_1stOrder = np.row_stack((np.ones(X_test.size),X_test.values))

FirstOrderTestPredictions = np.dot(np.transpose(X_test_1stOrder),W_1stOrder)



#Create a function for RMSLE since this is what is used by the competition

def rmsle(y, y_pred):

    assert len(y) == len(y_pred)

    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]

    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5



FirstOrderError = rmsle(y_test.values,FirstOrderTestPredictions)

print(FirstOrderError)



#Next add a row of the GrLivArea^2 to the X_1stOrder to create X_2ndOrder

X_2ndOrder = np.row_stack((X_1stOrder, np.square(X_train.values)))

X_2ndOrder_pinv = np.linalg.pinv(X_2ndOrder)

W_2ndOrder = np.dot(np.transpose(X_2ndOrder_pinv),y_train)



#Similarly add a row to the test data-set

X_test_2ndOrder = np.row_stack((np.ones(X_test.size),X_test.values,np.square(X_test.values)))



SecondOrderTestPredictions = np.dot(np.transpose(X_test_2ndOrder),W_2ndOrder)

SecondOrderError = rmsle(y_test.values,SecondOrderTestPredictions)

print(SecondOrderError)
train.plot(x='TotalBsmtSF', y='SalePrice', style='o')
X = train[['GrLivArea','TotalBsmtSF']]



#Again get a slice of data for training and a different one for cross-validation.

#Change the random state to make sure we are switching up the sets which are used

X_train, X_test, y_train, y_test = train_test_split(X, d, test_size=0.33, random_state=2)



#Build the appropriate X matrix

X_TwoVar = np.row_stack((np.ones(X_train['GrLivArea'].size),X_train['GrLivArea'].values,\

                  X_train['TotalBsmtSF'].values, np.square(X_train['TotalBsmtSF'].values)))

                        

X_pinv = np.linalg.pinv(X_TwoVar)

W_TwoVar = np.dot(np.transpose(X_pinv),y_train.values)



#Add a row of ones to the test set to allow the matrix multiplication

X_test_TwoVar = np.row_stack((np.ones(X_test['GrLivArea'].size),X_test['GrLivArea'].values,\

                  X_test['TotalBsmtSF'].values, np.square(X_test['TotalBsmtSF'].values)))

TwoVarTestPredictions = np.dot(np.transpose(X_test_TwoVar),W_TwoVar)

TwoVarError = rmsle(y_test.values,TwoVarTestPredictions)

print(TwoVarError)
train.plot(x='OverallQual',y='SalePrice', style='o')

X = train[['GrLivArea','TotalBsmtSF','OverallQual']]



#Again get a slice of data for training and a different one for cross-validation.

#Change the random state to make sure we are switching up the sets which are used

X_train, X_test, y_train, y_test = train_test_split(X, d, test_size=0.33, random_state=3)



#Build the appropriate X matrix

X_ThreeVar = np.row_stack((np.ones(X_train['GrLivArea'].size),X_train['GrLivArea'].values,\

                  X_train['TotalBsmtSF'].values, np.square(X_train['TotalBsmtSF'].values),\

                  X_train['OverallQual'].values, np.square(X_train['OverallQual'].values)))

                        

X_pinv = np.linalg.pinv(X_ThreeVar)

W_ThreeVar = np.dot(np.transpose(X_pinv),y_train.values)



#Add a row of ones to the test set to allow the matrix multiplication

X_test_ThreeVar = np.row_stack((np.ones(X_test['GrLivArea'].size),X_test['GrLivArea'].values,\

                  X_test['TotalBsmtSF'].values, np.square(X_test['TotalBsmtSF'].values),\

                  X_test['OverallQual'].values, np.square(X_test['OverallQual'].values)))

ThreeVarTestPredictions = np.dot(np.transpose(X_test_ThreeVar),W_ThreeVar)

ThreeVarError = rmsle(y_test.values,ThreeVarTestPredictions)

print(ThreeVarError)
train.plot(x='LotArea',y='SalePrice', style='o')



train = train[train.LotArea < 60000]

d = train['SalePrice']



train.plot(x='LotArea',y='SalePrice', style='o')



#Try a logarithmic fit

X = train[['GrLivArea','TotalBsmtSF','OverallQual','LotArea']]



#Again get a slice of data for training and a different one for cross-validation.

#Change the random state to make sure we are switching up the sets which are used

X_train, X_test, y_train, y_test = train_test_split(X, d, test_size=0.33, random_state=4)



#Build the appropriate X matrix

X_FourVar = np.row_stack((np.ones(X_train['GrLivArea'].size),X_train['GrLivArea'].values,\

                  X_train['TotalBsmtSF'].values, np.square(X_train['TotalBsmtSF'].values),\

                  X_train['OverallQual'].values, np.square(X_train['OverallQual'].values),\

                  X_train['LotArea'].values))

                        

X_pinv = np.linalg.pinv(X_FourVar)

W_FourVar = np.dot(np.transpose(X_pinv),y_train.values)



#Add a row of ones to the test set to allow the matrix multiplication

X_test_FourVar = np.row_stack((np.ones(X_test['GrLivArea'].size),X_test['GrLivArea'].values,\

                  X_test['TotalBsmtSF'].values, np.square(X_test['TotalBsmtSF'].values),\

                  X_test['OverallQual'].values, np.square(X_test['OverallQual'].values),\

                  X_test['LotArea'].values))

FourVarTestPredictions = np.dot(np.transpose(X_test_FourVar),W_FourVar)

FourVarError = rmsle(y_test.values,FourVarTestPredictions)

print(FourVarError)
#Get the desired features from the test set

X = test[['GrLivArea','TotalBsmtSF','OverallQual','LotArea']]

X = np.row_stack((np.ones(X['GrLivArea'].size),X['GrLivArea'].values,\

                  X['TotalBsmtSF'].values, np.square(X['TotalBsmtSF'].values),\

                  X['OverallQual'].values, np.square(X['OverallQual'].values),\

                  X['LotArea'].values))



#There seem to be missing data with respect to TotalBsmtSF. So we can actually

#solve the problem and get submittable results, we will replace all nan with zeros

X = np.nan_to_num(X)

Predictions = np.dot(np.transpose(X),W_FourVar)



#Get the Id for the properties

Ids = test['Id']

nanRow = np.argwhere(np.isnan(Predictions))





submissionDF = pd.DataFrame({'Id':Ids,'SalePrice':Predictions})

submissionDF.head()

submissionDF.to_csv('results.csv',index=False)



print(train.columns)
