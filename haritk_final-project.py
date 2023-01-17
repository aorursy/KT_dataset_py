# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#joining test and train to do preprocessing
df = train.append(test , ignore_index = True)
plt.hist(train.SalePrice,bins = 25)
#it is right skewed,mot good for training
plt.hist(np.log(train.SalePrice), bins = 25)
#it is almost normal distribution
#hence,we will use log to train the model for better accuracy
df.SalePrice = np.log(df.SalePrice)
#filling the missing values
df.LotFrontage.fillna(df.LotFrontage.median(), inplace=True)

# filling year of built as approx for msiing of GArgeyrbuilt. 
df.GarageYrBlt.fillna(df.YearBuilt, inplace=True)

# Use zero as there are none of these things
df.MasVnrArea.fillna(0, inplace=True)    
df.BsmtHalfBath.fillna(0, inplace=True)
df.BsmtFullBath.fillna(0, inplace=True)
df.GarageArea.fillna(0, inplace=True)
df.GarageCars.fillna(0, inplace=True)    
df.TotalBsmtSF.fillna(0, inplace=True)   
df.BsmtUnfSF.fillna(0, inplace=True)     
df.BsmtFinSF2.fillna(0, inplace=True)    
df.BsmtFinSF1.fillna(0, inplace=True)    
# Filling missing values for categorical features
df.PoolQC.fillna('NA', inplace=True)
df.MiscFeature.fillna('NA', inplace=True)    
df.Alley.fillna('NA', inplace=True)          
df.Fence.fillna('NA', inplace=True)         
df.FireplaceQu.fillna('NA', inplace=True)    
df.GarageCond.fillna('NA', inplace=True)    
df.GarageQual.fillna('NA', inplace=True)     
df.GarageFinish.fillna('NA', inplace=True)   
df.GarageType.fillna('NA', inplace=True)     
df.BsmtExposure.fillna('NA', inplace=True)     
df.BsmtCond.fillna('NA', inplace=True)        
df.BsmtQual.fillna('NA', inplace=True)        
df.BsmtFinType2.fillna('NA', inplace=True)     
df.BsmtFinType1.fillna('NA', inplace=True)     
df.MasVnrType.fillna('None', inplace=True)   
df.Exterior2nd.fillna('None', inplace=True) 

# mode filling
df.Functional.fillna(df.Functional.mode()[0], inplace=True)       
df.Utilities.fillna(df.Utilities.mode()[0], inplace=True)          
df.Exterior1st.fillna(df.Exterior1st.mode()[0], inplace=True)        
df.SaleType.fillna(df.SaleType.mode()[0], inplace=True)                
df.KitchenQual.fillna(df.KitchenQual.mode()[0], inplace=True)        
df.Electrical.fillna(df.Electrical.mode()[0], inplace=True)    
#doing encoding for ordinal variables
df.Alley = df.Alley.map({'NA':0, 'Grvl':1, 'Pave':2})
df.BsmtCond =  df.BsmtCond.map({'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
df.BsmtExposure = df.BsmtExposure.map({'NA':0, 'No':1, 'Mn':2, 'Av':3, 'Gd':4})
df['BsmtFinType1'] = df['BsmtFinType1'].map({'NA':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6})
df['BsmtFinType2'] = df['BsmtFinType2'].map({'NA':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6})
df.BsmtQual = df.BsmtQual.map({'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
df.ExterCond = df.ExterCond.map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
df.ExterQual = df.ExterQual.map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
df.FireplaceQu = df.FireplaceQu.map({'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
df.Functional = df.Functional.map({'Sal':1, 'Sev':2, 'Maj2':3, 'Maj1':4, 'Mod':5, 'Min2':6, 'Min1':7, 'Typ':8})
df.GarageCond = df.GarageCond.map({'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
df.GarageQual = df.GarageQual.map({'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
df.HeatingQC = df.HeatingQC.map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
df.KitchenQual = df.KitchenQual.map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
df.LandSlope = df.LandSlope.map({'Sev':1, 'Mod':2, 'Gtl':3}) 
df.PavedDrive = df.PavedDrive.map({'N':1, 'P':2, 'Y':3})
df.PoolQC = df.PoolQC.map({'NA':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4})
df.Street = df.Street.map({'Grvl':1, 'Pave':2})
df.Utilities = df.Utilities.map({'ELO':1, 'NoSeWa':2, 'NoSewr':3, 'AllPub':4})
#object list
qual = list( train.loc[:,train.dtypes == 'object'].columns.values )
#list of ordinals
l1 = ['Alley','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtQual',
           'ExterCond','ExterQual','FireplaceQu','Functional','GarageCond',
           'GarageQual','HeatingQC','KitchenQual','LandSlope','PavedDrive','PoolQC',
           'Street','Utilities']
c = qual.copy()
for i in l1:
    c.remove(i)
#c is list for one hot encopding(creating dummies)
#creatung dummies
for i in c:
    df = pd.concat([pd.get_dummies(df[i],prefix = i).iloc[:,1:],df],axis =1)
    df.drop(i,axis =1 ,inplace = True)
df.shape
#splitting the datset back into normal
train = df.iloc[:1460,:]
test = df.iloc[1460:,:]
test.shape
test.drop(['Id','SalePrice'],axis = 1,inplace = True)
#creating training set
y = np.log(train.SalePrice)
X = train.drop(['SalePrice', 'Id'], axis=1)
train_copy = train.copy()
#making train copt to do prediction on test as train contains id and saleprice which we need to remove
train_copy.drop(['Id','SalePrice'],axis =1,inplace = True)
from sklearn.linear_model import LinearRegression
clf  = LinearRegression()
clf1 = LinearRegression()
#doing cross validation
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=.2)
model = clf.fit(X_train,y_train)
pred = model.predict(X_test)
print ("R^2 is: \n", model.score(X_test, y_test))
model_lin = clf.fit(train_copy,y)

#pred on test
pred_lin = model_lin.predict(test)
pred_lin
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.scatter(pred,y_test)
#analysis of cv
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, pred))
residual = y_test - pred

#residual plot
plt.scatter(y_test,residual)
#ridge analysis
from sklearn.linear_model import Ridge
rm = Ridge()
ridge_model = rm.fit(X_train, y_train)
pred1 = ridge_model.predict(X_test)

plt.scatter(pred1, y_test, alpha=.75, color='b')
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, pred1))
model_ridge = clf.fit(train_copy,np.log(train.SalePrice))
#pred on test
pred_real_ridge = model_ridge.predict(test)
pred_real_ridge
#pred on tests are similar from linear regression and ridge
plt.scatter(pred_lin,pred_real_ridge)
#lasso model
from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.1)
model_lasso = clf.fit(train_copy,np.log(train.SalePrice))
pred_lasso =model_lasso.predict(test)
#lasso and linear_regression give approx same predictions
plt.scatter(pred_lasso,pred_lin)
#lasso and ridge give approx same predictions
plt.scatter(pred_lasso,pred_real_ridge)
#cv analysis and rmse for lasso
lasso_model = clf.fit(X_train, y_train)
pred1 = lasso_model.predict(X_test)

plt.scatter(pred1, y_test, alpha=.75, color='b')
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, pred1))
