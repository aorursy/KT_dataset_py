import pandas as pd 

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%config IPCompleter.greedy=True

train_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train_test_data = [train_data, test_data]
train_data.info()
#finding no of null values and the column name which contains null value

print("no of null values in each column of train set is\n",train_data.isna().sum())

nan_val= []

for i in train_data.columns:

    if(train_data[i].isna().sum() > 1000):

        nan_val.append(i)

print("\nno of columns with null values are\n",len(nan_val))      
print("no of null values in each column of train set is\n",train_data.isna().sum())

nan_val= []

for i in test_data.columns:

    if(test_data[i].isna().sum() > 1000):

        nan_val.append(i)

print("\nno of columns with null values are\n",len(nan_val))   
nan_val1= []

for i in test_data.columns:

    if(train_data[i].isna().sum() >1000):

        nan_val1.append(i)

#print("\nno of columns with null values are\n",len(nan_val1))    

print(nan_val1)
train_data.drop(columns = ['Alley', 'PoolQC', 'Fence', 'MiscFeature'],inplace = True)

test_data.drop(columns = ['Alley', 'PoolQC', 'Fence', 'MiscFeature'],inplace = True)
nan_val1= []

for i in test_data.columns:

    if(train_data[i].isna().sum()<500 and train_data[i].isna().sum()!=0 ):

        nan_val1.append(i)

print(nan_val1)
#for dataset in train_test_data:

#print(test_data['MasVnrType'].mode(),'\n')

#print(train_data.groupby('MasVnrType')['MasVnrArea'].mean())

#print(test_data.groupby('MasVnrType')['MasVnrArea'].mean())

train_data['MasVnrType'].fillna("None",inplace = True)

test_data['MasVnrType'].fillna("None",inplace = True)
nan_val1= []

for i in test_data.columns:

    if(train_data[i].isna().sum()<300 and train_data[i].isna().sum()!=0 ):

        nan_val1.append(i)

print(nan_val1)
train_data['MasVnrArea'].fillna(train_data.groupby('MasVnrType')['MasVnrArea'].transform('mean'),inplace = True)

test_data['MasVnrArea'].fillna(train_data.groupby('MasVnrType')['MasVnrArea'].transform('mean'),inplace = True)
nan_val1= []

for i in test_data.columns:

    if(train_data[i].isna().sum()<300 and train_data[i].isna().sum()!=0 ):

        nan_val1.append(i)

print(nan_val1)
#quality = train_data[['PavedDrive','SalePrice']]

#quality.groupby('PavedDrive').mean().plot.bar()

#print(quality.groupby('PavedDrive').mean())
re = train_data.corr()

re['LotFrontage']
A=train_data.LotFrontage

Anan=A[~np.isnan(A)] # Remove the NaNs



sns.distplot(Anan,hist=True,bins = 10)
train_data['LotFrontage'].fillna(train_data.LotFrontage.mean(),inplace = True)

test_data['LotFrontage'].fillna(train_data.LotFrontage.mean(),inplace = True)
print(train_data['LotFrontage'].isna().sum())

print(test_data['LotFrontage'].isna().sum())
nan_val1= []

for i in test_data.columns:

    if(train_data[i].isna().sum()<300 and train_data[i].isna().sum()!=0 ):

        nan_val1.append(i)

print(nan_val1)
print(train_data['BsmtQual'].value_counts())

print(test_data['BsmtQual'].value_counts())

train_data.BsmtQual.mode()[0]
train_data['BsmtQual'].fillna('None',inplace = True)

test_data['BsmtQual'].fillna('None',inplace = True)
nan_val1= []

for i in test_data.columns:

    if(train_data[i].isna().sum()<300 and train_data[i].isna().sum()!=0 ):

        nan_val1.append(i)

print(nan_val1)
print(train_data['BsmtCond'].value_counts())

print(test_data['BsmtCond'].value_counts())

train_data['BsmtCond'].fillna("None",inplace = True)

test_data['BsmtCond'].fillna("None",inplace = True)

nan_val1= []

for i in test_data.columns:

    if(train_data[i].isna().sum()<300 and train_data[i].isna().sum()!=0 ):

        nan_val1.append(i)

print(nan_val1)
print(train_data['BsmtExposure'].value_counts())

print(test_data['BsmtExposure'].value_counts())

train_data['BsmtExposure'].fillna("None",inplace = True)

test_data['BsmtExposure'].fillna("None",inplace = True)

nan_val1= []

for i in test_data.columns:

    if(train_data[i].isna().sum()<300 and train_data[i].isna().sum()!=0 ):

        nan_val1.append(i)

print(nan_val1)
print(train_data['BsmtFinType2'].value_counts())

print(test_data['BsmtFinType2'].value_counts())

train_data['BsmtFinType2'].fillna("None",inplace = True)

test_data['BsmtFinType2'].fillna("None",inplace = True)

nan_val1= []

for i in test_data.columns:

    if(train_data[i].isna().sum()<300 and train_data[i].isna().sum()!=0 ):

        nan_val1.append(i)

print(nan_val1)
print(train_data['Electrical'].value_counts())

print(test_data['Electrical'].value_counts())

train_data['Electrical'].fillna(train_data['Electrical'].mode()[0],inplace = True)

test_data['Electrical'].fillna(test_data['Electrical'].mode()[0],inplace = True)

nan_val1= []

for i in test_data.columns:

    if(train_data[i].isna().sum()<300 and train_data[i].isna().sum()!=0 ):

        nan_val1.append(i)

print(nan_val1)
print(train_data['GarageType'].value_counts())

print(test_data['GarageType'].value_counts())

train_data['GarageType'].fillna("None",inplace = True)

test_data['GarageType'].fillna("None",inplace = True)

nan_val1= []

for i in test_data.columns:

    if(train_data[i].isna().sum()<300 and train_data[i].isna().sum()!=0 ):

        nan_val1.append(i)

print(nan_val1)
print(train_data['GarageYrBlt'].value_counts())

print(test_data['GarageYrBlt'].value_counts())

train_data['GarageYrBlt'].fillna(0,inplace = True)

test_data['GarageYrBlt'].fillna(0,inplace = True)

nan_val1= []

for i in test_data.columns:

    if(train_data[i].isna().sum()<300 and train_data[i].isna().sum()!=0 ):

        nan_val1.append(i)

print(nan_val1)
print(train_data['GarageFinish'].value_counts())

print(test_data['GarageFinish'].value_counts())

train_data['GarageFinish'].fillna("None",inplace = True)

test_data['GarageFinish'].fillna("None",inplace = True)

nan_val1= []

for i in test_data.columns:

    if(train_data[i].isna().sum()<300 and train_data[i].isna().sum()!=0 ):

        nan_val1.append(i)

print(nan_val1)
print(train_data['GarageQual'].value_counts())

print(test_data['GarageQual'].value_counts())

train_data['GarageQual'].fillna("None",inplace = True)

test_data['GarageQual'].fillna("None",inplace = True)

nan_val1= []

for i in test_data.columns:

    if(train_data[i].isna().sum()<300 and train_data[i].isna().sum()!=0 ):

        nan_val1.append(i)

print(nan_val1)
print(train_data['GarageCond'].value_counts())

print(test_data['GarageCond'].value_counts())

train_data['GarageCond'].fillna("None",inplace = True)

test_data['GarageCond'].fillna("None",inplace = True)

nan_val1= []

for i in test_data.columns:

    if(train_data[i].isna().sum()<300 and train_data[i].isna().sum()!=0 ):

        nan_val1.append(i)

print(nan_val1)
print(train_data['BsmtFinType1'].value_counts())

print(test_data['BsmtFinType1'].value_counts())

train_data['BsmtFinType1'].fillna("None",inplace = True)

test_data['BsmtFinType1'].fillna("None",inplace = True)

nan_val1= []

for i in test_data.columns:

    if(train_data[i].isna().sum()>300):

        nan_val1.append(i)

print(nan_val1)
train_data['FireplaceQu'].fillna("None",inplace = True)

test_data['FireplaceQu'].fillna("None",inplace = True)
nan_val1 = []

for i in train_data.columns:

    if(train_data[i].isna().sum()>50):

        nan_val1.append(i)

print(nan_val1)
import seaborn as sns
nan_val1 = []

for i in test_data.columns:

    if(test_data[i].isnull().sum()>0):

        nan_val1.append(i)

print(nan_val1)
re['GarageCars']
train_data['GarageArea'].fillna(0,inplace = True)

test_data['GarageArea'].fillna(0,inplace = True)
a=(test_data.groupby('GarageArea')['GarageCars'].min())

b=(test_data.groupby('GarageArea')['GarageCars'].max())

c=(test_data.groupby('GarageArea')['GarageCars'].mean())

print(a,b,c)

#sns.distplot(test_data['GarageArea'])
test_data['GarageCars'].fillna(test_data.groupby('GarageArea')['GarageCars'].transform('mean'),inplace = True)
nan_val1 = []

for i in test_data.columns:

    if(test_data[i].isna().sum()>0):

        nan_val1.append(i)

print(nan_val1)
test_data.Exterior1st.fillna(test_data.Exterior1st.mode()[0],inplace = True)

test_data.Exterior2nd.fillna(test_data.Exterior2nd.mode()[0],inplace = True)

nan_val1 = []

for i in test_data.columns:

    if(test_data[i].isna().sum()>0):

        nan_val1.append(i)

print(nan_val1)
(test_data.BsmtFinSF1.fillna(0,inplace = True))

(test_data.BsmtFinSF2.fillna(0,inplace = True))
nan_val1 = []

for i in test_data.columns:

    if(test_data[i].isna().sum()>0):

        nan_val1.append(i)

print(nan_val1)
test_data.KitchenQual.fillna(test_data.KitchenQual.mode()[0],inplace = True)

test_data.KitchenQual.value_counts()
test_data.MSZoning.fillna(test_data.MSZoning.mode()[0],inplace = True)

test_data.MSZoning.isna().sum()
nan_val1 = []

for i in test_data.columns:

    if(test_data[i].isna().sum()>0):

        nan_val1.append(i)

print(nan_val1)
test_data.Utilities.fillna("AllPub",inplace = True)

nan_val1 = []

for i in test_data.columns:

    if(test_data[i].isna().sum()>0):

        nan_val1.append(i)

print(nan_val1)
test_data.TotalBsmtSF.fillna(test_data.TotalBsmtSF.mean(),inplace = True)

test_data.Functional.fillna("Typ",inplace = True)

nan_val1 = []

for i in test_data.columns:

    if(test_data[i].isna().sum()>0):

        nan_val1.append(i)

print(nan_val1)
print(test_data.BsmtUnfSF.mean())

test_data.BsmtUnfSF.fillna(0,inplace= True)

nan_val1 = []

for i in test_data.columns:

    if(test_data[i].isna().sum()>0):

        nan_val1.append(i)

print(nan_val1)
test_data.BsmtHalfBath.fillna(0.0,inplace = True)

test_data.BsmtFullBath.fillna(0.0,inplace=True)

test_data.SaleType.fillna(test_data.SaleType.mode()[0],inplace=True)
nan_val1 = []

for i in train_data.columns:

    if(train_data[i].isna().sum()>0):

        nan_val1.append(i)

print(nan_val1)
colsn=[]  

for i in test_data.columns:

    if(test_data[i].dtype == 'object'):

        colsn.append(i)

len(colsn)

test_data.shape
train_data.shape
from sklearn import preprocessing
train_copy = train_data.copy()

test_copy = test_data.copy()

enc = pd.concat([train_copy,test_copy],sort = False)

enc.shape
"""l = ['a','b','s','a']

le = preprocessing.LabelEncoder()

le.fit(l)

list(le.transform(l))

l = []

for i in test_data['SaleType']:

    l.append(i)

y = list(le.fit_transform(l))"""
le = preprocessing.LabelEncoder()

for i in colsn:

    v =[]

    for j in enc[i]:

        v.append(j)

    

    enc[i].replace(v,list(le.fit_transform(v)),inplace = True)
enc.dtypes
train_df = enc.iloc[:1460, :]

test_df = enc.iloc[1460:,:]
train_df.shape
test_df.drop('SalePrice',axis = 1,inplace = True)
test_df.isna().sum()
from sklearn.ensemble import RandomForestRegressor

import xgboost

from sklearn import linear_model

xtrain = train_df.drop(['SalePrice'],axis = 1)

ytrain = train_df['SalePrice']
rfc = RandomForestRegressor(n_estimators=900)

rfc.fit(xtrain,ytrain)

y_pred = rfc.predict(test_df)

y_pred
import xgboost

regressor=xgboost.XGBRegressor()





from sklearn.model_selection import RandomizedSearchCV





booster=['gbtree','gblinear']

base_score=[0.25,0.5,0.75,1]

n_estimators = [100, 500, 900, 1100, 1500]

max_depth = [2, 3, 5, 10, 15]

booster=['gbtree','gblinear']

learning_rate=[0.05,0.1,0.15,0.20]

min_child_weight=[1,2,3,4]



# Define the grid of hyperparameters to search

hyperparameter_grid = {

    'n_estimators': n_estimators,

    'max_depth':max_depth,

    'learning_rate':learning_rate,

    'min_child_weight':min_child_weight,

    'booster':booster,

    'base_score':base_score

    }

random_cv = RandomizedSearchCV(estimator=regressor,

            param_distributions=hyperparameter_grid,

            cv=5, n_iter=50,

            scoring = 'neg_mean_absolute_error',n_jobs = 4,

            verbose = 5, 

            return_train_score = True,

            random_state=42)

random_cv.fit(xtrain,ytrain)
#from sklearn.model_selection import GridSearchCV

#grid_cv = GridSearchCV(estimator=regressor,

#            param_grid=hyperparameter_grid,

#            cv=2, 

#            scoring = 'neg_mean_absolute_error',n_jobs = 100,

#            verbose = 5, 

#            return_train_score = True,

#            )

#grid_cv.fit(xtrain,ytrain)
random_cv.best_estimator_
#cl = xgboost.XGBRegressor(base_score=1, booster='gbtree', colsample_bylevel=1,

#             colsample_bynode=1, colsample_bytree=1, gamma=0,

#             importance_type='gain', learning_rate=0.1, max_delta_step=0,

#             max_depth=3, min_child_weight=4, missing=None, n_estimators=1100,

#             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,

#             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

#             silent=None, subsample=1, verbosity=1)

#cl.fit(xtrain,ytrain)

#y_pred2 = cl.predict(test_df)

#y_pred2
#Linear Regression

lm = linear_model.LinearRegression()

model = lm.fit(xtrain,ytrain)

y_pred3 = lm.predict(test_df)

y_pred3+20000
#for i in colsn:

 #   print(train_df[i].value_counts())
#subb = pd.DataFrame({

#       "Id": test_df["Id"],

#       "SalePrice": y_pred2

#   })

#subb.to_csv('house_pricev1.csv', index=False)
#remove utilities 


#sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")


