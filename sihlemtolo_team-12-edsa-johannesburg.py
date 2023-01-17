#importing the libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

#Blocking some nasty warnings

import warnings

warnings.filterwarnings('ignore', category=Warning)

print(os.listdir("../input"))
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



# A brief look into our training data



df_train.head()
df_train.set_index('Id', inplace=True)

df_test.set_index('Id', inplace=True)
plt.figure(figsize=[40,30])

_ = sns.heatmap(df_train.corr(),annot=True)
df_train.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd','2ndFlrSF'], axis=1, inplace=True)

df_test.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd','2ndFlrSF'], axis=1, inplace=True)
plt.scatter(df_train[['GrLivArea']],df_train[['SalePrice']])

plt.xlabel('Total Living Area Excluding Basement(square foot)')

plt.ylabel('Sale Price')
df_train = df_train[df_train['GrLivArea']<4500]
plt.scatter(df_train[['GrLivArea']],df_train[['SalePrice']])

plt.xlabel('Total Living Area Excluding Basement(square foot)')

plt.ylabel('Sale Price')
# A function to percentage missing of overall

def check_nulls(df):

    percent_missing = (df.isnull().sum() * 100 / len(df)).sort_values()

    return round(percent_missing,2)
check_nulls(df_train)
check_nulls(df_test)
categorical_list = [col for col in df_train.columns if df_train[col].dtypes == object]

numerical_list = [col for col in df_train.columns if df_train[col].dtypes != object]



print('Categories:', categorical_list)

print('Numbers:', numerical_list)
def fill_missing_values(df):

    lst = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1",

             "BsmtFinType2","Fence","FireplaceQu","GarageType","GarageFinish",

             "GarageQual","GarageCond","Electrical","GarageFinish","MiscFeature","MasVnrType","PoolQC"]

    for col in lst:

        df[col] = df[col].fillna("Not present")

        

    lst = ['GarageYrBlt','MasVnrArea','BsmtFinSF1','BsmtFinSF2','TotalBsmtSF',

           'BsmtUnfSF','BsmtFullBath','BsmtHalfBath','MasVnrArea','GarageCars']

    for col in lst:

        df[col] = df[col].fillna(0)

        

    lst = ['Utilities','MSZoning','Exterior1st','Exterior2nd','Electrical','KitchenQual']

    for col in lst:

        df[col] = df[col].fillna(df[col].mode()[0])

    

    df['Functional'] = df['Functional'].fillna('Typ')

    df['SaleType'] = df['SaleType'].fillna('Normal')

    df['LotFrontage'] = df['LotFrontage'].fillna(df.LotFrontage.mean())

    #removing 'PoolQC' as discussed

    df.drop('PoolQC', axis=1, inplace=True)

   

fill_missing_values(df_train)

fill_missing_values(df_test)
# Checking null values

print(df_train.isnull().sum().sum())

print(df_test.isnull().sum().sum())
# A function to label encode our ordinal data

def label_encode(df):

    df['MSSubClass'] = df['MSSubClass'].astype(object)

    df = df.replace({"Alley" : {"Not present" : 0, "Grvl" : 1, "Pave" : 2},

    "BsmtCond" : {"Not present" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

    "BsmtExposure" : {"Not present" : 0, "No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},

    "BsmtFinType1" : {"Not present" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 

    "ALQ" : 5, "GLQ" : 6},

    "BsmtFinType2" : {"Not present" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 

    "ALQ" : 5, "GLQ" : 6},

    "BsmtQual" : {"Not present" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},

    "CentralAir" : {"N" : 0, "Y" : 1},

    "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

    "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

    "FireplaceQu" : {"Not present" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

    "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 

    "Min2" : 6, "Min1" : 7, "Typ" : 8},

    "GarageCond" : {"Not present" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

    "GarageQual" : {"Not present" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

    "GarageFinish" :{"Not present" : 0, "Unf" : 1, "RFn" : 2, "Fin" : 3},

    "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

    "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

    "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},

    "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},

    "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},

    "PoolQC" : {"Not present" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},

    "Street" : {"Grvl" : 1, "Pave" : 2},

    "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4},

    "Fence": {"Not present" : 0, "MnWw" : 1, "GdWo" : 2, "MnPrv" : 3, "GdPrv" : 4 }},

                       

                     )

    return df



df_train = label_encode(df_train)

df_test = label_encode(df_test)
_ = sns.distplot(df_train["SalePrice"])
df_train['SalePrice']=np.log(df_train['SalePrice'])

_ = sns.distplot(df_train["SalePrice"])
cat_list = [col for col in df_train.columns if df_train[col].dtypes == object]

num_list = [col for col in df_train.columns if df_train[col].dtypes != object]
categorical_data = df_train[cat_list]

numerical_data = df_train[num_list]

df_train = categorical_data.join(numerical_data)
num_list.remove('SalePrice')

cat_test = df_test[cat_list]

num_test = df_test[num_list]

df_test = cat_test.join(num_test)
X = df_train.drop('SalePrice', axis=1).values

y = df_train['SalePrice'].values

df_test_values = df_test.values
#We import the LabelEncoder and OneHotEncoder classes

from sklearn.preprocessing import LabelEncoder, OneHotEncoder



# Let us now define a function that will use the classes we imported to encode our data

def encode(X):

    # We create an object of LabelEncoder

    labelencoder = LabelEncoder()

    for i in range(len(cat_list)):

        

        #Using the fit_transform method we will convert each column of the categorical data into

        #numerical values

        

        X[:,i] = labelencoder.fit_transform(X[:,i])

    for i in range(len(cat_list)):

        # Now we will convert the values into dummy variables

        onehotencoder = OneHotEncoder(categorical_features=[i])

        X = onehotencoder.fit_transform(X).toarray()

        #For each column, we will remove the first dummy variable. This is done to avoid dummy variable trap

        X = X[:,i:]

    
encode(X)

encode(df_test_values)
#We need to import the StandardScaler class

from sklearn.preprocessing import StandardScaler



#We create an object of the class

sc = StandardScaler()



#We scale both X (training data) and the testing data

X = sc.fit_transform(X)

df_test_values = sc.transform(df_test_values)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
import math

def rmsle(y, y_pred):

    assert len(y) == len(y_pred)

    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]

    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5
from sklearn.linear_model import LinearRegression

lm = LinearRegression()



#We train our model using 80% of the training data and predict

lm.fit(X_train,y_train)

y_pred_reg = lm.predict(X_test)
lm.intercept_
rmsle(np.exp(y_test), np.exp(y_pred_reg))
from sklearn.linear_model import Lasso

lasso_model = Lasso()

lasso_model.fit(X_train,y_train)
y_pred_lasso = lasso_model.predict(X_test)
rmsle(np.exp(y_test), np.exp(y_pred_lasso))
from sklearn.linear_model import LassoCV

lcv = LassoCV()

lcv.fit(X_train,y_train)
lcv.alpha_
lcv.intercept_
y_pred_lassocv = lcv.predict(X_test)
rmsle(np.exp(y_test),np.exp(y_pred_lassocv))
model = Lasso(lcv.alpha_)

model.fit(X,y)

y_pred = model.predict(df_test_values)

predictions = np.exp(y_pred)
result=pd.DataFrame({'Id':df_test.index, 'SalePrice':predictions})

result.to_csv('submission.csv', index=False)
fig = plt.figure(figsize=(20,10))

plt.subplot(1, 3, 1)

plt.plot(np.arange(len(y_test)), y_test, label='Testing')

plt.plot(np.arange(len(y_pred_reg)), y_pred_reg, label='Regression')

plt.legend()

plt.subplot(1,3,2)

plt.plot(np.arange(len(y_test)), y_test, label='Testing')

plt.plot(np.arange(len(y_pred_lasso)), y_pred_lasso, label="Lasso")

plt.legend()

plt.subplot(1,3,3)

plt.plot(np.arange(len(y_test)), y_test, label='Testing')

plt.plot(np.arange(len(y_pred_lassocv)), y_pred_lassocv, label="Lasso CV")

plt.legend()

plt.show()
result.head()