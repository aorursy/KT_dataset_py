from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import numpy as np

import pandas as pd

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import sklearn

import seaborn as sns

from sklearn.preprocessing import Imputer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_train.head()

df_test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

df_test.shape
df_train.shape
train_ID= df_train['Id']

train_ID

test_ID= df_test['Id']

test_ID
df_test.drop('Id',axis=1,inplace=True)

df_test.drop('Utilities',axis=1,inplace=True)

df_test.shape



#for training the data 

y_train=df_train.SalePrice.values

y_train
df_train.describe()
#lets check for the outliers detection 

#fig, ax=plt.subplots()

#ax.scatter(x=df_train['GrLivArea'], y=df_train['SalePrice'])

#plt.ylabel('SalePrice',fontsize=13)

#plt.xlabel('GrLivArea',fontsize=13)

#plt.show()



#Deleting outliers

#df_train = df_train.drop(df_train[(df_train['GrLivArea']>3000) & (df_train['SalePrice']>11.5)].index)



#again plot it 

#fig,ax=plt.subplots()

#ax.scatter(x=df_train['GrLivArea'], y=df_train['SalePrice'])

#plt.ylabel('SalePrice',fontsize=13)

#plt.xlabel('GrLivArea',fontsize=13)

#plt.show()
#corelation plot

corrmat = df_train.corr()

top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.6]

plt.figure(figsize=(6,6))

g = sns.heatmap(

    df_train[top_corr_features].corr(), 

    annot = True, cmap = "Blues", 

    cbar = False, vmin = .5, 

    vmax = .7, square=True

    )
df_train['SalePrice'].describe()

sns.distplot(df_train['SalePrice']);



#skewness and kurtosis  https://towardsdatascience.com/transforming-skewed-data-73da4c2d0d16

print("Skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())

#checking skewness and kurtosis and applying log function to reduce it.

from scipy import stats

from scipy.stats import norm, skew  #for statastics



#plot histogram and probabaility

fig=plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

sns.distplot(df_train['SalePrice'],fit=norm);

(mu,sigma)= norm.fit(df_train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')

plt.subplot(1,2,2)

res = stats.probplot(df_train['SalePrice'], plot=plt)

plt.suptitle('Before transformation')

# Apply log transformation

df_train.SalePrice = np.log1p(df_train.SalePrice )
# New prediction

y_train = df_train.SalePrice.values
# Plot histogram and probability after transformation

fig = plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

sns.distplot(df_train['SalePrice'], fit=norm);

(mu, sigma) = norm.fit(df_train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')

plt.subplot(1,2,2)

res = stats.probplot(df_train['SalePrice'], plot=plt)

plt.suptitle('After transformation')
# Missing data in train

train_miss = df_train.isnull().sum()

print(train_miss)



train_miss.sort_values(ascending=False)

print(train_miss)
#missing values in test data

test_miss = df_test.isnull().sum()

print(test_miss)



test_miss.sort_values(ascending=False)

print(test_miss)
#Fill out values with most common value

commonNa = [

    'MSZoning','Electrical','KitchenQual',

    'Exterior1st','Exterior2nd','SaleType',

    'LotFrontage','Functional'

    ]



#Fill with zero value

toZero = [

    'MasVnrArea','GarageYrBlt','BsmtHalfBath',

    'BsmtFullBath','GarageArea','GarageCars',

    'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',

    'TotalBsmtSF'

    ]



#Fill with No data

toNoData = [

    'PoolQC','MiscFeature','Alley',

    'Fence','FireplaceQu','GarageType', 

    'GarageFinish','GarageQual', 

    'GarageCond','BsmtQual','BsmtCond', 

    'BsmtExposure','BsmtFinType1','BsmtFinType2',

    'MasVnrType'

    ]

#Function fill missing values

def fillNan(df):

    df['Functional']=df['Functional'].fillna('Typ')

    

    for i in commonNa:

        df[i]=df[i].fillna(df[i].mode()[0])

    for i in toNoData:

        df[i]=df[i].fillna('None')

    for i in toZero:

        df[i]=df[i].fillna(0)

        

        #Removing utilities . No Predictive value

        #df.drop(['Utilities'], axis=1, inplace=True)

        

    return df
df_train.head()
df_train =fillNan(df_train) #passing (df_train,df_test in place of df)      

df_train.isnull().sum()
#for test data

df_test =fillNan(df_test) #passing (df_test in place of df)       

df_test.isnull().sum()
def skew_(df):

    #columns which are skew-candidates

    colls = [col for col in df.columns if df[col].dtype in ['int64','float']]

    skews_df = [col for col in df[colls].columns if df[col].skew() > .7]



    #function to correct skew

    def skewfix(data, data2):

        for i in data2:

            data[i] = np.log1p(data[i])

            return data

    return skewfix(df, skews_df)



df_train, df_test = skew_(df_train), skew_(df_test)

#We need to encode variables with categorical data:

encoder = LabelEncoder()

#sc = StandardScaler()



def encode(df):

    cat_df = [col for col in df.columns if df[col].dtype not in ['int','float']]

    for col in cat_df:

        df[col] = encoder.fit_transform(df[col])

    #df_ = sc.fit_transform(df)

    #df = pd.DataFrame(data=df_, columns = df.columns)

    return df



df_test,df_train=encode(df_test),encode(df_train)
#X = df_train.copy()

X=df_train.drop(['Id', 'SalePrice'], axis=1)

#print(X)



##Below steps are not required, those are for my assumption



#X3=df_train[['OverallQual', 'YearBuilt', 'TotalBsmtSF'  ,'1stFlrSF' , 'GrLivArea' ,'GarageCars' ,'GarageArea']]

#X6=X3.iloc[:-1,:]

#Y3=df_test[['OverallQual', 'YearBuilt', 'TotalBsmtSF'  ,'1stFlrSF' , 'GrLivArea' ,'GarageCars' ,'GarageArea']]

y=df_train.SalePrice.values

#Y3.shape

#X6.shape

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.35,random_state=100)
from sklearn.svm import SVR

regressor = SVR()

regressor.fit(X_train,y_train)

svrpredict=regressor.predict(X_test)
for i in range(0,433):

   print("Error in value number",i,(y_test[i]-svrpredict[i]))
from sklearn.metrics import mean_squared_error

mean_squared_error(y_test,svrpredict)
import xgboost

model = xgboost.XGBRegressor(colsample_bytree=0.4,

                 gamma=0,                 

                 learning_rate=0.07,

                 max_depth=3,

                 min_child_weight=1.5,

                 n_estimators=10000,                                                                    

                 reg_alpha=0.75,

                 reg_lambda=0.45,

                 subsample=0.6,

                 seed=42) 



model.fit(X_train,y_train)

xgb_prediction = model.predict(X_test)





def inv_y(transformed_y):

    return np.exp(transformed_y)

#print(inv_y(xgb_prediction))

for i in range(0,433):

   print("Error in value number",i,(y_test[i]-xgb_prediction[i]))
from sklearn.metrics import mean_squared_error

mean_squared_error(y_test,xgb_prediction)
#X3=df_train[['OverallQual', 'YearBuilt', 'TotalBsmtSF'  ,'1stFlrSF' , 'GrLivArea' ,'GarageCars' ,'GarageArea']]

#X6=X3.iloc[:-1,:]

#Y3=df_test[['OverallQual', 'YearBuilt', 'TotalBsmtSF'  ,'1stFlrSF' , 'GrLivArea' ,'GarageCars' ,'GarageArea']]

#Y3.shape

#y=df_train.iloc[:-1,:].SalePrice.values

#X6.shape

#y.shape



X=df_train.drop(['Id', 'SalePrice'], axis=1)

X4=X.iloc[:-1,:]

X5=X4.drop('Utilities',axis=1)

y=df_train.iloc[:-1,:].SalePrice.values

X5.shape

df_test.shape

y.shape



import xgboost

model = xgboost.XGBRegressor(colsample_bytree=0.4,

                 gamma=0,                 

                 learning_rate=0.07,

                 max_depth=3,

                 min_child_weight=1.5,

                 n_estimators=10000,                                                                    

                 reg_alpha=0.75,

                 reg_lambda=0.45,

                 subsample=0.6,

                 seed=42) 



model.fit(X5,y)

xgb_prediction = model.predict(df_test)



#for inverse, to get original values

def inv_y(transformed_y):

    return np.exp(transformed_y)

#print(inv_y(xgb_prediction))
submission=pd.DataFrame({ "Id": test_ID, "SalePrice": inv_y(xgb_prediction)})

submission.to_csv('Submissionfile.csv',index=False)