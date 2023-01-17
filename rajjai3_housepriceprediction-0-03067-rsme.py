import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sea

from sklearn.preprocessing import StandardScaler,LabelEncoder,MinMaxScaler

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

from sklearn.impute import SimpleImputer
df_train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_train.shape
def checkForNull(df):

    colList=df.columns

    for col in colList:

        nullCount=pd.isnull(df[col]).sum()

        if(nullCount!=0):

            print("{}-->{}".format(col,nullCount))

            

def imputeData(df):

    colList=df.columns

    for col in colList:

        if(df[col].dtypes=='O'):

            imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

            df_temp=imp.fit_transform(np.array(df[col]).reshape(-1,1))

            df[col]=LabelEncoder().fit_transform(df_temp)

        else:

            imp = SimpleImputer(missing_values=np.nan, strategy='median')

            df[col]=imp.fit_transform(np.array(df[col]).reshape(-1,1))
checkForNull(df_train)
df_train=df_train.drop(['Id','Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1,inplace=False)

df_train.head(5)
sea.scatterplot(df_train['LotFrontage'],df_train['SalePrice'])

plt.show()
df_train=df_train.drop(df_train.loc[df_train['LotFrontage']>250].index,axis=0)

df_train=df_train.drop(df_train.loc[df_train['LotArea']>100000].index,axis=0)

df_train=df_train.drop(df_train.loc[(df_train['OverallCond']==2) & (df_train['SalePrice']>300000)].index,axis=0)

df_train=df_train.drop(df_train.loc[df_train['LowQualFinSF']>550].index,axis=0)

df_train=df_train.drop(df_train.loc[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index,axis=0)

df_train=df_train.drop(df_train.loc[(df_train['OpenPorchSF']>500) & (df_train['SalePrice']<100000)].index,axis=0)

df_train=df_train.drop(df_train.loc[df_train['EnclosedPorch']>500].index,axis=0)

df_train=df_train.drop(df_train.loc[df_train['MiscVal']>3000].index,axis=0)

df_train.shape
checkForNull(df_train)
imputeData(df_train)
checkForNull(df_train)
X=df_train.drop('SalePrice',axis=1)

scaler=MinMaxScaler().fit(np.array(df_train['SalePrice']).reshape(-1,1))

Y=scaler.transform(np.array(df_train['SalePrice']).reshape(-1,1))

#X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=101)
from sklearn.feature_selection import SelectKBest,chi2

best=SelectKBest(chi2,k=70).fit(X,df_train['SalePrice'])

best_X=best.transform(X)



X_train,X_test,Y_train,Y_test=train_test_split(best_X,Y,test_size=0.1,random_state=101)
indices = np.argsort(best.scores_)[::-1]

for i in indices:

    print("{}--->{}".format(df_train.columns[i],best.scores_[i]))
boost=XGBRegressor(n_estimators=150,learning_rate=0.09,max_depth=10,booster='gbtree',verbosity=0,n_jobs=-1,random_state=47)

boost.fit(X_train,Y_train)

np.sqrt(mean_squared_error(Y_train,boost.predict(X_train)))
np.sqrt(mean_squared_error(Y_test,boost.predict(X_test)))
df_test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

checkForNull(df_test)
df_test=df_test.drop(['Id','Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1,inplace=False)

df_test.head(5)
imputeData(df_test)
checkForNull(df_test)
output=boost.predict(best.transform(df_test))

out_transformed=scaler.inverse_transform(output.reshape(-1,1)).reshape(-1,)

out_transformed