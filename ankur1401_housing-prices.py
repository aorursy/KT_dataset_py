import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import GridSearchCV,train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,AdaBoostRegressor

from sklearn.metrics import classification_report,mean_squared_error,mean_squared_log_error
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
train
test
Y=train['SalePrice']
data=pd.concat((train.drop('SalePrice',axis=1),test))

data=data.reset_index()

data=data.drop('index',axis=1)
data
data.info()
print(open('../input/data_description.txt').read())
corr_matrix=train.iloc[:,1:].corr()

mask=np.zeros_like(corr_matrix,dtype=np.bool)

mask[np.triu_indices_from(mask)]=True
plt.figure(figsize=(40,40))

sns.heatmap(corr_matrix,annot=True,mask=mask)
plt.figure(figsize=(40,20))

sns.heatmap(data.isna(),cmap='viridis',yticklabels=False)
for i in data[data['MSZoning'].isnull()]['MSSubClass'].values:

    mask=(data['MSSubClass']==i) & (data['MSZoning'].isnull()==False)

    mask2=(data['MSSubClass']==i) & (data['MSZoning'].isnull()==True)    

    idx=data[data[mask2].isnull()].index

    data.loc[idx,'MSZoning']=data[mask]['MSZoning'].mode().item()
plt.scatter(x=data['LotFrontage'],y=data['1stFlrSF'],s=2)
#Filling out missing values of "LotFrontage" using Linear Regression.



lin_reg=LinearRegression()

lin_reg.fit(np.array(data[data['LotFrontage'].isna()==False]['1stFlrSF']).reshape(-1,1),

            data[data['LotFrontage'].isna()==False]['LotFrontage'])



def LotFrontage_FIX(arr):

    if pd.isnull(arr[0]):

        return np.around(lin_reg.predict(np.array(arr[1]).reshape(-1,1)))[0]

    else:

        return arr[0]

    

data['LotFrontage']=data[['LotFrontage','1stFlrSF']].apply(LotFrontage_FIX,axis=1)
#Filling "NA" for No Alley access.



data['Alley'].fillna('NA',inplace=True)
#Filling out missing values of "Utilities" with most frequent value in "Utilities".



data['Utilities'].fillna(data['Utilities'].mode().item(),inplace=True)
#Filling missing value in 'Exterior1st' & 'Exterior2nd' with mode.



data['Exterior1st'].fillna(data['Exterior1st'].mode().item(),inplace=True)



data['Exterior2nd'].fillna(data['Exterior2nd'].mode().item(),inplace=True)
#Filling out missing values of "MasVnrArea" with 0.



data['MasVnrArea'].fillna(0,inplace=True)
#Filling out missing values of "MasVnrType" using Random Forest Classification.



rf_obj=RandomForestClassifier(max_depth=5,n_estimators=50)

rf_obj.fit(np.array(data[data['MasVnrType'].isna()==False]['MasVnrArea']).reshape(-1,1),

           data[data['MasVnrType'].isna()==False]['MasVnrType'])



def MasVnrType_FIX(arr):

    if pd.isnull(arr[0]):

        return rf_obj.predict(np.array(arr[1]).reshape(-1,1))[0]

    else:

        return arr[0]

data['MasVnrType']=data[['MasVnrType','MasVnrArea']].apply(MasVnrType_FIX,axis=1)
#Filling "NA" for No Basement.



mask=(data['BsmtQual'].isna() & 

      data['BsmtCond'].isna() & 

      data['BsmtExposure'].isna() & 

      data['BsmtFinType1'].isna() & 

      data['BsmtFinType2'].isna())



data.loc[data[mask].index,['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']]="NA"
#Filling Missing values of Basement ('BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2').



cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']



def Bsmt_FIX(df,colno):

    idx=df[df.iloc[:,colno].isnull()].index



    for i in idx:

        value = df.loc[i,np.append(cols[:colno],cols[colno+1:])].values

        mask=(df[np.append(cols[:colno],cols[colno+1:])]==value)

        df.loc[i,cols[colno]] = (df[mask.all(axis=1)].iloc[:,colno].mode().values[0])



    return df.iloc[:,colno]





for i in range(5):

    data.loc[:,cols[i]]=Bsmt_FIX(data.loc[:,cols],i)
#Filling Missing values of Basement ('BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath').



cols_=['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']

mask=(data[data[cols_].isnull().any(axis=1)][cols]=='NA').all(axis=1)

data.loc[mask.index,cols_]=0.0
#Filling missing value in 'Electrical' with mode.



data['Electrical'].fillna(data['Electrical'].mode().item(),inplace=True)
#There is only 1 missing value in "KitchenQual".



print('Kitchen rating with missing value is',data[data['KitchenQual'].isna()]['KitchenAbvGr'].item())

print('Kitchen Quality with Rating=1 is',data[data['KitchenAbvGr']==1]['KitchenQual'].mode().item())
#Filling out missing values of "KitchenQual" with "TA".



data['KitchenQual'].fillna('TA',inplace=True)
#Filling missing value in 'Functional' with mode.



data['Functional'].fillna(data['Functional'].mode()[0],inplace=True)
#Filling "NA" for No Fireplace.



def FireplaceQu_FIX(arr):

    if arr[0]==0:

        return "NA"

    else:

        return arr[1]



data['FireplaceQu']=data[['Fireplaces','FireplaceQu']].apply(FireplaceQu_FIX,axis=1)
#Filling "NA" for No Garage.

mask=(data['GarageType'].isna() & 

      data['GarageYrBlt'].isna() & 

      data['GarageFinish'].isna() & 

      data['GarageQual'].isna() & 

      data['GarageCond'].isna())



data.loc[data[mask].index,['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond']]="NA"
rf_obj=RandomForestClassifier(n_estimators=7,max_depth=5)

rf_obj.fit((np.array(data[(data['GarageArea'].isnull()==False) & 

            (data['GarageCars'].isnull()==False)]['GarageArea'])).reshape(-1,1),

           

           np.array(data[(data['GarageArea'].isnull()==False) & 

            (data['GarageCars'].isnull()==False)]['GarageCars']))



data['GarageArea'].fillna(data['GarageArea'].mean(),inplace=True)



def GarageCars_FIX(arr):

    if pd.isnull(arr[1]):

        return rf_obj.predict(np.array(arr[0]).reshape(-1,1))[0]

    else:

        return arr[1]

    

data['GarageCars']=data[['GarageArea','GarageCars']].apply(GarageCars_FIX,axis=1)
data['GarageFinish'].fillna(data[data['GarageFinish']!='NA']['GarageFinish'].mode().item(),inplace=True)

data['GarageYrBlt'].fillna(data[data['GarageYrBlt']!='NA']['GarageYrBlt'].mode().item(),inplace=True)

data['GarageQual'].fillna(data[data['GarageQual']!='NA']['GarageQual'].mode().item(),inplace=True)

data['GarageCond'].fillna(data[data['GarageCond']!='NA']['GarageCond'].mode().item(),inplace=True)
data['GarageYrBlt']=data['GarageYrBlt'].astype(str)
#Filling "NA" for No Pools.



def PoolQC_FIX(arr):

    if arr[0]==0:

        return "NA"

    else:

        return arr[1]



data['PoolQC']=data[['PoolArea','PoolQC']].apply(PoolQC_FIX,axis=1)
data['PoolQC'].fillna('Gd',inplace=True)
#Filling out missing values of "Fence" with "NA".



data['Fence'].fillna('NA',inplace=True)
#Filling "NA" for '0' MiscVal.



def MiscFeature_FIX(arr):

    if arr[0]==0:

        return "NA"

    else:

        return arr[1]



data['MiscFeature']=data[['MiscVal','MiscFeature']].apply(MiscFeature_FIX,axis=1)
rf_obj=RandomForestClassifier()

rf_obj.fit(np.array(data[data['MiscFeature'].isnull()==False]['MiscVal']).reshape(-1,1),

           np.array(data[data['MiscFeature'].isnull()==False]['MiscFeature']))



def MiscFeature_FIX_2(arr):

    if pd.isnull(arr[0]):

        return rf_obj.predict(np.array(arr[1]).reshape(-1,1))[0]

    else:

        return arr[0]

    

data['MiscFeature']=data[['MiscFeature','MiscVal']].apply(MiscFeature_FIX_2,axis=1)
data['SaleType'].fillna(data['SaleType'].mode()[0],inplace=True)
plt.figure(figsize=(40,20))

sns.heatmap(data.isna(),cmap='viridis',yticklabels=False)
data.info()
X=data.values
y=Y.values
cat_idx=[2,5,6,7,8,9,10,11,12,13,14,15,16,21,22,23,24,25,27,28,29,30,31,32,33,35,39,40,41,42,53,55,57,58,59,60,63,64,65,72,73,74,78,79]
for i in cat_idx:

    label_encode=LabelEncoder()

    X[:,i]=label_encode.fit_transform(X[:,i])
def Score(y_true,y_pred):

    return mean_squared_error(np.log10(y_true),np.log10(y_pred))
X_train,X_val,y_train,y_val=train_test_split(X[:1460,1:],y,test_size=0.33,random_state=42)
from lightgbm import LGBMRegressor
lgb=LGBMRegressor(boosting_type='dart',learning_rate=0.1,max_depth=5,n_estimators=1000)

lgb.fit(X_train,y_train)
lgb.score(X_train,y_train)
lgb.score(X_val,y_val)
Score(y_train,lgb.predict(X_train))
Score(y_val,lgb.predict(X_val))
lgb.fit(X[:1460,1:],y)
df=pd.DataFrame(np.concatenate((X[1460:,0].reshape(-1,1),lgb.predict(X[1460:,1:]).reshape(-1,1)),axis=1),columns=['Id','SalePrice'])
df.to_csv('Submission.csv',index=False)