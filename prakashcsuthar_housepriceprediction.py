import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set()
#import the Data
df_train=pd.read_csv("train.csv")
df_test=pd.read_csv("test.csv")
df_train['TrainData'] = 1
df_test['TrainData'] =0
#Lets remove the outliers in the sale Price and then build the model.
# df_train['SalePrice'].plot(kind='box')
q25 = df_train['SalePrice'].quantile(0.25)
q75 = df_train['SalePrice'].quantile(0.75)
IQR = q75 - q25
LowerValue = q25 - (1.5 * IQR)
UpperValue = q75 + (1.5 * IQR)
df_train = df_train[df_train['SalePrice'] <=  UpperValue]
#Merging the test and train Data set
df = pd.concat([df_train, df_test])
print(df.shape)
print(df_train.shape)
print(df_test.shape)
df.head()
plt.plot(df['SalePrice'])
## CHECK THE NULL COLUMNS
df_nulls =df.isnull().mean()
df_nulls [df_nulls >  0.8]
##droping the columns which have too many null values.
df.drop(columns=['Alley','PoolQC','Fence','MiscFeature'] , inplace=True)
# CHECK THE VARIANCE CATEGORICAL FEATURES SPREAD.
cat_features=df.select_dtypes(include=['object']).columns
len(cat_features)
n_col = 3
n_rows = int(len(cat_features) / n_col) + 1
fig , ax = plt.subplots(n_rows, n_col)
fig.set_figheight(5*n_rows)
fig.set_figwidth(10)
i=0
for feature in cat_features:
    df_cnt = df[feature].value_counts()
    ax[int(i/n_col) , np.mod(i,n_col) ].bar(df_cnt.index , 
                                            df_cnt.values / df_cnt.sum()
                                            )
    ax[int(i/n_col), np.mod(i,n_col)].set_title(feature)
    i=i+1

# It is seen that for some of the features the 95% or  more values are concentrated 
# to a single category. We can consider these features are constant and drop them.
# We check only for top 5 values
df_cnt =pd.DataFrame(columns=['Value1','Value2','Value3','Value4','Value5']
                     , index = cat_features)
for c in cat_features:
    i=1
    for rec in df[c].value_counts() / df[c].value_counts().sum():
       df_cnt.loc[c]['Value' + str(i) ]  = round(rec,4)
       i = i + 1

col_with_constant_values=df_cnt[df_cnt['Value1'] >= 0.95].index
df.drop(columns=col_with_constant_values,inplace=True)
# col_with_constant_values
cat_features=df.select_dtypes(include=['object']).columns
len(cat_features)
df_null_cnt= df[cat_features].isnull().sum()
df_null_cnt[df_null_cnt>0].index
# From Data Description we replace 
# Exterior2nd --> None
# MasVnrType --> None
# BsmtQual --> NA
# BsmtCond --> NA
# BsmtExposure --> NA
# BsmtFinType1 --> NA
# BsmtFinType2 --> NA
# FireplaceQu --> NA
# GarageType --> NA
# GarageFinish --> NA
# GarageQual --> NA
# SaleType --> Oth
df['Exterior2nd'] = df['Exterior2nd'].fillna('None')
df['MasVnrType'] = df['MasVnrType'].fillna('None')
df['BsmtQual'] = df['BsmtQual'].fillna('NA')
df['BsmtCond'] = df['BsmtCond'].fillna('NA')
df['BsmtExposure'] = df['BsmtExposure'].fillna('NA')
df['BsmtFinType1'] = df['BsmtFinType1'].fillna('NA')
df['BsmtFinType2'] = df['BsmtFinType2'].fillna('NA')
df['FireplaceQu'] = df['FireplaceQu'].fillna('NA')
df['GarageType'] = df['GarageType'].fillna('NA')
df['GarageFinish'] = df['GarageFinish'].fillna('NA')
df['GarageQual'] = df['GarageQual'].fillna('NA')
df['SaleType'] = df['SaleType'].fillna('Oth')

#Refresh the list
df_null_cnt= df[cat_features].isnull().sum()
df_null_cnt[df_null_cnt>0].index
# For the remaining features we impute them with most occuring value
df['MSZoning']=df['MSZoning'].fillna(df['MSZoning'].mode()[0])
df['Exterior1st']=df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
df['Electrical']=df['Electrical'].fillna(df['Electrical'].mode()[0])
df['KitchenQual']=df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])
df['Functional']=df['Functional'].fillna(df['Functional'].mode()[0])
number_features = df.select_dtypes(exclude=['object']).columns
# len(number_features)
number_features
df_null_cnt= df[number_features].isnull().sum()
df_null_cnt[df_null_cnt>0].index
df_null_cnt.sort_values()
## Replace BsmtFinSF1, BsmtFinSF2, GarageArea, BsmtUnfSF, TotalBsmtSF, GarageCars,
## BsmtHalfBath, BsmtFullBath as zero.
df['BsmtFinSF1'] = df['BsmtFinSF1'].fillna(0)
df['BsmtFinSF2'] = df['BsmtFinSF2'].fillna(0)
df['GarageArea'] = df['GarageArea'].fillna(0)
df['BsmtUnfSF'] = df['BsmtUnfSF'].fillna(0)
df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(0)
df['GarageCars'] = df['GarageCars'].fillna(0)
df['BsmtHalfBath'] = df['BsmtHalfBath'].fillna(0)
df['BsmtFullBath'] = df['BsmtFullBath'].fillna(0)
df_null_cnt= df[number_features].isnull().sum()
df_null_cnt[df_null_cnt>0].index
## Replacing others with average value
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())
df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].mean())
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].mean())

cat_features = df.select_dtypes(include=['object']).columns
cat_features
# Based on the data description we understand that Following features have the ordinal 
# values. We can replace them with the ordinal values.
# LotShape , ExterQual, ExterCond, BsmtQual, BsmtCond, BsmtExposure, HeatingQC, 
# CentralAir, KitchenQual, FireplaceQu, GarageFinish, GarageQual
map_dict={'Reg':3 ,'IR1':2 ,'IR2':1 ,'IR3':0 ,'Ex':4 ,'Gd':3 ,'TA':2 ,'Fa':1 ,
'Po':0 ,'NA': 0 ,'Av':2,'Mn':1 ,'No':0 ,'NA':0 ,'N':0 ,'Y':1 ,'Fin':3 ,
'RFn':2 ,'Unf':1 ,'NA':0}
ordinal_cols =['LotShape' , 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
                'HeatingQC', 'CentralAir', 'KitchenQual', 'FireplaceQu', 'GarageFinish', 
                 'GarageQual']
for c in ordinal_cols:
    df[c] = df[c].map(map_dict)
# For the remaining categorical variable we generate the dummy values
cat_features = df.select_dtypes(include=['object']).columns
df=pd.get_dummies(df, columns= cat_features , prefix = cat_features) 
X=df[df['TrainData'] ==1]
X=X.drop(columns=['TrainData','SalePrice','Id'])
X_test=df[df['TrainData'] ==0]
X_test=X_test.drop(columns=['TrainData','SalePrice','Id'])
y=df[df['TrainData'] ==1]['SalePrice']
X.index = df[df['TrainData'] ==1]['Id']
X_test.index = df[df['TrainData'] ==0]['Id']
y.index = df[df['TrainData'] ==1]['Id']
# Now we do the sclalling of the variable with standar scaller
from sklearn.preprocessing import StandardScaler
sclr = StandardScaler()
sclr_fit = sclr.fit(X)
X= pd.DataFrame(sclr_fit.transform(X), columns=X.columns , index= X.index)
X_test= pd.DataFrame(sclr_fit.transform(X_test), columns=X_test.columns, index = X_test.index )
#Lets build the model with XGboost
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.3)
# model = RandomForestRegressor(max_depth=20)
model=xgb.XGBRegressor(max_depth=15,n_estimators=200)
model = model.fit(xtrain,ytrain)
print(f'Scoring for Train Data {round(mean_squared_error (y_pred =model.predict(xtrain),y_true= ytrain))}  against min of {ytrain.mean()}')
print(f'Scoring for Test Data {round(mean_squared_error (y_pred =model.predict(xtest),y_true= ytest))}  against min of {ytest.mean()}')
y_predicted= model.predict(X_test)
df_test['SalePrice'] = y_predicted
df_test['SalePrice'].to_csv('Submission.csv')