import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import os

print(os.listdir("../input"))

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
train_data.head()
houses_data = train_data.append(test_data,ignore_index=True)

houses_data.shape
SaleCondition_pivot=houses_data.pivot_table(index='SaleCondition',

values='SalePrice',aggfunc=np.median)

SaleCondition_pivot.plot(kind='bar',color='pink')

plt.xticks(rotation=0)
houses_data['SaleCondition_d']=np.where(houses_data['SaleCondition']!='Partial',0,1)

SaleCondition_pivot=houses_data.pivot_table(index='SaleCondition_d',

values='SalePrice',aggfunc=np.median)

SaleCondition_pivot.plot(kind='bar',color='pink')

plt.xticks(rotation=0)
cols=houses_data.columns

i=14 #2

SaleCondition_pivot=houses_data.pivot_table(index=cols[i],

values='SalePrice',aggfunc=np.median)

SaleCondition_pivot.plot(kind='bar',color='pink')

plt.xticks(rotation=0)
#Tahmin etmemiz gereken değişken analizi

train_data['SalePrice'].describe()
is_null_cols=houses_data.isnull()

a=(is_null_cols.sum()/len(is_null_cols)*100).sort_values(axis=0, ascending=False) 

a[a.apply(lambda x: x>0)]
x=houses_data.isnull().sum()

x[x>0]
def fill_with_zero(column_lst):

    for column in column_lst:

        houses_data[column]=houses_data[column].fillna(0)
def fill_with_none(column_lst):

    for column in column_lst:

        houses_data[column]=houses_data[column].fillna("None")
zero_cols=["GarageCars","GarageArea","GarageYrBlt",

           'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']

fill_with_zero(zero_cols)

none_cols=["Alley","BsmtQual","BsmtCond","BsmtExposure",'BsmtFinType1''',"BsmtFinType2",

"FireplaceQu","GarageType","GarageFinish","GarageCond","PoolQC","Fence",'GarageQual']

fill_with_none(none_cols)

houses_data["LotFrontage"] = houses_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

houses_data['Electrical'] = houses_data['Electrical'].fillna(houses_data['Electrical'].value_counts().index[0])

houses_data['Exterior1st'] = houses_data['Exterior1st'].fillna(houses_data['Exterior1st'].value_counts().index[0])

houses_data['Exterior2nd'] = houses_data['Exterior2nd'].fillna(houses_data['Exterior2nd'].value_counts().index[0])

houses_data['KitchenQual'] = houses_data['KitchenQual'].fillna(houses_data['KitchenQual'].value_counts().index[0])

houses_data['SaleType'] = houses_data['SaleType'].fillna(houses_data['SaleType'].value_counts().index[0])

houses_data['MSZoning'] = houses_data['MSZoning'].fillna(houses_data['MSZoning'].value_counts().index[0])

houses_data["MasVnrType"] = houses_data["MasVnrType"].fillna("None")

houses_data["MasVnrArea"] = houses_data["MasVnrArea"].fillna(0)

houses_data.SalePrice=houses_data.SalePrice.fillna(0)

houses_data.SalePrice=np.log1p(houses_data.SalePrice)
qual_dict = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

houses_data["ExterQual"] = houses_data["ExterQual"].map(qual_dict).astype(int)

houses_data["ExterCond"] = houses_data["ExterCond"].map(qual_dict).astype(int)

houses_data["BsmtQual"] = houses_data["BsmtQual"].map(qual_dict).astype(int)

houses_data["BsmtCond"] = houses_data["BsmtCond"].map(qual_dict).astype(int)

houses_data["HeatingQC"] = houses_data["HeatingQC"].map(qual_dict).astype(int)

houses_data["KitchenQual"] = houses_data["KitchenQual"].map(qual_dict).astype(int)

houses_data["FireplaceQu"] = houses_data["FireplaceQu"].map(qual_dict).astype(int)

houses_data["GarageQual"] = houses_data["GarageQual"].map(qual_dict).astype(int)

houses_data["GarageCond"] = houses_data["GarageCond"].map(qual_dict).astype(int)

houses_data["BsmtExposure"] = houses_data["BsmtExposure"].map({'None': 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}).astype(int)
#Normalize

num_cols=houses_data.select_dtypes(['number']).columns

for num_col in num_cols:

    col=houses_data[num_col]

    col=(col-np.mean(col))/np.std(col)
dummies=pd.get_dummies(houses_data.select_dtypes(include=['object']), drop_first=True)

houses_data.drop(houses_data.select_dtypes(['object']).columns,axis=1,inplace=True)

houses_data.shape
houses_data = pd.concat([houses_data, dummies], axis=1)

#houses_data.append(dummies,ignore_index=True)
x=train_data.shape[0]

x
numero=houses_data.isnull().sum()

numero[numero>0]
train_data=houses_data[:x]
test_data=houses_data[x:]
rmse_df=pd.DataFrame(columns=['Alg','RMSE'])
y=train_data.SalePrice

x=train_data.drop(['SalePrice','Id'],axis=1)

# veri seti %80-%20 train-test ayrılmıştır.

x_train, x_test, y_train, y_test=train_test_split(x,y,random_state=13,test_size=.2)

test_data.drop('SalePrice',axis=1,inplace=True)
from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor(n_estimators=1000)

rfr_model=rfr.fit(x_train,y_train)

rfr_predictions=rfr_model.predict(x_test)



print(rfr_model.score(x_test,y_test))
print('R squared=',rfr_model.score(x_test,y_test))



rmse_rfr=mean_squared_error(y_test,rfr_predictions)

print('RMSE',rmse_rfr)

rmse_df=rmse_df.append({'Alg':'RandomForestRegressor','RMSE':rmse_rfr}, ignore_index=True)





plt.scatter(rfr_predictions, y_test,alpha=.75)

plt.xlabel='Predicted Sale Price'

plt.ylabel='Actual Sale Price'
print(' Predicted \t'+str(np.exp(rfr_predictions[1]))+'\n Actual \t'+str(np.exp(y_test.reset_index().SalePrice[1])))
print(' Predicted \t'+str(np.exp(rfr_predictions[10]))+'\n Actual \t'+str(np.exp(y_test.reset_index().SalePrice[10])))
predictions_rfr=rfr_model.predict(test_data.drop(['Id'],axis=1))
submission=pd.DataFrame()

submission['Id']=test_data.Id



untransformed_preds=np.exp(predictions_rfr)



#print(predictions_ridge[:5],untransformed_preds[:5])



submission['SalePrice']=untransformed_preds

submission.head()