import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

%matplotlib inline
import os
print(os.listdir("../input"))
df = pd.read_csv('../input/train.csv')
df.head()
df.shape
nulls = pd.DataFrame(df.isnull().sum())

nulls = nulls[nulls[0] >0]

print(nulls)

plt.figure(figsize=(20,6))
sns.heatmap(df.isnull(), cbar=False,yticklabels=False)
columns = ['Alley','PoolQC','Fence','MiscFeature','FireplaceQu','LotFrontage']
df.drop(columns, axis=1, inplace = True)
df['MasVnrType'].unique()
df['MasVnrType'].fillna(value = 'None', inplace = True)
df['MasVnrType'].unique()
df.update(df['MasVnrArea'].fillna(value = 0, inplace = True))
df.update(df['BsmtQual'].fillna(value = 'NA', inplace = True))
df['BsmtQual'].unique()
df.update(df['BsmtCond'].fillna(value = 'NA', inplace = True))
df['BsmtCond'].unique()
columns = ['BsmtExposure','BsmtFinType1','BsmtFinType2','GarageYrBlt','GarageType','GarageFinish','GarageQual','GarageCond']
df.drop(columns, axis=1, inplace = True)
df['Electrical'].isnull().sum()
df.update(df.Electrical.dropna(inplace= True))
df['Electrical'].isnull().sum()
plt.figure(figsize=(20,6))
sns.heatmap(df.isnull(), cbar=False,yticklabels=False)
CorrTarget = df.corr()
CorrTarget['SalePrice']
dicCorrelations = dict(CorrTarget['SalePrice'])
BestCorrelations = []

for k,v in dicCorrelations.items():
    if v >0.4:
        BestCorrelations.append(k)
    else:
        continue
print(BestCorrelations)
plt.figure(figsize=(12,8))
sns.heatmap(df[BestCorrelations].corr(),cmap = 'magma',annot = True, linecolor = 'black',lw = 1)
df_bestcorr = df[BestCorrelations]
df_bestcorr.head(3)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df_bestcorr.drop('SalePrice',axis=1), df_bestcorr['SalePrice'], test_size = 0.3)
y_train= y_train.values.reshape(-1,1)
y_test= y_test.values.reshape(-1,1)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train,y_train.ravel())
predictions = model.predict(X_test)
plt.figure(figsize=(12,8))
plt.scatter(y_test,predictions, marker = ('v'))
plt.xlabel('Y Test')
plt.ylabel('Predict')
error = y_test-predictions
error = error.reshape(-1,1)
sns.distplot(error,bins=50, color = 'red')
from sklearn.model_selection import GridSearchCV
parameters = {'min_samples_leaf':[1,20], 'min_samples_split':[2,200],'n_estimators':[100,250,500,750,1000]}
grid = GridSearchCV(model,parameters)

grid.fit(X_train,y_train.ravel())
grid.best_params_
best_model = grid.best_estimator_
predictions = best_model.predict(X_test)

plt.figure(figsize=(12,8))
plt.scatter(y_test,predictions, marker = ('v'))
plt.xlabel('Y Test')
plt.ylabel('Predict')
error = y_test-predictions
print(error.sum())
error = error.reshape(-1,1)

sns.distplot(error,bins=50, color = 'red')
predictPrice = best_model.predict(df_bestcorr.drop('SalePrice',axis=1))
x = pd.DataFrame(predictPrice,columns=['SalePrice_Predicted'])

result_comparision = pd.concat([df,x], axis = 1)
result_comparision.head()
result_comparision['Model_Error'] = result_comparision.SalePrice - result_comparision.SalePrice_Predicted
result_comparision[['SalePrice','SalePrice_Predicted','Model_Error']].head()
df_test = pd.read_csv('../input/test.csv', usecols = 
                      [  'OverallQual',
                         'YearBuilt',
                         'YearRemodAdd',
                         'MasVnrArea',
                         'TotalBsmtSF',
                         '1stFlrSF',
                         'GrLivArea',
                         'FullBath',
                         'TotRmsAbvGrd',
                         'Fireplaces',
                         'GarageCars',
                         'GarageArea'])
df_test.isnull().sum()
df_test.update(df_test['MasVnrArea'].fillna(value = 0, inplace = True))
df_test['TotalBsmtSF'].fillna(value = df_test.TotalBsmtSF.mean(), inplace = True)
df_test['GarageCars'].fillna(value = df_test.GarageCars.mean(), inplace = True)
df_test['GarageArea'].fillna(value = df_test.GarageArea.mean(), inplace = True)
df_test.isnull().sum()
predict_result = best_model.predict(df_test)
predict_result = pd.DataFrame(predict_result,columns=['SalePrice'])
index = pd.read_csv('../input/test.csv')
Id = index['Id']
Id = pd.DataFrame(Id)

result = pd.concat([Id,predict_result.round(2)], axis =1)
result.head()
result.to_csv('submission.csv',index=False)