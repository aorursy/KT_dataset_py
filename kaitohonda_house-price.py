import numpy as np
import pandas as pd

train= pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

train.head()
train.info()
print(train.shape,test.shape)
train = train.drop('Alley',axis=1).drop('FireplaceQu',axis=1).drop('PoolQC',axis=1).drop('Fence',axis=1).drop('MiscFeature',axis=1)

test =test.drop('Alley',axis=1).drop('FireplaceQu',axis=1).drop('PoolQC',axis=1).drop('Fence',axis=1).drop('MiscFeature',axis=1)
train_id = train['Id']
test_id = test['Id']

y_train = train['SalePrice']
x_train = train.drop(['Id','SalePrice'],axis=1)
x_test = test.drop('Id',axis=1)
x_train = x_train.fillna(x_train.median())
x_test = x_test.fillna(x_test.median())

x_train.info()
for i in range(x_train.shape[1]):
    if x_train.iloc[:,i].dtype == object:
        mode = x_train.mode()[x_train.columns.values[i]].values
        for j in range(x_train.shape[0]):
            if x_train.isnull().iloc[j,i]==True:
                x_train.iloc[j,i] =mode
for i in range(x_test.shape[1]):
    if x_test.iloc[:,i].dtype == object:
        mode = x_test.mode()[x_test.columns.values[i]].values
        for j in range(x_test.shape[0]):
            if x_test.isnull().iloc[j,i]==True:
                x_test.iloc[j,i] = mode
x_train.isnull().sum().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#ラベルエンコーダー(訓練セット）
for i in range(x_train.shape[1]):
    if x_train.iloc[:,i].dtypes == object:
        le.fit(list(x_train[x_train.columns.values[i]].values)) 
        x_train[x_train.columns.values[i]] = le.transform(list(x_train[x_train.columns.values[i]].values))

#ラベルエンコーダー(テストセット）
for i in range(x_test.shape[1]):
    if x_test.iloc[:,i].dtypes == object:
        le.fit(list(x_test[x_test.columns.values[i]].values)) 
        x_test[x_test.columns.values[i]] = le.transform(list(x_test[x_test.columns.values[i]].values))

x_train.info()
from sklearn.feature_selection import SelectKBest,f_regression

selector = SelectKBest(score_func=f_regression,k=5)
selector.fit(x_train,y_train)
print(selector.get_support())
x_train_selected =pd.DataFrame({'OverallQual':x_train['OverallQual'],'ExterQual':x_train['ExterQual'],'GrLivArea':x_train['GrLivArea'],'GarageCars':x_train['GarageCars'],'GarageArea':x_train['GarageArea']})

x_test_selected = pd.DataFrame({'OverallQual':x_test['OverallQual'],'ExterQual':x_test['ExterQual'],'GrLivArea':x_test['GrLivArea'],'GarageCars':x_test['GarageCars'],'GarageArea':x_test['GarageArea']})

x_train_selected.head()
from sklearn.model_selection import train_test_split
xp_train,xp_test,yp_train,yp_test = train_test_split(x_train_selected,y_train,test_size=0.3,random_state=1)
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

forest=RandomForestRegressor()
svr=SVR()

parameters_forest={'n_estimators':[100,500,1000,3000],'max_depth':[3,6,12]}
parameters_svr={'C':[0.1,10,1000],'epsilon':[0.01,0.1,0.5]}
from sklearn.model_selection import GridSearchCV
# ランダムフォレスト
grid_forest = GridSearchCV(forest,parameters_forest)
grid_forest.fit(xp_train,yp_train)

# SVR
grid_svr = GridSearchCV(svr,parameters_svr)
grid_svr.fit(xp_train,yp_train)
from sklearn.metrics import mean_squared_error
yp_pred_svr=grid_forest.predict(xp_test)
print(mean_squared_error(yp_test,yp_pred_forest))
grid_forest.best_params_
best_forest = RandomForestRegressor(max_depth=6,n_estimators=100)
best_forest.fit(x_train_selected,y_train)
result = np.array(best_forest.predict(x_test_selected))

df_result =pd.DataFrame(result,columns=['SalePrice'])
df_result =pd.concat([test_id,df_result],axis=1)
df_result.to_csv('houseprices.csv',index=False)
result = np.array(best_forest.predict(x_test_selected))
# 提出用ファイルの作成
df_result=pd.DataFrame(result,columns=['SalePrice'])
df_result=pd.concat([test_id,df_result],axis=1)
df_result.to_csv('houseprices.csv', index=False)
