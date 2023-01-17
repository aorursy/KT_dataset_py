import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
trainData = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
testData = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sampleData = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
trainData.head()
trainData.info()
sns.distplot(trainData.SalePrice)
plt.show()
SalePrice = trainData.SalePrice
SalePrice_log = np.log(SalePrice)
sns.distplot(SalePrice_log)
plt.show()
numeric_var = trainData.select_dtypes(exclude='object')
trainData.drop(columns = ['SalePrice']).corrwith(SalePrice,axis=0).plot.bar(figsize = (12,12))
plt.figure(figsize=(12,12))
fig = sns.heatmap(trainData.corr(), linewidth = 0.3)
train_id = trainData.Id
test_id = testData.Id
trainData.set_index('Id')
testData.set_index('Id')
data = pd.concat([trainData, testData])
with pd.option_context('display.max_rows',None,'display.max_columns',None):
    display(data.isnull().sum())
data.drop(columns=['Alley','PoolQC','Fence','MiscFeature'], inplace = True)
num_col = data.select_dtypes(exclude='object').columns
obj_cat = data.select_dtypes(include='object').columns
for i in num_col:
    data[i] = data[i].fillna(data[i].mean())
for i in obj_cat:
    data[i] = data[i].fillna(data[i].mode()[0])
with pd.option_context('display.max_rows',None,'display.max_columns',None):
    display(data.isnull().sum())
data.drop(columns = ['SalePrice'], inplace = True)
data.drop(['GrLivArea', '1stFlrSF', 'OverallQual', 'GarageCars'], axis=1, inplace=True)
data = data.set_index('Id')
data
data2 = data.copy(deep=True)
obj_col = data2.select_dtypes(include='object').columns
data_dummy = pd.get_dummies(data2, columns = obj_col, drop_first=True)
train = data_dummy.iloc[:train_id.shape[0],:]
test = data_dummy.iloc[train_id.shape[0]:,:]
train
test
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = pd.DataFrame(sc.fit_transform(train), columns = train.columns.values)
x_test = pd.DataFrame(sc.transform(test), columns = test.columns.values)
x
from sklearn.model_selection import train_test_split

x_train,x_test_1, y_train, y_test_1 = train_test_split(x,SalePrice_log,test_size = 0.2, random_state = 0)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor()
regressor.fit(x_train,y_train)
y_pred= regressor.predict(x_test_1)
from sklearn.metrics import r2_score,mean_squared_error
r2_score(y_test_1,y_pred)
mean_squared_error(y_test_1,y_pred)
param = {'max_depth': [3,5,8],
        'n_estimators': [100,300,500],
        'criterion': ['mse', 'mae'],
        'max_features': ['sqrt','log2','auto']}
from sklearn.model_selection import RandomizedSearchCV
regressor = RandomForestRegressor()
random = RandomizedSearchCV(estimator = regressor, param_distributions=param, n_iter = 5,scoring='neg_mean_squared_error',n_jobs=-1,cv=5)
random.fit(x_train,y_train)
random.best_params_
regressor2 = RandomForestRegressor(n_estimators=100,
 max_features='auto',
 max_depth=8,
 criterion='mse')
regressor2.fit(x_train,y_train)
y_pred = regressor2.predict(x_test_1)
r2_score(y_test_1,y_pred)
prediction = regressor.predict(x_test)
pred = np.exp(prediction)
Id = pd.DataFrame(test_id, columns = ['Id'])

predi = pd.DataFrame(pred, columns = ['SalePrice'])
result = pd.concat([Id,predi],axis = 1)

