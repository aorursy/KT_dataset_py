import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd
df1 = pd.read_csv('/kaggle/input/bigmart-sales-dataset/Train_UWu5bXk.txt',sep=',')
df1.head()
df1.shape
df1.columns
df1.isnull().sum()
df1['Item_Type_Category'] = df1['Item_Identifier'].apply(lambda x: x[0:2])
df1['Item_Type_Category'] = df1['Item_Type_Category'].map({'FD' : 'FOOD','NC' : 'Non-Consumable','DR' : 'Drinks'})
df1['data']  = 'Train'
df1.columns
df2 = pd.read_csv('/kaggle/input/bigmart-sales-dataset/Test_u94Q5KV.txt',sep=',')
df2['Item_Type_Category'] = df2['Item_Identifier'].apply(lambda x: x[0:2])

df2['Item_Type_Category'] = df2['Item_Type_Category'].map({'FD' : 'FOOD','NC' : 'Non-Consumable','DR' : 'Drinks'})
df2['data'] = 'Test'
df2.columns
#df2['Item_Outlet_Sales'] = np.nan
df = pd.concat([df1,df2],axis = 0)
df = df.drop(['Item_Identifier','Item_Type'],axis = 1)
df.columns
df.Item_Visibility = df.Item_Visibility.replace(0,np.mean(df.Item_Visibility))
df.shape
num_var = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['int64','float64']]

cat_var = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['object']]
print(num_var)

print(cat_var)
df.columns
df[num_var].isnull().sum()
df.Item_Weight = df.Item_Weight.fillna(df.Item_Weight.mean())
df[num_var].isnull().sum()
df[cat_var].isnull().sum()
df.Outlet_Size.mode().iloc[0]
df.Outlet_Size = df.Outlet_Size.fillna(df.Outlet_Size.mode().iloc[0])
df[cat_var].isnull().sum()
df[cat_var].columns
df.columns
df[cat_var].columns
for i in df[cat_var].columns:

    print(df[i].unique())
df['Item_Fat_Content'].loc[(df['Item_Fat_Content']=='low fat') | (df['Item_Fat_Content']=='LF')] = 'Low Fat'
df['Item_Fat_Content'].loc[(df['Item_Fat_Content']=='reg')] = "Regular"
df.Item_Fat_Content.unique()
df[cat_var].head()
df.columns
df['No_Of_Years_Before_Established'] = 2019 - df.Outlet_Establishment_Year
df = df.drop('Outlet_Establishment_Year',axis = 1)
cat_var = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['object']]
df[cat_var].columns[:-1]
df = pd.get_dummies(df,columns = df[cat_var].columns[:-1],drop_first = True)
df.columns
df_train = df[df['data'] == 'Train']
df_train = df_train.drop('data',axis = 1)
df_test = df[df['data']== 'Test']
df_test = df_test.drop('data',axis = 1)
df_test = df_test.drop('Item_Outlet_Sales',axis = 1)
X = df_train.drop('Item_Outlet_Sales',axis =1)
y = df_train.Item_Outlet_Sales
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.3,random_state=1)
from sklearn.linear_model import LinearRegression,Ridge,Lasso
lm = LinearRegression()
lm.fit(X_train,y_train)
y_pred = lm.predict(X_val)
from sklearn.metrics import mean_squared_error,r2_score
np.sqrt(mean_squared_error(y_val,y_pred))
r2_score(y_val,y_pred)
predict = lm.predict(X_train)
r2_score(y_train,predict)
#r2_score(y_val,RFE_y_pred)
#predict = rfe_6.predict(X_train)
#r2_score(y_train,predict)
from sklearn.model_selection import GridSearchCV
lambdas = np.linspace (1,100,100)

params = {'alpha':lambdas}

model = Ridge(fit_intercept = True)
Ridge = GridSearchCV(model,param_grid = params , cv = 10, scoring = 'neg_mean_absolute_error')
Ridge.fit(X_train,y_train)
Ridge.best_estimator_
Ridge_model = Ridge.best_estimator_
Ridge_y_pred = Ridge_model.predict(X_val)
np.sqrt(mean_squared_error(y_val,Ridge_y_pred))
r2_score(y_val,Ridge_y_pred)
predict = Ridge_model.predict(X_train)
r2_score(y_train,predict)
lambdas = np.linspace (1,100,100)

params = {'alpha':lambdas}

model_1 = Lasso(fit_intercept = True)
Lasso = GridSearchCV(model_1,param_grid = params , cv = 10, scoring = 'neg_mean_absolute_error')
Lasso.fit(X_train,y_train)
Lasso_model = Lasso.best_estimator_
Lasso_y_pred = Lasso_model.predict(X_val)
np.sqrt(mean_squared_error(y_val,Lasso_y_pred))
r2_score(y_val,Lasso_y_pred)
predict = Lasso_model.predict(X_train)
r2_score(y_train,predict)
from sklearn import tree
reg = tree.DecisionTreeRegressor(max_depth=3)

reg.fit(X_train,y_train)
y_pred = reg.predict(X_val)
np.sqrt(mean_squared_error(y_val,y_pred))
r2_score(y_val,y_pred)
predict = reg.predict(X_train)
r2_score(y_train,predict)
param_grid = {'criterion':['mse','mae'],

             'min_samples_split':[2,10,20],

             'max_depth':[None,2,5,10],

             'min_samples_leaf':[1,5,10],

             'max_leaf_nodes':[None,5,10,20]}
from sklearn.model_selection import GridSearchCV

dt = tree.DecisionTreeRegressor()

reg = GridSearchCV(dt,param_grid , cv=2)

reg.fit(X_train,y_train)
reg.best_estimator_
DT_model = reg.best_estimator_
dt_y_pred = DT_model.predict(X_val)
np.sqrt(mean_squared_error(y_val,dt_y_pred))
r2_score(y_val,dt_y_pred)
predict = DT_model.predict(X_train)
r2_score(y_train,predict)
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor()

reg.fit(X_train,y_train)
RF_y_pred = reg.predict(X_val)
np.sqrt(mean_squared_error(y_val,RF_y_pred))
r2_score(y_val,RF_y_pred)
predict = reg.predict(X_train)
r2_score(y_train,predict)
lambdas = np.arange(1,1000,100)
param_grid = {'n_estimators': lambdas,

              'max_features': ['auto','sqrt']}

             
from sklearn.model_selection import GridSearchCV

rf = RandomForestRegressor()

reg = GridSearchCV(rf,param_grid , cv=2)

reg.fit(X_train,y_train)
model = reg.best_estimator_
model.fit(X_train,y_train)
RF_y_pred = model.predict(X_val)
np.sqrt(mean_squared_error(y_val,RF_y_pred))
r2_score(y_val,RF_y_pred)
predict = model.predict(X_train)
r2_score(y_train,predict)
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()

model.fit(X_train,y_train)
KNN_y_pred = model.predict(X_val)
r2_score(y_val,KNN_y_pred)
predict = model.predict(X_train)
r2_score(y_train,predict)
params = np.arange(1,101,10)



param_grid = {'n_neighbors':params,

             'metric':['minkowski','manhattan','euclidean']}



knn = KNeighborsRegressor()
reg = GridSearchCV(knn,param_grid , cv=2)

reg.fit(X_train,y_train)
model = reg.best_estimator_
model.fit(X_train,y_train)
KNN_y_pred = model.predict(X_val)
r2_score(y_val,KNN_y_pred)
predict = model.predict(X_train)
r2_score(y_train,predict)
df_test.head()
Linear_Ridge = Ridge_model.predict(df_test)
DT_GridSearch = DT_model.predict(df_test)
Test_predicted = (Linear_Ridge+DT_GridSearch)/2
df_train.Item_Outlet_Sales.mean()
Test_predicted.mean()