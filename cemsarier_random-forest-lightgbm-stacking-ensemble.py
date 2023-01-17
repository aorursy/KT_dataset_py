# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.options.display.max_columns = 15
pd.options.display.max_rows = 70
train = pd.read_csv('../input/bigmart-sales-data/Train.csv')
test = pd.read_csv('../input/bigmart-sales-data/Test.csv')
#data types
train.dtypes
#null check
train.info()
train.isnull().sum()/len(train)*100
len(pd.unique(train["Item_Identifier"])) #1559 different item
len(pd.unique(train["Outlet_Identifier"])) #10 outlets
### OUTLET SALES ANALYSIS ###
train.groupby(["Outlet_Identifier"])["Item_Outlet_Sales"].agg("mean").sort_values(ascending=False)
train.groupby(["Outlet_Identifier"])["Item_Weight"].agg("mean").sort_values(ascending=False)
train.groupby(["Outlet_Establishment_Year"])["Item_Outlet_Sales"].agg("mean").sort_values(ascending=False)
train.groupby(["Outlet_Establishment_Year"])["Item_Outlet_Sales"].agg("count").sort_values(ascending=False)
train.groupby(["Outlet_Size"])["Item_Outlet_Sales"].agg("mean").sort_values(ascending=False)
train.groupby(["Outlet_Size"])["Item_Outlet_Sales"].agg("count").sort_values(ascending=False)
train.groupby(["Outlet_Identifier","Item_Type"])["Item_Outlet_Sales"].agg("count").sort_values().groupby("Outlet_Identifier").tail(1)
train.groupby(["Outlet_Identifier","Item_Type"])["Item_Outlet_Sales"].agg("mean").sort_values().groupby("Outlet_Identifier").tail(1)
### ITEM ANALYSIS ###
plot_df = train.groupby(["Item_Type"])["Item_Weight"].agg('mean').sort_values(ascending=False).reset_index()
ticks = np.arange(0,len(plot_df))
labels = plot_df["Item_Type"]

plt.figure(figsize=(8,8))
plot_df.plot(kind='bar')
plt.xticks(ticks,labels)
plt.show()
train.groupby(["Item_Type","Item_Fat_Content"])["Item_Fat_Content"].agg("count").sort_values().groupby("Item_Type").tail(1)
plot_df = train.groupby(["Item_Type"])["Item_Visibility"].agg('mean').sort_values(ascending=False).reset_index()
ticks = np.arange(0,len(plot_df))
labels = plot_df["Item_Type"]

plt.figure(figsize=(8,8))
plot_df.plot(kind='bar')
plt.xticks(ticks,labels)
plt.show()
df = pd.concat([train,test],axis=0)
pd.value_counts(df[df["Outlet_Size"].isnull()]["Outlet_Identifier"])
pd.value_counts(df[df["Item_Weight"].isnull()]["Item_Identifier"])
#filling outlet size with similar rows wrt outlet type
df["Outlet_Size"] = df.groupby(["Outlet_Type"])["Outlet_Size"].transform(lambda x: x.fillna(x.mode()[0]))


#filling item weight with similar rows wrt Item_Type and Item_Fat_Content
df["Item_Weight"] = df.groupby(["Item_Type","Item_Fat_Content"])["Item_Weight"].transform(lambda x: x.fillna(x.median()))
#encoding ordinal column outlet size
df['Outlet_Size']=df['Outlet_Size'].replace({'Small':1,
                                             'Medium':2,
                                             'High':3})

pd.value_counts(df['Item_Fat_Content'])
#correcting mispelled column
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF': 'lf',
                                                         'Low Fat':'lf',
                                                         'low fat':'lf',
                                                         'reg':'reg',
                                                         'Regular':'reg'
                                                        })
items = df["Item_Identifier"]
df = df.drop(columns="Item_Identifier")
df = df.drop(columns="Outlet_Identifier")
df["Outlet_Year"] = 2020- df["Outlet_Establishment_Year"]
df = df.drop(columns="Outlet_Establishment_Year")

#Categorical value handling
df.columns[df.dtypes=='object']
df = pd.get_dummies(df,columns = df.columns[df.dtypes=='object'])
for col in df.columns[df.columns!="Item_Outlet_Sales"]:
    df[col] = (df[col]-df[col].min()) / (df[col].max()-df[col].min())
df['Item_MRP_X_Visi']=df['Item_MRP'] * df['Item_Visibility']
#df['Item_MRP_+_Visi']=df['Item_MRP'] + df['Item_Visibility']
df['Item_MRP_X_Weight']=df['Item_MRP'] * df['Item_Weight']
#df['Item_MRP_+_Weight']=df['Item_MRP'] + df['Item_Weight']
#df['Fat_Con_+_Weight']=df['Item_Fat_Content']+df['Item_Weight']
#df['Fat_Con_X_Weight']=df['Item_Fat_Content']*df['Item_Weight']
df['Total_Points']=df['Item_MRP']*df['Item_Visibility']*df['Item_Weight']
df['MrpPerUnit']=df['Item_MRP']/(df['Outlet_Size']+1)
train = df[0:len(train)]
test = df[len(train):]
test.drop(columns="Item_Outlet_Sales", inplace=True)
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
X = train.drop(columns="Item_Outlet_Sales")
y = train["Item_Outlet_Sales"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state=5)
### RANDOM FOREST ###
"""
rfr = RandomForestRegressor()

rf_grid = {'n_estimators' : [100,200,500,800,1000,1200],
           'max_depth' : [3,5,7,10,15,25,40,None],
           'min_samples_split':[2,4,6,10],
           'min_samples_leaf':[2,4,6,8]   
           }

search = RandomizedSearchCV(rfr,rf_grid,scoring='neg_mean_squared_error',cv=3, verbose=2, n_jobs=6, n_iter = 50)
search.fit(X,y)


print(search.best_params_)
print(search.best_estimator_)
print(search.cv_results_)
print(search.best_score_)
"""

rfr_best = RandomForestRegressor(n_estimators=1200, max_depth=5,min_samples_split=2,min_samples_leaf=2)
rfr_best.fit(X_train,y_train)
pred = rfr_best.predict(X_test)
np.sqrt(mean_squared_error(y_test, pred))
### LGBM REGRESSOR ###
"""
lgbmr = LGBMRegressor()
lgb_grid = {
    'n_estimators': [100, 200, 400, 500],
    'colsample_bytree': [0.9, 1.0],
    'max_depth': [5,10,15,20,25,35,None],
    'num_leaves': [20, 30, 50, 100],
    'reg_alpha': [1.0, 1.1, 1.2, 1.3],
    'reg_lambda': [1.0, 1.1, 1.2, 1.3],
    'min_split_gain': [0.2, 0.3, 0.4],
    'subsample': [0.8, 0.9, 1.0],
    'learning_rate': [0.05, 0.1]
}

search = RandomizedSearchCV(lgbmr,lgb_grid,scoring='neg_mean_squared_error',cv=3, verbose=2, n_jobs=6, n_iter = 100)
search.fit(X,y)

print(search.best_params_)
print(search.best_estimator_)
print(search.cv_results_)
print(search.best_score_)
"""

#Default parameters giving almost exact result as the grid search gives. So I used default ones.
lgb_best = LGBMRegressor()
lgb_best.fit(X_train,y_train)
pred2 = lgb_best.predict(X_test)
np.sqrt(mean_squared_error(y_test, pred2))
### STACKING ###

from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor

xgb = XGBRegressor()
lgbm = LGBMRegressor()
rf = RandomForestRegressor()

stack = StackingCVRegressor(regressors=(rf, lgbm, xgb),
                            meta_regressor=xgb, cv=3,
                            use_features_in_secondary=True,
                            store_train_meta_features=True,
                            shuffle=False,
                            random_state=42)

stack.fit(X_train, y_train)

X_test.columns = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29', 'f30', 'f31', 'f32', 'f33']
stack_pred = stack.predict(X_test)
np.sqrt(mean_squared_error(y_test, stack_pred))

### ENSEMBLE ###
pred_df = pd.DataFrame({'pred':pred, 'pred2':pred2, 'stack': stack_pred, 'target': y_test})

final_pred = pred*0.6 + pred2*0.2 + stack_pred*0.2

print(np.sqrt(mean_squared_error(y_test, final_pred)))
### SUBMISSION ###
pred1_test = rfr_best.predict(test)
pred2_test = lgb_best.predict(test)
test.columns = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29', 'f30', 'f31', 'f32', 'f33']
stack_test = stack.predict(test)
final_pred_test = pred1_test*0.6 + pred2_test*0.2 + stack_test*0.2

