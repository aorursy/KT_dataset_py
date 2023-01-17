# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
test=pd.read_csv("/kaggle/input/big-mart-sales-dataset/Test_u94Q5KV.csv")
train=pd.read_csv("/kaggle/input/big-mart-sales-dataset/Train_UWu5bXk.csv")
train.head()
train.info()
train.describe()
def idrop(a):
    a.drop("Item_Identifier",axis=1,inplace=True)
    a.drop("Outlet_Identifier",axis=1,inplace=True)
idrop(train)
def split(a):
    num=a.select_dtypes(include=[np.number]) 
    cat=a.select_dtypes(exclude=[np.number])
   
    return num,cat
    
train.columns
import numpy as np
def missvalue(a):
    l=['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type',
       'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size',
       'Outlet_Location_Type', 'Outlet_Type', 'Item_Outlet_Sales']
    for i in l:
        if a[i].dtypes!='O':
            med=a[i].median()
            a[i].fillna(med,inplace=True)
        else:
            m=a[i].value_counts().index[0]
            a[i].fillna(m,inplace=True)  
missvalue(train)
x,y=split(train)
y.head()
for t in y.columns:
    print(t+":")
    print(y[t].value_counts())
    print("-"*80)
x.columns
for t in x:
    
    print(t+":")
    sns.set(style="darkgrid")
    plt.figure()
    x[t].hist()
    plt.show()
x.columns

columns=x.columns
columns=['Item_Weight', 'Item_Visibility', 'Item_MRP',
       'Outlet_Establishment_Year', 'Item_Outlet_Sales']
sns.heatmap(x.corr())
x.corr()
for i in y.columns:
      sns.set(style="darkgrid")
      plt.figure()
    
      chart=sns.countplot(i,data=y)
      chart.set_xticklabels(chart.get_xticklabels(),rotation=90)
pd.pivot_table(train,index="Item_Fat_Content",values="Item_Outlet_Sales").sort_values("Item_Outlet_Sales",ascending=False)
y.columns
pd.pivot_table(train,index='Item_Type',values="Item_Outlet_Sales").sort_values("Item_Outlet_Sales",ascending=False)
pd.pivot_table(train,index='Outlet_Size',values="Item_Outlet_Sales").sort_values("Item_Outlet_Sales",ascending=False)
y.columns
pd.pivot_table(train,index='Outlet_Location_Type',values="Item_Outlet_Sales").sort_values("Item_Outlet_Sales",ascending=False)
pd.pivot_table(train,index='Outlet_Type',values="Item_Outlet_Sales").sort_values("Item_Outlet_Sales",ascending=False)
pd.pivot_table(train,index='Item_Type',values="Item_Weight").sort_values("Item_Weight",ascending=False)
train.isnull().sum()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap="viridis")
corr_matrix=train.corr()
corr_matrix["Item_Outlet_Sales"].sort_values(ascending=False)
def add(train):
    train["mrp per weight"]=train["Item_MRP"]/train["Item_Weight"]
    train["visibility per mrp"]=train["Item_Visibility"]/train["Item_MRP"]
    train["mrp per year"]=train["Item_MRP"]/train["Outlet_Establishment_Year"]
add(train)
corr_matrix=train.corr()
corr_matrix["Item_Outlet_Sales"].sort_values(ascending=False)
def fdrop(a):
    a.drop("Item_Weight",axis=1,inplace=True)
    a.drop("Outlet_Establishment_Year",axis=1,inplace=True)
    a.drop("Item_Visibility",axis=1,inplace=True)
fdrop(train)
corr_matrix=train.corr()
corr_matrix["Item_Outlet_Sales"].sort_values(ascending=False)
def split(a):
    num=a.select_dtypes(include=[np.number]) 
    cat=a.select_dtypes(exclude=[np.number])
    cat=pd.get_dummies(cat)
    return num,cat
    
t,q=split(train)
y.head()
q.head()
from sklearn.preprocessing import StandardScaler
labels=t["Item_Outlet_Sales"].values
t.drop(["Item_Outlet_Sales"],axis=1,inplace=True)
t.head()
scaler=StandardScaler()
def scaling(x,y):
     f_s=scaler.fit_transform(x.values)
     q_s=y.values
     vk=np.concatenate((f_s,q_s),axis=1)
     return vk
features=scaling(t,q)
from sklearn.model_selection import train_test_split
import numpy
numpy.random.seed(1234)
(x_train,x_test,y_train,y_test) = train_test_split(features,labels, train_size=0.75, random_state=42)
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x_train,y_train)
pred=lin_reg.predict(x_test)
from sklearn.metrics import mean_squared_error
lin_mse=mean_squared_error(y_test,pred)
lin_rmse=np.sqrt(lin_mse)
lin_rmse
from sklearn.tree import DecisionTreeRegressor
tree_reg=DecisionTreeRegressor()
tree_reg.fit(x_train,y_train)
pred=tree_reg.predict(x_test)
lin_mse=mean_squared_error(y_test,pred)
lin_rmse=np.sqrt(lin_mse)
lin_rmse
from sklearn.model_selection import cross_val_score
scores=cross_val_score(tree_reg,features,labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)

rmse_scores
rmse_scores.mean()
scores=cross_val_score(tree_reg,x_train,y_train,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)
rmse_scores
scores=cross_val_score(lin_reg,x_train,y_train,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)
rmse_scores
from sklearn.ensemble import RandomForestRegressor
forest_reg=RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
forest_reg.fit(x_train,y_train)
pred=forest_reg.predict(x_test)
lin_mse=mean_squared_error(y_test,pred)
lin_rmse=np.sqrt(lin_mse)
lin_rmse
scores=cross_val_score(forest_reg,x_train,y_train,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)
rmse_scores.mean ()
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
param_grid = [
 {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
 {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
 ]
forest_reg=RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
 scoring='neg_mean_squared_error')
grid_search.fit(x_train,y_train)
grid_search.best_params_
final_model = grid_search.best_estimator_
final_prediction=final_model.predict(x_test)
final_mse = mean_squared_error(y_test, final_prediction)
final_rmse = np.sqrt(final_mse) 
final_rmse
test.head()
p=test["Item_Identifier"]
w=test['Outlet_Identifier']
result = pd.concat([p, w], axis=1, join='inner')

idrop(test)
import numpy as np
def miss(a):
    l=['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type',
       'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size',
       'Outlet_Location_Type', 'Outlet_Type']
    for i in l:
        if a[i].dtypes!='O':
            med=a[i].median()
            a[i].fillna(med,inplace=True)
        else:
            m=a[i].value_counts().index[0]
            a[i].fillna(m,inplace=True)   
miss(test)
add(test)
fdrop(test)
t,q=split(test)
feat=scaling(t,q)
ped=grid_search.best_estimator_.predict(feat)
ped
