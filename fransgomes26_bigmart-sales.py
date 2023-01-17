import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder,normalize

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score,confusion_matrix,accuracy_score

from xgboost import XGBRegressor,plot_importance

import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv('../input/bigmart/bigmart_train.txt')



data
data.isnull().sum()
import matplotlib.pyplot as plt

%matplotlib inline

data['Item_Fat_Content'].value_counts().plot.bar()
data['Item_Type'].value_counts().plot(kind='bar')
data['Outlet_Size'].value_counts().plot(kind='bar')
plt.figure(1)

plt.subplot(231)

data['Outlet_Location_Type'].value_counts().plot(kind='bar',figsize=(16,5))

plt.subplot(232)

data['Outlet_Type'].value_counts().plot(kind='bar',figsize=(16,5))

plt.subplot(233)

data['Outlet_Establishment_Year'].value_counts().plot(kind='bar',figsize=(16,5))
data['Item_Visibility'] = data['Item_Visibility'].replace({0.000000:np.nan})
data.isnull().sum()
data['Item_Weight'].fillna(data['Item_Weight'].mean(),inplace=True)

data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0],inplace=True)

data['Item_Visibility'].fillna(data['Item_Visibility'].mean(),inplace=True)
data.head()
categorial_features = data.select_dtypes(include=[np.object])

categorial_features.head()
numerical_features = data.select_dtypes(include=[np.number])

numerical_features.head()
data['Item_Fat_Content'].value_counts()
data['Item_Fat_Content']=data['Item_Fat_Content'].replace({"LF":"Low Fat","reg":"Regular","low fat":"Low Fat"})
data['Item_Identifier'] = data['Item_Identifier'].apply(lambda x: x[0:2])

data['Item_Identifier'] = data['Item_Identifier'].replace({"FD":"Food","DR":"Drinks","NC":"Non-Consumable"})
data['Item_Identifier'].value_counts()
data['Item_Identifier'].value_counts().plot.bar()


plt.scatter(data['Item_Type'],data['Item_Outlet_Sales'],c='C')

plt.xlabel("Item Type")

plt.ylabel("Item Outlet Sales")

plt.show()
data.head()
data['Item_Visibility'] = data['Item_Visibility']**(1/3)
le = LabelEncoder()

data['Item_Fat_Content']=le.fit_transform(data['Item_Fat_Content'])

data['Outlet_Size']=le.fit_transform(data['Outlet_Size'])
data.head()
data = pd.get_dummies(data,columns=['Item_Identifier','Item_Type','Outlet_Identifier','Outlet_Location_Type','Outlet_Type'])
data.columns
data.head(10)
test_data = pd.read_csv('../input/bigmart/bigmart_test.txt')
test_data.head()
data['Item_Outlet_Sales'] = data['Item_Outlet_Sales']**(1/3)

sns.distplot(data['Item_Outlet_Sales'])

colors=(0,0,0)

plt.scatter(data['Item_Fat_Content'],data['Item_Outlet_Sales'],c='C')

plt.xlabel("Item Type")

plt.ylabel("Item Outlet Sales")

plt.show()
colors=(0,0,0)

plt.scatter(data['Outlet_Size'],data['Item_Outlet_Sales'],c='C')

plt.xlabel("Item Type")

plt.ylabel("Item Outlet Sales")

plt.show()
test_data.isnull().sum()
test_data['Item_Weight'].fillna(test_data['Item_Weight'].mean(),inplace=True)

test_data['Outlet_Size'].fillna(test_data['Outlet_Size'].mode()[0],inplace=True)
test_data['Item_Fat_Content']=test_data['Item_Fat_Content'].replace({"LF":"Low Fat","reg":"Regular","low fat":"Low Fat"})
test_data['Item_Identifier'] = test_data['Item_Identifier'].apply(lambda x: x[0:2])

test_data['Item_Identifier'] = test_data['Item_Identifier'].replace({"FD":"Food","DR":"Drinks","NC":"Non-Consumable"})
test_data.head()
le = LabelEncoder()

test_data['Item_Fat_Content']=le.fit_transform(test_data['Item_Fat_Content'])

test_data['Outlet_Size']=le.fit_transform(test_data['Outlet_Size'])
test_data = pd.get_dummies(test_data,columns=['Item_Identifier','Item_Type','Outlet_Identifier','Outlet_Location_Type','Outlet_Type'])
test_data.head()
x = data.drop(['Item_Outlet_Sales'],axis=1)

x.head()
y = data['Item_Outlet_Sales']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.3)
model = LinearRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)
r2score = r2_score(y_test,y_pred)

r2score


sns.distplot(data['Item_MRP'])

sns.distplot(data['Item_Visibility'])
sns.distplot(data['Item_Weight'])
model = XGBRegressor(n_estimators=100)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

r2score = r2_score(y_test,y_pred)

print(r2score)

plot_importance(model)

model = RandomForestRegressor(n_estimators=100,random_state=1)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

r2score = r2_score(y_test,y_pred)

r2score
matrix = data.corr() 

f, ax = plt.subplots(figsize=(15, 15)) 

sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");
data['Outlet_Size']