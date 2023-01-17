import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df_train=pd.read_csv("../input/bigmart-sales-data/Train.csv")

df_test=pd.read_csv("../input/bigmart-sales-data/Test.csv")
df_train.head()
df_test.head()
df_train.info()
df_test.info()
df_train.shape
df_train.describe()
df_test.shape
df_train.duplicated().sum()
df_train.corr()
# dealing with null values

df_train['Item_Weight']=df_train['Item_Weight'].replace(np.NaN,df_train['Item_Weight'].mean())

df_train=df_train.dropna(subset=['Outlet_Size'])
df_test['Item_Weight']=df_test['Item_Weight'].replace(np.NaN,df_test['Item_Weight'].mean())

df_test=df_test.dropna(subset=['Outlet_Size'])
df_train['Item_Fat_Content']=df_train['Item_Fat_Content'].replace({'low fat':'Low Fat','reg':'Regular','LF':'Low Fat'})
df_train.info()
df_test.info()
df_train['Item_Type'].value_counts()
plt.figure(figsize=(8,8))

sns.countplot(df_train['Item_Type'])

plt.xticks(rotation=90);

plt.title("count of each Item")
df_train['Item_Fat_Content'].value_counts()
sns.countplot(df_train['Item_Fat_Content'])

plt.xticks(rotation=90);

plt.title("Fat type")
df_train['Outlet_Location_Type'].value_counts()
sns.countplot(df_train['Outlet_Location_Type'])

plt.xticks(rotation=90);

plt.title("Fat type")
df_train['Outlet_Size'].value_counts()
sns.countplot(df_train['Outlet_Size'])

plt.xticks(rotation=90);

plt.title("size of outlet")
plt.figure(figsize=(20,20))

sns.regplot(x=df_train['Item_Weight'],y=df_train['Item_MRP']);

plt.xlabel("Item_Weight")

plt.ylabel("Item_MRP")

plt.title("Item_Weight vs Item_MRP")
plt.figure(figsize=(20,20))

sns.regplot(x=df_train['Item_Visibility'],y=df_train['Item_MRP']);

plt.xlabel("Item_Visibility")

plt.ylabel("Item_MRP")

plt.title("Item_Visibility vs Item_MRP")
predictors=['Item_Weight','Item_Fat_Content','Item_Visibility','Item_MRP','Outlet_Size','Outlet_Location_Type','Outlet_Type']

target=['Item_Outlet_Sales']
train=df_train.copy()

test=df_test.copy()
train.head()
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

lb=LabelEncoder()

categ=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type']

for i in categ:

    train[i]=lb.fit_transform(train[i])

train_data=pd.get_dummies(train, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type'])
train_data.head()
train_data=train_data.drop(['Item_Identifier','Outlet_Identifier','Outlet_Establishment_Year'],axis=1)

train_data.head()
train_data.info()
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

lb=LabelEncoder()

categ=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type']

for i in categ:

    test[i]=lb.fit_transform(test[i])

test.head()
test_data=pd.get_dummies(test, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type'])
test.head()
test_data=test_data.drop(['Item_Identifier','Outlet_Identifier','Outlet_Establishment_Year'],axis=1)

test_data=test_data.drop(['Item_Fat_Content_3','Item_Fat_Content_4'],axis=1)
test_data=test_data.drop(['Item_Fat_Content_2'],axis=1)
test_data.head()
test_data.info()
X_train=train_data.drop(['Item_Outlet_Sales'],axis=1)

y_train=train_data.Item_Outlet_Sales
from sklearn.linear_model import LinearRegression

reg=LinearRegression()

reg.fit(X_train,y_train)
prediction=reg.predict(test_data)
prediction
predicted_data=pd.DataFrame({'Item_Identifier':test['Item_Identifier'],'Item_Outlet_Sales':prediction})
predicted_data.head()
predicted_data.to_csv("Predicted_outlet_sales.csv",index=False)