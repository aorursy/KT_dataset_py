import numpy as np # linear algebra

import pandas as pd # data processing

import seaborn as sns

import matplotlib.pyplot as plt

import xgboost as xgb

from scipy import stats





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import warnings

warnings.filterwarnings("ignore")
## Importing Datasets

train_data =  pd.read_csv('../input/big-mart-sales-prediction/train_v9rqX0R.csv')

test_data = pd.read_csv('../input/big-mart-sales-prediction/test_AbJTz2l.csv')
train_data.head()
test_data.head()
print(train_data.shape)

print(test_data.shape)
train_data.describe().T
train_data.isnull().sum()
test_data.isnull().sum()
train_data['source'] = 'train'

test_data['source'] = 'test'

df = pd.concat([train_data,test_data], ignore_index=True)
df.isnull().sum()
for i in train_data.describe().columns:

    sns.distplot(train_data[i].dropna())

    plt.show()
for i in train_data.describe().columns:

    sns.boxplot(train_data[i].dropna())

    plt.show()
plt.figure(figsize=(15,10))

sns.countplot(train_data.Item_Type)

plt.xticks(rotation=90)
train_data.Item_Type.value_counts()
plt.figure(figsize=(10,8))

sns.countplot(train_data.Outlet_Size)

plt.show()
plt.figure(figsize=(10,8))

sns.countplot(train_data.Outlet_Location_Type)
train_data.Outlet_Location_Type.value_counts()
plt.figure(figsize=(10,8))

sns.countplot(train_data.Outlet_Type)

plt.xticks(rotation=90)
train_data.Outlet_Type.value_counts()
plt.figure(figsize=(10,8))

plt.xlabel("Item_Weight")

plt.ylabel("Item_Outlet_Sales")

plt.title("Itam Weight and Item Outlet Sales")

sns.scatterplot(x='Item_Weight', y='Item_Outlet_Sales', hue='Item_Type',size='Item_Weight',data=train_data)
plt.figure(figsize=(13,9))

plt.xlabel("Item_Visibility")

plt.ylabel("Item_Outlet_Sales")

plt.title("Item Visibility and Item Outlet Sales",fontsize=15)

sns.scatterplot(x="Item_Visibility", y="Item_Outlet_Sales", hue="Item_Type", size= 'Item_Weight',data=train_data)
plt.figure(figsize=(12,7))

plt.xlabel("Item_visibility")

plt.ylabel("Maximum Retail Price")

plt.title("Item_visibility and Maximum Retail Price")

plt.plot(train_data.Item_Visibility, train_data.Item_MRP, ".", alpha=0.3)
Outlet_Type_pivot = train_data.pivot_table(index='Outlet_Type',values='Item_Outlet_Sales', aggfunc=np.median)



Outlet_Type_pivot.plot(kind='bar', color='red', figsize=(12,8))

plt.xlabel("Outlet_Type")

plt.ylabel("Item_Outlet_Sales")

plt.title("Impact of Outlet_type on Item_Outlet_Sales")

plt.show()
Item_Fat_Content_pivot = train_data.pivot_table(index='Item_Fat_Content', values='Item_Outlet_Sales', aggfunc=np.median)



Item_Fat_Content_pivot.plot(kind='bar',color='blue', figsize=(12,7))

plt.xlabel("Item_Fat_Content")

plt.ylabel("Item_Outlet_Sales")

plt.title("Impact of Item_Fat_Content on Item_outlet_Sales")

plt.xticks(rotation=0)

plt.show()
df['Item_Fat_Content'].value_counts()
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'})
df['Item_Fat_Content'].value_counts()
train_data['Item_Fat_Content'] = train_data['Item_Fat_Content'].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'})
Item_Fat_Content_pivot = train_data.pivot_table(index='Item_Fat_Content', values='Item_Outlet_Sales', aggfunc=np.median)



Item_Fat_Content_pivot.plot(kind='bar',color='blue', figsize=(12,7))

plt.xlabel("Item_Fat_Content")

plt.ylabel("Item_Outlet_Sales")

plt.title("Impact of Item_Fat_Content on Item_outlet_Sales")

plt.xticks(rotation=0)

plt.show()
train_data.corr()
plt.figure(figsize=(35,15))

sns.heatmap(train_data.corr(), vmax=1,square=True, cmap='viridis')

plt.title("Correlation between different attributes")
df['Item_Weight'].mean()
df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
df.isnull().sum()
df['Outlet_Size'].value_counts()
df['Outlet_Size'].fillna("Medium", inplace=True)
df.isnull().sum()    ## now we dont have any null values
print(train_data.shape)

print(df.shape)
df[df['Item_Visibility']==0]['Item_Visibility'].count()
df['Item_Visibility'].fillna(df['Item_Visibility'].median(), inplace=True)                ## 0 is replaced by Medium
df['Outlet_Establishment_Year'].value_counts()
df['Outlet_Years'] = 2009 - df['Outlet_Establishment_Year']

df['Outlet_Years'].describe()
df['Item_Type'].value_counts()

df['Item_Identifier'].value_counts()
##Changing only the first 2 characters (i,e the category ID)

df['New_Item_Type'] = df['Item_Identifier'].apply(lambda x: x[0:2])
## Rename them to more intuitive categories::

df['New_Item_Type'] = df['New_Item_Type'].map({'FD':'Food','NC':'Non_Consumable','DR':'Drinks'})



df['New_Item_Type'].value_counts()
## Mark non-consumable as separate category in Low-fat.



df.loc[df['New_Item_Type']=="Non_Consumable","Item_Fat_Content"] = "Non-Edible"

df['Item_Fat_Content'].value_counts()
item_visib_avg = df.pivot_table(values='Item_Visibility', index='Item_Identifier')
item_visib_avg
function = lambda x: x['Item_Visibility']/item_visib_avg['Item_Visibility'][item_visib_avg.index==x['Item_Identifier']][0]



df['item_visib_avg'] = df.apply(function, axis=1).astype(float)
df.head()
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()



df['Outlet'] = label.fit_transform(df['Outlet_Identifier'])

varib = ['Item_Fat_Content','Outlet_Location_Type', 'Outlet_Size','New_Item_Type','Outlet_Type','Outlet']



for i in varib:

    df[i] = label.fit_transform(df[i])

    
df.head()
df = pd.get_dummies(df, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','New_Item_Type','Outlet'])

df.dtypes
df.drop(['Item_Type','Outlet_Establishment_Year'], axis=1, inplace=True)
train_data = df.loc[df['source']=='train']

test_data = df.loc[df['source']=='test']
train_data.drop(['source'], axis=1,inplace=True)
test_data.drop(['Item_Outlet_Sales','source'], axis=1,inplace=True)
X_train = train_data.drop(['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier'],axis=1).copy()

y_train = train_data['Item_Outlet_Sales']

X_test = test_data.drop(['Item_Identifier','Outlet_Identifier'], axis=1).copy()
from sklearn.linear_model import LinearRegression

lr = LinearRegression(normalize=True)



lr.fit(X_train , y_train)
lr_pred = lr.predict(X_test)
lr_pred
lr_accuracy = round(lr.score(X_train,y_train) * 100)

lr_accuracy
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)



tree.fit(X_train,y_train)



tree_pred = tree.predict(X_test)

tree_pred
tree_accuracy = round(tree.score(X_train, y_train)*100)

tree_accuracy
from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf = 100,n_jobs=4)



rf.fit(X_train,y_train)



rf_pred = rf.predict(X_test)



rf_accuracy = round(rf.score(X_train,y_train) * 100)

rf_accuracy
from xgboost import XGBRegressor



model = XGBRegressor(n_estimators=1000, learning_rate = 0.05)

model.fit(X_train,y_train)
pred = model.predict(X_test)

pred
model.score(X_train,y_train)*100