# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
 
data=pd.read_csv("../input/bigmart-sales-data/Test.csv")
data.head(10)
X=data.iloc[:,:10]
Y=data.iloc[:,-1]

data.shape
data.isnull().values.any()
data.isnull().sum()
test=data
test.head(2)
train = pd.read_csv('../input/bigmart-sales-data/Train.csv')
train.head(2)
train.shape,test.shape
train.columns
test.columns
train['source']='train'
test['source']='test'
test['Item_Outlet_Sales']=0

data = pd.concat([train, test], sort = False)
print(train.shape, test.shape, data.shape)
data['Item_Outlet_Sales'].describe()
sns.distplot(data['Item_Outlet_Sales'])
data.dtypes
categorial_f=data.select_dtypes(include=[np.object])
categorial_f.head(2)
numerical_features = data.select_dtypes(include=[np.number])
numerical_features.head(2)
data.apply(lambda x: sum(x.isnull()))
data.apply(lambda x:len(x.unique()))
for col in categorial_f:
    print('\n%s column:' %col)
    print(data[col].value_counts())
plt.figure(figsize = (10,9))

plt.subplot(311)
sns.boxplot(x='Outlet_Size', y='Item_Outlet_Sales', data=data, palette="Set1")

plt.subplot(312)
sns.boxplot(x='Outlet_Location_Type', y='Item_Outlet_Sales', data=data, palette="Set1")

plt.subplot(313)
sns.boxplot(x='Outlet_Type', y='Item_Outlet_Sales', data=data, palette="Set1")

plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 1.5)

plt.show()

plt.figure(figsize = (14,9))

plt.subplot(211)
ax = sns.boxplot(x='Outlet_Identifier', y='Item_Outlet_Sales', data=data, palette="Set1")
ax.set_title("Outlet_Identifier vs. Item_Outlet_Sales", fontsize=15)
ax.set_xlabel("", fontsize=12)
ax.set_ylabel("Item_Outlet_Sales", fontsize=12)

plt.subplot(212)
ax = sns.boxplot(x='Item_Type', y='Item_Outlet_Sales', data=data, palette="Set1")
ax.set_title("Item_Type vs. Item_Outlet_Sales", fontsize=15)
ax.set_xlabel("", fontsize=12)
ax.set_ylabel("Item_Outlet_Sales", fontsize=12)

plt.subplots_adjust(hspace = 0.9, top = 0.9)
plt.setp(ax.get_xticklabels(), rotation=45)

plt.show()
item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')

missing_values = data['Item_Weight'].isnull()
print('Missing values: %d' %sum(missing_values))

data.loc[missing_values,'Item_Weight']  = data.loc[missing_values,'Item_Identifier'].apply(lambda x: item_avg_weight.at[x,'Item_Weight'])
print('Missing values after immputation %d' %sum(data['Item_Weight'].isnull()))
plt.figure(figsize = (14,9))

plt.subplot(211)

plt.subplot(211)
ax = sns.boxplot(x='Outlet_Establishment_Year', y='Item_Outlet_Sales', data=data, palette="Set1")
ax.set_title("Outlet_Establishment_Year with. Item_Outlet_Sales", fontsize=15)
ax.set_xlabel("Outlet_Establishment_Year", fontsize=12)
ax.set_ylabel("Item_Outlet_Sales", fontsize=12)
data.Item_Weight.isnull().sum()

data['Item_Visibility'].replace(0.00000,np.nan)#first fill by nam for simplicity
data['Item_Visibility'].fillna(data.groupby('Item_Identifier')['Item_Visibility'].transform('mean'))
data['Item_Visibility'].fillna(data.groupby('Item_Identifier')['Item_Visibility'].transform('mean'))
data.Item_Visibility.isnull().sum()
data['Item_Fat_Content']=data['Item_Fat_Content'].replace({'low fat':'Low Fat','reg':'Regular','LF':'Low Fat'})
data.Item_Fat_Content.value_counts()
data['Outlet_Size']
from sklearn.preprocessing import LabelEncoder
number=LabelEncoder()
var_mod=['Item_Fat_Content','Outlet_Location_Type','Outlet_Type','Item_Type']
data['Outlet']=number.fit_transform(data['Outlet_Identifier'])
data['Identifier']=number.fit_transform(data['Item_Identifier'])
data['Item_Fat']=number.fit_transform(data['Item_Fat_Content'])
data['Outlet_Typ']=number.fit_transform(data['Outlet_Type'])
data['Outlet_Location']=number.fit_transform(data['Outlet_Location_Type'])
data['Item_Typ']=number.fit_transform(data['Item_Type'])
data["Year"]=number.fit_transform(data['Outlet_Establishment_Year'])

data.head(2)
X=data[data.columns[13:20]]
X.head(2)
y=data[data.columns[11]]
y
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)
print(X_train.shape,y_train.shape)
print( X_test.shape, y_test.shape)
Linear_Model=LinearRegression(normalize=True)
Linear_Model.fit(X_train,y_train)
Linear_Model.predict(X_test)

print (Linear_Model.score(X_test, y_test))
from sklearn.tree import DecisionTreeRegressor
decisiontree = DecisionTreeRegressor(random_state=0)
model=decisiontree.fit(X_train,y_train)
model.predict(X_test)
print (model.score(X_test, y_test))
