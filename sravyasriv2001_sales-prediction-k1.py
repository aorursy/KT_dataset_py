import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression
Test = pd.read_csv("../input/bigmart-sales-data/Test.csv")

Train = pd.read_csv("../input/bigmart-sales-data/Train.csv")
X_train=Train

X_test=Test

y_train=Train['Item_Outlet_Sales']

col=Train.columns
Train.info
Train[col[1]].fillna(value=Train[col[1]].mean(),inplace=True)
Train['New']=Train['Outlet_Size'].map({'Small':1,'Medium':2,'High':3})
Train.drop('Outlet_Size',axis=1,inplace=True)

Train.rename(columns={'New':'Outlet_Size'},inplace=True)

Train['Outlet_Size'].fillna(value=Train['Outlet_Size'].mean(),inplace=True)
categorical_columns = [x for x in Train.dtypes.index if Train.dtypes[x]=='object']

print(len(categorical_columns))
categorical_columns=[x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]
Train['Item_type_combined']=Train['Item_Identifier'].apply(lambda x:x[0:2])

Train['Item_type_combined']=Train['Item_type_combined'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})

Train['Item_Fat_Content']=Train['Item_Fat_Content'].map({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat','Regular':'Regular','Low Fat':'Low Fat'})

Train['Item_Fat_Content'].value_counts().sum()

Train.drop(['Item_Type','Outlet_Type','Outlet_Identifier','Outlet_Establishment_Year','Outlet_Location_Type','Outlet_Size'],axis=1,inplace=True)

le = LabelEncoder()

Train['Item_Fat_Content'] = le.fit_transform(Train['Item_Fat_Content'])

Train['Item_type_combined'] = le.fit_transform(Train['Item_type_combined'])

Train['Item_Identifier'] = le.fit_transform(Train['Item_Identifier'])

corr=Train.corr()
X_train = Train

X_test = Test

y = Train['Item_Outlet_Sales']
lr = LinearRegression();
lr.fit(X_train, y)
y_pred = lr.predict(X_train)
col=Test.columns
Test[col[1]].fillna(value=Test[col[1]].mean(),inplace=True)

Test['Outlet_Size']=Test['Outlet_Size'].map({'Small':1,'Medium':2,'High':3})

Test['Outlet_Size']=Test['Outlet_Size'].fillna(value=2.0,inplace=True)

Test['Item_type_combined']=Test['Item_Identifier'].apply(lambda x:x[0:2])

Test['Item_type_combined']=Test['Item_type_combined'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})

Test['Item_Fat_Content']=Test['Item_Fat_Content'].map({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat','Regular':'Regular','Low Fat':'Low Fat'})

Test.drop(['Item_Type','Outlet_Type','Outlet_Identifier','Outlet_Establishment_Year','Outlet_Location_Type','Outlet_Size'],axis=1,inplace=True)
le = LabelEncoder()

Test['Item_Fat_Content'] = le.fit_transform(Test['Item_Fat_Content'])

Test['Item_type_combined'] = le.fit_transform(Test['Item_type_combined'])

Test['Item_Identifier'] = le.fit_transform(Test['Item_Identifier'])
le = LabelEncoder()

Test['Item_Fat_Content'] = le.fit_transform(Test['Item_Fat_Content'])

Test['Item_type_combined'] = le.fit_transform(Test['Item_type_combined'])

Test['Item_Identifier'] = le.fit_transform(Test['Item_Identifier'])

Test['Item_Outlet_Sales']=0.0
Y_pred = lr.predict(Test)
Test['Item_Outlet_Sales']=np.expm1(Y_pred)