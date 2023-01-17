from sklearn import linear_model

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

from sklearn import preprocessing  

from sklearn.model_selection import train_test_split
import pandas as pd

import numpy as np

from math import sqrt 
import seaborn as sns

import matplotlib.pyplot as plt
lm = linear_model.LinearRegression()
train = pd.read_csv('../input/bigmart-sales-data/Train.csv') ########### Reading training Data 
test = pd.read_csv('../input/bigmart-sales-data/Test.csv')  ########### Reading testing Data 
train['source'] = 'train'

test['source'] = 'test'

test['Item_Outlet_Sales'] = 0

data = pd.concat([train, test], sort = False)

print(train.shape, test.shape, data.shape)
data
data.dtypes 

data.isnull().sum()
data['Item_Weight']=data['Item_Weight'].fillna(data['Item_Weight'].mean())
data
sns.countplot(data.Outlet_Location_Type)
sns.boxplot(data['Item_Weight'])
plt.figure(figsize=(12,8))

sns.countplot(data.Outlet_Type)
plt.figure(figsize=(25,8))

sns.countplot(data.Outlet_Identifier)
plt.figure(figsize=(25,8))

sns.countplot(data.Item_Type)
data['Outlet_Size'] = data['Outlet_Size'].fillna(data['Outlet_Size'].value_counts().index[0])
data
sns.boxplot(x='Outlet_Size',y='Item_Outlet_Sales',data = data)



sns.boxplot(x='Item_Outlet_Sales',y='Item_Fat_Content',data = data)
plt.figure(figsize=(20,8))

sns.boxplot(y='Item_Outlet_Sales',x='Outlet_Type',hue = 'Outlet_Location_Type',data = data)
data['Item_Outlet_Sales'].describe()
sns.distplot(data['Item_Outlet_Sales'])
print("skewness :",data['Item_Outlet_Sales'].skew())

print("kurtosis :",data['Item_Outlet_Sales'].kurt())
train.columns
#### Unique items in data set

data.apply(lambda x : len(x.unique()))
le = LabelEncoder()

enc = OneHotEncoder()
label = ['Item_Identifier','Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Establishment_Year']

one = ['Outlet_Size','Outlet_Location_Type','Outlet_Type']
for col in label:

    data[col]=le.fit_transform(data[col]) 
data
for col in one:

    data[col]=le.fit_transform(data[col]) 
data
def model(data):

    grp = data.groupby('source')

    test = grp.get_group('test')

    train = grp.get_group('train')

    train = train.drop('source',axis=1)

    test = test.drop('source',axis=1)

    Y = train.Item_Outlet_Sales

    X = train.drop('Item_Outlet_Sales',axis=1)

    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state = 42)

    model = lm.fit(x_train, y_train)

    predictions = lm.predict(x_test)

    RMSE = sqrt((( y_test-predictions)**2).mean())

    return RMSE
print('RMSE with raw data :',model(data))
data.corr(method='pearson')##### Linear correlation
rmv = ['Item_Identifier','Item_Weight','Item_Fat_Content','Item_Type','Outlet_Establishment_Year']
data.drop(rmv, axis=1, inplace=True)
data.columns
print('RMSE after reducing some features :',model(data))
def remove_outlier(data,column):

    Q3 = data[column].quantile(.75)

    Q1 = data[column].quantile(.25)

    IQR = Q3-Q1

    data = data[~((data[column] < (Q1 - 1.5 * IQR)) |(data[column] > (Q3 + 1.5 * IQR)))]

    return data

    
data_o =remove_outlier(data,'Outlet_Size')
data_o
print('RMSE after removing outliers :',model(data_o))