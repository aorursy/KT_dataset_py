import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline 
train = pd.read_csv('../input/bigmart-sales-data/Train.csv')

test = pd.read_csv('../input/bigmart-sales-data/Test.csv')
train['source'] = 'train'

test['source'] = 'test'

test['Item_Outlet_Sales'] = 0

data = pd.concat([train,test],sort=False)
data.shape
data.info()
data['Item_Outlet_Sales'].describe()
import seaborn as sns

sns.distplot(data['Item_Outlet_Sales'])

plt.show()
data.info()
cat_var = data.select_dtypes(include =[np.object])

cat_var.shape
num_var = data.select_dtypes(include=[np.float64])

num_var.shape
cat_var.isnull().sum()
num_var.isnull().sum()
data.apply(lambda x : len(x.unique()))
data.boxplot(column='Item_Outlet_Sales',by='Item_Fat_Content')
for var in cat_var:

    data[var].fillna(method='ffill',inplace=True)
for var in num_var:

    mean = np.around(np.mean(data[var]))

    data[var].fillna(mean,inplace = True)
data.isnull().sum()
plt.figure(figsize =(10,9))

plt.subplot(311)



ax = sns.boxplot(x = 'Outlet_Size',y = 'Item_Outlet_Sales',data=data ,palette="Set1")

ax.set_title("Outlet_size vs Item_Outlet_Sales")

plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 1.5)
plt.figure(figsize=(10,9))

plt.subplot(212)



ax = sns.boxplot(x ='Outlet_Type' ,y='Item_Outlet_Sales',data=data,palette="Set1" )
#Determine average visibility of a product

visibility_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')



#Impute 0 values with mean visibility of that product:

missing_values = (data['Item_Visibility'] == 0)



data.loc[missing_values,'Item_Visibility'] = data.loc[missing_values,'Item_Identifier'].apply(lambda x: visibility_avg.at[x, 'Item_Visibility'])

data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x:x[0:2])

data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'food','NC':'Non_consum','DR':'Drink'})

data['Item_Type_Combined'].value_counts()
data.info()
plt.figure(figsize=(10,9))

plt.subplot(212)



ax = sns.boxplot(x ='Item_Type_Combined' ,y='Item_Outlet_Sales',data=data,palette="Set1" )
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'})
data['Item_Fat_Content'].value_counts()
plt.figure(figsize=(10,9))

plt.subplot(212)



ax = sns.boxplot(x ='Item_Fat_Content' ,y='Item_Outlet_Sales',data=data,palette="Set1" )
data.groupby('Outlet_Establishment_Year')['Item_Outlet_Sales'].mean().plot.bar()
data['Outlet_Establishment_Year'].describe()
data['Outlet_Year'] = 2019 - data['Outlet_Establishment_Year']

data['Outlet_Year'].describe()
from sklearn.preprocessing import LabelEncoder



encode = LabelEncoder()



data['Outlet'] = encode.fit_transform(data['Outlet_Identifier'])



var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']

for i in var_mod:

    data[i] = encode.fit_transform(data[i])
data.dtypes
data = pd.get_dummies(data,columns = var_mod)
data.dtypes
plt.figure(figsize=(10,9))

plt.subplot(212)



ax = sns.boxplot(x ='Outlet_Year' ,y='Item_Outlet_Sales',data=data,palette="Set1" )
train = data.loc[data['source']=="train"]

test = data.loc[data['source']=="test"]
train.shape , test.shape