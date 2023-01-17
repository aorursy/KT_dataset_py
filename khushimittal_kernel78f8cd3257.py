# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("../input/big-mart-sales-dataset/Train_UWu5bXk.csv")
test_data = pd.read_csv("../input/big-mart-sales-dataset/Test_u94Q5KV.csv")
train_data.head()
test_data.head()
print(train_data.shape)
print(test_data.shape)
train_data.info()
['Item_Fat_Content', 'Item_Type ', ' Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
train_data.isnull().sum()
data = train_data.append(test_data, sort=False)
data.isnull().sum()
train_data.describe()
train_data.describe().columns
for i in train_data.describe().columns:
    sns.distplot(data[i].dropna())
    plt.show()
for i in train_data.describe().columns:
    sns.boxplot(data[i].dropna())
    plt.show()
sns.boxplot(data['Item_Visibility'])
data['Item_Visibility'].describe()
sns.boxplot(y=data['Item_Weight'], x=data['Outlet_Identifier'])
plt.xticks(rotation='vertical')
data['Item_Fat_Content'].value_counts()
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF': 'Low Fat','reg':'Regular', 'low fat':'Low Fat'})
data.groupby('Item_Identifier')['Item_Weight'].mean().head(5)
for i in data.groupby('Item_Identifier')['Item_Weight'].mean().index:
    data.loc[data.loc[:,'Item_Identifier']==i, 'Item_Weight'] = data.groupby('Item_Identifier')['Item_Weight'].mean()[i]
data['Outlet_Type'].value_counts()
data.Outlet_Size[data['Outlet_Type']=='Grocery Store'].value_counts()
data.Outlet_Size[data['Outlet_Type']=='Supermarket Type2'].value_counts()
data.Outlet_Size[data['Outlet_Type']=='Supermarket Type1'].value_counts()
data.Outlet_Size[data['Outlet_Type']=='Supermarket Type3'].value_counts()
data.Outlet_Size.fillna(data.Outlet_Size[data['Outlet_Type']=='Grocery Store'].mode()[0], inplace=True)
data.Outlet_Size.fillna(data.Outlet_Size[data['Outlet_Type']=='Supermarket Type1'].mode()[0], inplace=True)
data.Outlet_Size.fillna(data.Outlet_Size[data['Outlet_Type']=='Supermarket Type2'].mode()[0], inplace=True)
data.Outlet_Size.fillna(data.Outlet_Size[data['Outlet_Type']=='Supermarket Type3'].mode()[0], inplace=True)
for i in data.groupby('Item_Identifier')['Item_Visibility'].mean().index:
    data.loc[data.loc[:,'Item_Identifier']==i, 'Item_Visibility']=data.groupby('Item_Identifier')['Item_Visibility'].mean()[i]
data['Outlet_Establishment_Year']=2013-data['Outlet_Establishment_Year']
data
data.isnull().sum()
train_data=data.dropna()
test_data=data[data['Item_Outlet_Sales'].isnull()]
test_data.drop('Item_Outlet_Sales', axis=1, inplace=True)
sns.boxplot(train_data['Item_Visibility'])
train_data['Item_Visibility'].describe()
print(test_data.shape)
print(train_data.shape)
len(train_data)

len(test_data)
from sklearn.preprocessing import LabelEncoder
categorical_list = ['Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Size','Outlet_Location_Type','Outlet_Type','Outlet_Establishment_Year']
le = LabelEncoder()
for i in categorical_list:
    train_data[i] =le.fit_transform(train_data[i])
    train_data[i]=train_data[i].astype('category')
    test_data[i]=le.fit_transform(test_data[i])
    test_data[i]=test_data[i].astype('category')
data
test_data.head()
train_data.head()
train_data.corr()
from sklearn.linear_model import LinearRegression as LR
lm = LR(normalize=True)
lm.fit(train_data.drop(['Item_Outlet_Sales','Outlet_Identifier','Item_Identifier'],axis=1),train_data['Item_Outlet_Sales'])
train_data
y_train=train_data['Item_Outlet_Sales']
X_train=train_data.drop('Item_Outlet_Sales', axis=1)
train = train_data.drop(['Item_Outlet_Sales'], axis=1)
predictions=train_data['Item_Outlet_Sales']
out=[]
LM_model=LR(normalize=True)
for i in range(len(test_data)):
    LM_fit=LM_model.fit(train.drop(['Outlet_Identifier','Item_Identifier'], axis=1), predictions)
    Output=LM_fit.predict(test_data.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1)[test_data.index==i])
    out.append(Output)
    train.append(test_data[test_data.index==i])
    predictions.append(pd.Series(Output))
len(out)
len(test_data)
outp=np.vstack(out)
ansp = pd.Series(data = outp[:,0], index=test_data.index, name='Item_Outlet_Sales')
outp_df=pd.DataFrame([test_data['Item_Identifier'], test_data['Outlet_Identifier'], ansp]).T
outp_df.to_csv('UploadLMP.csv', index= False)
mod1_train_pred=lm.predict(train_data.drop(['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'], axis=1))

from sklearn import metrics
from math import sqrt
sqrt(metrics.mean_squared_error(train_data['Item_Outlet_Sales'], mod1_train_pred))/np.mean(train_data['Item_Outlet_Sales'])

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
ann=MLPRegressor(activation='relu',alpha=2.0,learning_rate='adaptive',warm_start=True,hidden_layer_sizes=(2500,),max_iter=1000)
ann.fit(train_data.drop(['Item_Outlet_Sales','Outlet_Identifier','Item_Identifier'],axis=1),train_data['Item_Outlet_Sales'])
ann_train_pred=ann.predict(train_data.drop(['Item_Outlet_Sales','Outlet_Identifier','Item_Identifier'],axis=1))
r2_score(train_data['Item_Outlet_Sales'],ann_train_pred)
sqrt(metrics.mean_squared_error(train_data['Item_Outlet_Sales'],ann_train_pred))/np.mean(train_data['Item_Outlet_Sales'])
ann_pred=ann.predict(test_data.drop(['Item_Identifier','Outlet_Identifier'],axis=1))

ann_ans=pd.Series(data=ann_pred,index=test_data.index,name='Item_Outlet_Sales')
ann_out=pd.DataFrame([test_data['Item_Identifier'],test_data['Outlet_Identifier'],ann_ans]).T
ann_out.to_csv('Uploadann.csv',index=False)

