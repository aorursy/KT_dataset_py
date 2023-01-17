# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import  matplotlib.pyplot as plt

import seaborn as sns
train=pd.read_csv("/kaggle/input/train_kOBLwZA.csv")

test=pd.read_csv("/kaggle/input/test_t02dQwI.csv")
train["source"]=1

test["source"]=0
df=pd.concat([train,test],sort=False)
df.tail()
df.info()
df.isnull().sum()
df.head()
df.Item_Fat_Content.value_counts()
max(df.Item_Visibility)
df.Item_Weight.mean()
#df.Item_Weight.fillna(df.Item_Weight.mean(),inplace=True)

#df["Item_Weight"] = df.groupby(['Item_Identifier','Item_Type','Item_Fat_Content'])['Item_Weight'].transform(lambda x: x.fillna(x.mean()))

df.head()
df['Item_Identifier_Type'] = df['Item_Identifier'].apply(lambda x: 1 if x.startswith('F') else (2 if x.startswith('D') else (3 if x.startswith('N') else 0)))

df['Item_Identifier_Type'][(df['Item_Identifier'].str.startswith('F')) & (df['Item_Type']=='Dairy')]=4

df['Item_Identifier_Type'][(df['Item_Identifier'].str.startswith('D')) & (df['Item_Type']=='Dairy')]=5
df.Item_Fat_Content.value_counts()
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace(['Low Fat', 'LF'],'low fat')
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace(['reg'],'Regular')
#df["Item_Weight"] = df.groupby(['Item_Identifier','Item_Type','Item_Fat_Content'])['Item_Weight'].transform(lambda x: x.fillna(x.mean()))



df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
df = pd.get_dummies(df, columns = ['Item_Fat_Content'])
df.head()
df.Item_Type.value_counts()
p=['Fruits and Vegetables','Snack Foods','Household','Frozen Foods','Dairy','Baking Goods','Canned','Health and Hygiene','Meat','Soft Drinks','Breads','Hard Drinks','Others','Starchy Foods','Breakfast','Seafood']
df['Item_Type']= df['Item_Type'].replace(p,[15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0])
df.head()
df.groupby(["Outlet_Type","Outlet_Size"]).count()
df[(df["Outlet_Type"]=="Supermarket Type2") | (df["Outlet_Type"]=="Supermarket Type3")].isnull().sum()
df["Outlet_Size"]=np.where(df["Outlet_Type"]=="Grocery Store","Small",df["Outlet_Size"])

df['Outlet_Size'] = np.where(((df['Outlet_Type']=='Supermarket Type1') & (df['Outlet_Size'].isnull())),'Small',df['Outlet_Size'])

df['Outlet_Size'].value_counts()
#df['Outlet_Size']=df['Outlet_Size'].replace(["Small","Medium","High"],[2,1,0])



df = pd.get_dummies(df, columns = ['Outlet_Size'])
#df['Outlet_Location_Type'] = df['Outlet_Location_Type'].replace(['Tier 3','Tier 2','Tier 1'],[3,2,1]) 



df = pd.get_dummies(df, columns = ['Outlet_Location_Type'])
df.head()
df = pd.get_dummies(df, columns = ['Outlet_Type'])
df.head()
df["years"]=2019-df.Outlet_Establishment_Year
del df["Outlet_Establishment_Year"]
df.isnull().sum()
df.head()
df1=df.groupby(["Outlet_Identifier"])["Item_Outlet_Sales"].mean()
df1=df1.reset_index()

df1=df1.rename(columns={'Item_Outlet_Sales':'Item_Outlet_Sales_mean'})
df1.head()
df = pd.merge(df, df1,  how='left', left_on=['Outlet_Identifier'], right_on = ['Outlet_Identifier'])
df.head()
#df.Item_Identifier.value_counts()

#df['Item_Identifier'] = df['Item_Identifier'].factorize()[0]

del df["Item_Identifier"]
df['Outlet_Identifier'] = df['Outlet_Identifier'].factorize()[0]
df.head()
fig, ax = plt.subplots()

ax.scatter(x = train['Item_Visibility'], y = train['Item_Outlet_Sales'])

plt.ylabel('Item_Outlet_Sales', fontsize=13)

plt.xlabel('Item_Visibility', fontsize=13)

plt.show()
#del df["Item_Visibility"]

#df['Item_Visibility'][df['Item_Visibility']>=0.17637255]=0.17637255

#df['Item_Visibility'][df['Item_Visibility']==0]=0.0035747
plt.figure(figsize=(20,10))

corr=df.corr()

sns.heatmap(df.corr(),annot=True)
train = df.loc[df['source'] == 1]

test = df.loc[df['source'] == 0]
train.head()
del train["source"]

test = test.drop(['Item_Outlet_Sales','source'], axis=1)
y=train["Item_Outlet_Sales"]

del train["Item_Outlet_Sales"]

x=train
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state = 42)
import xgboost as xgb

from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
'''model = XGBRegressor()

grid_search = GridSearchCV(model, param_grid = params, scoring = "neg_mean_squared_error", n_jobs=-1,cv=5, verbose=3)

grid_search.fit(x_train,y_train)'''
modelxg =  GradientBoostingRegressor(alpha=0.999, criterion='friedman_mse', init=None,

                          learning_rate=0.061, loss='huber', max_depth=3,

                          max_features=None, max_leaf_nodes=None,

                          min_impurity_decrease=0.0, min_impurity_split=None,

                          min_samples_leaf=4, min_samples_split=4,

                          min_weight_fraction_leaf=0.0, n_estimators=102,

                          n_iter_no_change=None, presort='auto',

                          random_state=None, subsample=1.0, tol=0.0001,

                          validation_fraction=0.1, verbose=0, warm_start=False)
modelxg.fit(x_train,y_train)
y_pred = modelxg.predict(x_test)
from sklearn.metrics import mean_squared_error as mse

print(np.sqrt(mse(y_test,y_pred)))
op = modelxg.predict(test)
sub = pd.read_csv('/kaggle/input/test_t02dQwI.csv')
sub.head()
sub1=pd.DataFrame()
sub1["Item_Identifier"]=sub["Item_Identifier"]

sub1["Outlet_Identifier"]=sub["Outlet_Identifier"]
sub1["Item_Outlet_Sales"]=op
sub1.head()
sub1.to_csv('sub1.csv',index=False)

from IPython.display import FileLink

FileLink(r'sub1.csv')