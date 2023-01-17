#importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#Reading the data

train = pd.read_csv("../input/train_v9rqX0R.csv")

test = pd.read_csv('../input/test_AbJTz2l.csv')
train.shape
train.head()
train.dtypes
train.isnull().sum()
train.nunique()
# Before we go for EDA, listing numerical (continuous) , categorical and target features :

conti = ['Item_Weight', 'Item_Visibility','Item_MRP']

categ = ['Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Establishment_Year',

         'Outlet_Size', 'Outlet_Location_Type','Outlet_Type']

target = ['Item_Outlet_Sales']
for col in conti:

    sns.distplot(train[col].dropna(), bins=80)

    plt.show()
for col in categ:

    sns.countplot(train[col])

    plt.xticks(rotation = 90)

    plt.show()
for cols in conti:

    sns.scatterplot(x = train[cols], y = train['Item_Outlet_Sales'])

    plt.show()
for cols in categ:

    sns.violinplot(x = train[cols].fillna('NULL'), y = train['Item_Outlet_Sales'])

    plt.xticks(rotation = 90)

    plt.show()
# Note that :
sns.violinplot(x = train['Outlet_Size'].fillna('NULL'), y = train['Item_Outlet_Sales'])

plt.xticks(rotation = 90)
train['Outlet_Size'].fillna('Small', inplace = True)
train['Item_Weight'].fillna(train['Item_Weight'].mean(), inplace = True)
print(train.isnull().sum())
test['Outlet_Size'].fillna('Small', inplace = True)

test['Item_Weight'].fillna(train['Item_Weight'].mean(), inplace = True)
row = train.shape[0]
df = pd.concat([train,test], sort = 'False',ignore_index=True)

df.shape
print(conti)

print(categ)
df['Item_Visibility'].mean()
for i in range(0,14204):

    if df.loc[i,'Item_Visibility']==0:

        df.loc[i,'Item_Visibility'] = 0.06595278007399345       
plt.hist(df['Item_Visibility'].dropna(), bins=100)
## There is some information that can be extracted form item identifier column.

for i in range(0,14204):

    df.loc[i,'Details'] = df.loc[i, 'Item_Identifier'][:2]
df.Details.nunique()
for i in range(0,14204):

    df.loc[i,'Details_3'] = df.loc[i, 'Item_Identifier'][2:3]
df.Details_3.nunique()
df.head()
df['Item_Fat_Content'].replace({'Low Fat':0, 'LF':0,'low fat':0,'Regular':1, 'reg':1}, inplace = True)
df.drop('Item_Identifier', axis=1, inplace=True)
train_data = df.loc[0:8522]

test_data = df[8523:]
test_data.drop('Item_Outlet_Sales', axis=1, inplace=True)
X = train_data.drop('Item_Outlet_Sales', axis=1)

y = train_data['Item_Outlet_Sales']
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)
# Label Encoding

from sklearn.preprocessing import LabelEncoder
lab = ['Details','Details_3','Outlet_Size','Outlet_Location_Type','Outlet_Identifier']



# Make copy to avoid changing original data 

label_X_train = X_train.copy()

label_X_valid = X_valid.copy()

label_test_data = test_data.copy() 

# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

for col in lab:

    label_X_train[col] = label_encoder.fit_transform(X_train[col])

    label_X_valid[col] = label_encoder.transform(X_valid[col])

    label_test_data[col] = label_encoder.transform(label_test_data[col])
label_X_train.head()
from category_encoders import CountEncoder

ce = CountEncoder()

ce.fit(label_X_train['Item_Type'])

label_X_train['Item_Type'+'_count'] = ce.transform(label_X_train['Item_Type'])

label_X_valid['Item_Type'+'_count'] = ce.transform(label_X_valid['Item_Type'])

label_test_data['Item_Type'+'_count'] = ce.transform(label_test_data['Item_Type'])
from category_encoders import CountEncoder

ce2 = CountEncoder()

ce2.fit(label_X_train['Outlet_Type'])

label_X_train['Outlet_Type'+'_count'] = ce2.transform(label_X_train['Outlet_Type'])

label_X_valid['Outlet_Type'+'_count'] = ce2.transform(label_X_valid['Outlet_Type'])

label_test_data['Outlet_Type'+'_count'] = ce2.transform(label_test_data['Outlet_Type'])
label_X_train.drop(['Outlet_Type', 'Item_Type'], axis=1, inplace=True)

label_X_valid.drop(['Outlet_Type', 'Item_Type'], axis=1, inplace=True)

label_test_data.drop(['Outlet_Type', 'Item_Type'], axis=1, inplace=True)
# Removing right skewness of ITEM-VISBILITY 

label_X_train['Item_Visibility'] = np.log(label_X_train['Item_Visibility'] + 1)
label_X_valid['Item_Visibility'] = np.log(label_X_valid['Item_Visibility'] + 1)
from sklearn.preprocessing import Normalizer

scaler = Normalizer()

scaler.fit(label_X_train)



X_train_scaled = pd.DataFrame(scaler.transform(label_X_train),columns=label_X_train.columns)

X_valid_scaled = pd.DataFrame(scaler.transform(label_X_valid),columns=label_X_valid.columns)

X_test_scaled = pd.DataFrame(scaler.transform(label_test_data),columns=label_test_data.columns)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

lr_model = LinearRegression()

print('Training Model!')

lr_model.fit(X_train_scaled, y_train)

pred = lr_model.predict(X_valid_scaled)

print(mean_squared_error(y_valid, pred))
from sklearn.ensemble import RandomForestRegressor

tree_model = RandomForestRegressor()

print('Training Model!')

tree_model.fit(X_train_scaled, y_train)

pred = tree_model.predict(X_valid_scaled)

print(mean_squared_error(y_valid, pred))
from xgboost import XGBRegressor

xgb_model = XGBRegressor()

print('Training Model!')

xgb_model.fit(X_train_scaled, y_train)

pred = xgb_model.predict(X_valid_scaled)

print(mean_squared_error(y_valid, pred))
estimator = range(50, 500, 50)
tune = {}

for n in estimator:

    model = RandomForestRegressor(n_estimators=n)

    model.fit(X_train_scaled,y_train)

    pre = model.predict(X_valid_scaled)

    tune[n] = mean_squared_error(y_valid, pre)

    
keys = list(tune.keys())

values = list(tune.values())
plt.figure(figsize=(12,6))

sns.lineplot(keys, values)
from sklearn.ensemble import RandomForestRegressor

tree_model = RandomForestRegressor(n_estimators=400)

print('Training Model!')

tree_model.fit(X_train_scaled, y_train)

pred = tree_model.predict(X_valid_scaled)

print(mean_squared_error(y_valid, pred))
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)

my_model.fit(X_train_scaled, y_train, 

             early_stopping_rounds=5, 

             eval_set=[(X_valid_scaled, y_valid)], 

             verbose=False)
pred = my_model.predict(X_valid_scaled)

print(mean_squared_error(y_valid, pred))