import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, Imputer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score, confusion_matrix
df_train = pd.read_csv("/kaggle/input/orion19-datathon/Orion'19_Datathon_data_freshmen/google_play_store_train.csv")

df_test = pd.read_csv("/kaggle/input/orion19-datathon/Orion'19_Datathon_data_freshmen/test.csv")
df_train.shape
df_test.shape
df_train.head()
print(pd.isnull(df_train).sum())

print()

print(pd.isnull(df_test).sum())
imputer  = Imputer("NaN", 'mean', 0)

imputer  = imputer.fit(df_train.iloc[:, [3]])

df_train.iloc[:, [3]] = imputer.transform(df_train.iloc[:, [3]])
print(pd.isnull(df_train).sum())

print()

print(pd.isnull(df_test).sum())
print(df_train.shape)

print(df_test.shape)
df_train['Type'].value_counts()
df_train = df_train.fillna({'Type':'Free'})
df_train['Content Rating'].value_counts()
df_train = df_train.fillna({'Content Rating' : 'Everyone'})
print(pd.isnull(df_train).sum())

print()

print(pd.isnull(df_test).sum())
ids = df_test['App_ID']
print(df_train.shape)

print(df_test.shape)
df_train.dtypes
df_train.info()
dummy_fields = ['Category', 'Type', 'Content Rating', 'Genres']

for each in dummy_fields:

    dummies = pd.get_dummies(df_train[each], drop_first=False)

    df_train = pd.concat([df_train, dummies], axis=1)
fields_to_drop = ['Category', 'Type', 'Content Rating', 'Genres', 'App_ID', 'App', 'Last Updated', 'Current Ver', 'Android Ver', 'Unnamed: 14']

df_train = df_train.drop(fields_to_drop, axis=1)    
df_train.shape
dummy_fields_test = ['Category', 'Type', 'Content Rating', 'Genres']

for each in dummy_fields_test:

    dummies_test = pd.get_dummies(df_test[each], drop_first=False)

    df_test = pd.concat([df_test, dummies_test], axis=1)
fields_to_drop = ['Category', 'Type', 'Content Rating', 'Genres', 'App_ID', 'App', 'Last Updated', 'Current Ver', 'Android Ver', 'Unnamed: 13']

df_test = df_test.drop(fields_to_drop, axis=1)    
df_test.shape
df_train.Size = [x.strip('M') for x in df_train.Size]
df_train.Size = [x.strip('k') for x in df_train.Size]
df_train.Installs = [x.strip('+') for x in df_train.Installs]
df_train.Price = [x.strip('$') for x in df_train.Price]
df_train.Installs = [x.replace(',', "") for x in df_train.Installs]
df_test.Size = [x.strip('M') for x in df_test.Size]

df_test.Size = [x.strip('k') for x in df_test.Size]

df_test.Installs = [x.strip('+') for x in df_test.Installs]

df_test.Price = [x.strip('$') for x in df_test.Price]

df_test.Installs = [x.replace(',', "") for x in df_test.Installs]
l = []

for i in df_train.columns:

  if i not in df_test.columns:

    l.append(i)
l.remove('Rating')

l   
fields_to_drop_new = l

df_train = df_train.drop(fields_to_drop_new, axis=1) 
df_train.shape 
df_test.shape
df_train = df_train[df_train.Reviews != '3.0M']

df_train = df_train[df_train.Size != 'Varies with device']
df_train = df_train.astype({"Reviews":'float64', "Size":'float64'}) 
df_train = df_train.astype({"Reviews":'int64', "Size":'int64'}) 
df_train = df_train.astype({"Installs":'float64', "Price":'float64'}) 
df_train = df_train.astype({"Installs":'int64', "Price":'int64'}) 
df_test["Size"]= df_test["Size"].replace('Varies with device', 0)
df_test = df_test.astype({"Reviews":'float64'}) 
df_test = df_test.astype({"Reviews":'int64'}) 
df_test = df_test.astype({"Installs":'int64', "Price":'float64'}) 
df_test = df_test.astype({"Installs":'int64', "Price":'int64'}) 
df_test = df_test.astype({"Size":'float64'}) 
df_test = df_test.astype({"Size":'int64'}) 
df_test["Size"]= df_test["Size"].replace(0, df_test['Size'].mean())

df_train.head()
df_test.head()
X = df_train.iloc[:, 1:]
X.head()
y = df_train['Rating']
reg = LinearRegression()

reg.fit(X, y)
X.shape
y.shape
y_test = df_test.iloc[:, :]
y_test.shape
y_pred = reg.predict(y_test)
ids.shape
y_pred.shape
coeff = pd.DataFrame(X.columns)

coeff['Coefficient Estimate'] = pd.Series(reg.coef_)
print(coeff)
predictors = X.columns

coef = pd.Series(reg.coef_,predictors).sort_values()

coef.plot(kind='bar', title='Modal Coefficients', figsize=(18,10))
output = pd.DataFrame({ 'App_ID' : ids, 'Rating': y_pred})
output.head()
output['Rating'] = [round(x, 1) for x in output['Rating']]
output.head()
output.to_csv('output.csv', index=False)