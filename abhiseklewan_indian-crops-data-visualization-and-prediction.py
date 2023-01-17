



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import sys

from sklearn import model_selection

from sklearn.metrics import explained_variance_score

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR







crops_data = pd.read_csv("../input/apy.csv")



crops_data['Season'] = crops_data['Season'].str.rstrip()

crops_data['Crop_Year']=crops_data['Crop_Year'].astype(str)
cultivation_data = crops_data[['Crop_Year', 'Crop', 'Area', 'Production']]

y='Rice'

cultivation_data=cultivation_data.groupby('Crop').get_group(y)

cultivation_data=cultivation_data.groupby('Crop_Year')[['Production', 'Area']].sum()

    

print(cultivation_data)

print("Bar plot of the above data")

cultivation_data.dropna().plot(kind='bar', figsize=(20,10), logy=True, color=['dodgerblue', 'aqua'])

print("Line plot of the above data")

cultivation_data.dropna().plot(figsize=(20,10), logy=True, color=['dodgerblue', 'aqua'], linestyle='solid', marker='o', alpha=0.8, markersize=8)

x='Assam'

y='Rice'

cultivation_data = crops_data[['State_Name', 'Crop_Year', 'Crop', 'Area', 'Production']]

    

cultivation_data=cultivation_data.groupby('State_Name').get_group(x)

cultivation_data=cultivation_data.groupby('Crop').get_group(y)

cultivation_data=cultivation_data.groupby('Crop_Year')[['Production', 'Area']].sum()

    

print(cultivation_data)

print("Bar plot of the above data")

cultivation_data.dropna().plot(kind='bar', figsize=(20,10), logy=True, color=['dodgerblue', 'aqua'])

print("Line plot of the above data")

cultivation_data.dropna().plot(figsize=(20,10), logy=True, color=['dodgerblue', 'aqua'], linestyle='solid', marker='o', alpha=0.8, markersize=8)
x='Uttar Pradesh'

z='ETAH'

y='Maize'

cultivation_data = crops_data[['State_Name','District_Name', 'Crop_Year', 'Crop', 'Area', 'Production']]

  

cultivation_data=cultivation_data.groupby('State_Name').get_group(x)

cultivation_data=cultivation_data.groupby('Crop').get_group(y)

cultivation_data=cultivation_data.groupby('District_Name').get_group(z)

cultivation_data=cultivation_data.groupby('Crop_Year')[['Production', 'Area']].sum()

  

print(cultivation_data)

print("Bar plot of the above data")

cultivation_data.dropna().plot(kind='bar', figsize=(20,10), logy=True, color=['dodgerblue', 'aqua'])

print("Line plot of the above data")

cultivation_data.dropna().plot(figsize=(20,10), logy=True, color=['dodgerblue', 'aqua'], linestyle='solid', marker='o', alpha=0.8, markersize=8)
x='Rabi'

z='Barley'

cultivation_data=crops_data[['Season', 'Crop_Year', 'Crop', 'Area', 'Production']]

cultivation_data=cultivation_data.groupby('Season').get_group(x)

cultivation_data=cultivation_data.groupby('Crop').get_group(z)

cultivation_data=cultivation_data.groupby('Crop_Year')[['Production', 'Area']].sum()

  

print(cultivation_data)

print("Bar plot of the above data")

cultivation_data.dropna().plot(kind='bar', figsize=(20,10), logy=True, color=['dodgerblue', 'aqua'])

print("Line plot of the above data")

cultivation_data.dropna().plot(figsize=(20,10), logy=True, color=['dodgerblue', 'aqua'], linestyle='solid', marker='o', alpha=0.8, markersize=8)
x='Bihar'

y='Wheat'

cultivation_data = crops_data[['State_Name', 'Crop_Year', 'Crop', 'Area', 'Production']]

    

cultivation_data=cultivation_data.groupby('State_Name').get_group(x)

cultivation_data=cultivation_data.groupby('Crop').get_group(y)

cultivation_data=cultivation_data.groupby('Crop_Year')[['Production', 'Area']].sum()

    

a='Jharkhand'

b='Wheat'

cultivation_data1 = crops_data[['State_Name', 'Crop_Year', 'Crop', 'Area', 'Production']]

    

cultivation_data1=cultivation_data1.groupby('State_Name').get_group(a)

cultivation_data1=cultivation_data1.groupby('Crop').get_group(b)

cultivation_data1=cultivation_data1.groupby('Crop_Year')[['Production', 'Area']].sum()

    

cultivation_data.rename(columns={'Production':'Production_of_state1'}, inplace=True)

cultivation_data.rename(columns={'Area':'Area_of_state1'}, inplace=True)

cultivation_data1.rename(columns={'Production':'Production_of_state2'}, inplace=True)

cultivation_data1.rename(columns={'Area':'Area_of_state2'}, inplace=True)

df12=pd.concat([cultivation_data, cultivation_data1], axis=1)

    

print(df12)

ax=df12.plot(kind='bar', figsize=(20,10), color=['darkorange', 'forestgreen', 'darkgoldenrod', 'limegreen'], grid=True)

print("COMPARISON OF PRODUCTION")

ax=df12.plot(y=['Production_of_state1', 'Production_of_state2'], figsize=(20,10), color=['darkorange', 'darkgoldenrod'], grid=True, linestyle='solid', marker='o', alpha=0.8, markersize=8)

print("COMPARISON OF AREA")

ax=df12.plot(y=['Area_of_state1', 'Area_of_state2'], figsize=(20,10), color=['forestgreen', 'limegreen'], grid=True, linestyle='solid', marker='o', alpha=0.8, markersize=8)
x='West Bengal'

z='MALDAH'

y='Sugarcane'

cultivation_data = crops_data[['State_Name', 'District_Name', 'Crop_Year', 'Crop', 'Area', 'Production']]

  

cultivation_data=cultivation_data.groupby('State_Name').get_group(x)

cultivation_data=cultivation_data.groupby('District_Name').get_group(z)

cultivation_data=cultivation_data.groupby('Crop').get_group(y)

cultivation_data=cultivation_data.groupby('Crop_Year')[['Production', 'Area']].sum()

  

a='Uttar Pradesh'

c='KHERI'

b='Sugarcane'

cultivation_data1 = crops_data[['State_Name', 'District_Name', 'Crop_Year', 'Crop', 'Area', 'Production']]

  

cultivation_data1=cultivation_data1.groupby('State_Name').get_group(a)

cultivation_data1=cultivation_data1.groupby('District_Name').get_group(c)

cultivation_data1=cultivation_data1.groupby('Crop').get_group(b)

cultivation_data1=cultivation_data1.groupby('Crop_Year')[['Production', 'Area']].sum()

  

cultivation_data.rename(columns={'Production':'Production_of_district1'}, inplace=True)

cultivation_data.rename(columns={'Area':'Area_of_district1'}, inplace=True)

cultivation_data1.rename(columns={'Production':'Production_of_district2'}, inplace=True)

cultivation_data1.rename(columns={'Area':'Area_of_district2'}, inplace=True)

df12=pd.concat([cultivation_data, cultivation_data1], axis=1)



print(df12)

ax=df12.plot(kind='bar', figsize=(20,20), color=['darkorange', 'forestgreen', 'darkgoldenrod', 'limegreen'], grid=True)

print("COMPARISON OF PRODUCTION")

ax=df12.plot(y=['Production_of_district1', 'Production_of_district2'], figsize=(20,10), color=['darkorange', 'darkgoldenrod'], grid=True, linestyle='solid', marker='o', alpha=0.8, markersize=8)

print("COMPARISON OF AREA")

ax=df12.plot(y=['Area_of_district1', 'Area_of_district2'], figsize=(20,10), color=['forestgreen', 'limegreen'], grid=True, linestyle='solid', marker='o', alpha=0.8, markersize=8)
x='2010'

y='Rice'

  

dd1=crops_data[['State_Name', 'Crop_Year', 'Crop', 'Area', 'Production']]



dd1=dd1.groupby('Crop_Year').get_group(x)

dd1=dd1.groupby('Crop').get_group(y)

dd1=dd1.groupby('State_Name')[['Production', 'Area']].sum()

dd1=dd1.sort_values(by='Production', ascending=False)

print(dd1)

print("Pie chart plot of the data for top ten states")

dd1[:10].plot(kind='pie', y='Production', figsize=(10,10), autopct='%1.1f%%')

dd1[:10].plot(kind='pie', y='Area', figsize=(10,10), autopct='%1.1f%%')

print('Joint plot showing the about the ratio')
x='2011'

y='Wheat'

z='Assam'

  

dd1=crops_data[['State_Name', 'District_Name', 'Crop_Year', 'Crop', 'Area', 'Production']]



dd1=dd1.groupby('Crop_Year').get_group(x)

dd1=dd1.groupby('Crop').get_group(y)

dd1=dd1.groupby('State_Name').get_group(z)

dd1=dd1.groupby('District_Name')[['Production', 'Area']].sum()

dd1=dd1.sort_values(by='Production', ascending=False)

print(dd1)

print("Pie chart plot of the data for top ten states")

dd1[:10].plot(kind='pie', y='Production', figsize=(10,10), autopct='%1.1f%%')

dd1[:10].plot(kind='pie', y='Area', figsize=(10,10), autopct='%1.1f%%')
models = []

models.append(LinearRegression())

models.append(DecisionTreeRegressor())

models.append(KNeighborsRegressor(n_neighbors = 2))

models.append(SVR(gamma='auto'))

names=['LR', 'DTR', 'KNR', 'SVR']
cultivation_data = crops_data[['Crop_Year', 'Crop', 'Area', 'Production']]

y='Banana'

cultivation_data=cultivation_data.groupby('Crop').get_group(y)

cultivation_data=cultivation_data.groupby('Crop_Year')[['Production', 'Area']].sum()

predofprod=list()

X = cultivation_data['Area'].values.reshape(-1, 1)

Y = cultivation_data['Production'].ravel()

validation_size = 0.30

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=validation_size)

varlist=list()

for model in models:

    model.fit(X_train, y_train)

    y_pred=model.predict(X_test)

    #print(explained_variance_score(y_test, y_pred))

    r1=explained_variance_score(y_test, y_pred)

    varlist.append(r1)

print("Explained variance score for different models")

print('LR : ', varlist[0])

print('DTR : ', varlist[1])

print('KNR : ', varlist[2])

print('SVR : ', varlist[3])

sns.barplot(x=varlist, y=names)

best_model_var=varlist[0]

best_model=models[0]

for i in range(len(names)):

    if varlist[i]>best_model_var:

        best_model_var=varlist[i]

        best_model=models[i]

print(best_model)



best_model.fit(X_train, y_train)

z=12345

predofprod.append(z)

predofprod=np.reshape(predofprod, (1,-1))

prod=best_model.predict(predofprod)

print("Production will be",prod[0])
x='Assam'

y='Potato'

cultivation_data = crops_data[['State_Name', 'Crop_Year', 'Crop', 'Area', 'Production']]

predofprod=list()  

cultivation_data=cultivation_data.groupby('State_Name').get_group(x)

cultivation_data=cultivation_data.groupby('Crop').get_group(y)

cultivation_data=cultivation_data.groupby('Crop_Year')[['Production', 'Area']].sum()

X = cultivation_data['Area'].values.reshape(-1, 1)

Y = cultivation_data['Production'].ravel()

validation_size = 0.30

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=validation_size)

varlist=list()

for model in models:

    model.fit(X_train, y_train)

    y_pred=model.predict(X_test)

    #print(explained_variance_score(y_test, y_pred))

    r1=explained_variance_score(y_test, y_pred)

    varlist.append(r1)

print("Explained variance score for different models")

print('LR : ', varlist[0])

print('DTR : ', varlist[1])

print('KNR : ', varlist[2])

print('SVR : ', varlist[3])

sns.barplot(x=varlist, y=names)

best_model_var=varlist[0]

best_model=models[0]

for i in range(len(names)):

    if varlist[i]>best_model_var:

        best_model_var=varlist[i]

        best_model=models[i]

print(best_model)



best_model.fit(X_train, y_train)

z=12345

predofprod.append(z)

predofprod=np.reshape(predofprod, (1,-1))

prod=best_model.predict(predofprod)

print("Production will be",prod[0])
x='Kerala'

w='MALAPPURAM'

y='Coconut '

cultivation_data = crops_data[['State_Name', 'District_Name', 'Crop_Year', 'Crop', 'Area', 'Production']]

predofprod=list()  

cultivation_data=cultivation_data.groupby('State_Name').get_group(x)

cultivation_data=cultivation_data.groupby('Crop').get_group(y)

cultivation_data=cultivation_data.groupby('District_Name').get_group(w)

cultivation_data=cultivation_data.groupby('Crop_Year')[['Production', 'Area']].sum()

X = cultivation_data['Area'].values.reshape(-1, 1)

Y = cultivation_data['Production'].ravel()

validation_size = 0.30

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=validation_size)

varlist=list()

for model in models:

    model.fit(X_train, y_train)

    y_pred=model.predict(X_test)

    #print(explained_variance_score(y_test, y_pred))

    r1=explained_variance_score(y_test, y_pred)

    varlist.append(r1)

print("Explained variance score for different models")

print('LR : ', varlist[0])

print('DTR : ', varlist[1])

print('KNR : ', varlist[2])

print('SVR : ', varlist[3])

sns.barplot(x=varlist, y=names)

best_model_var=varlist[0]

best_model=models[0]

for i in range(len(names)):

    if varlist[i]>best_model_var:

        best_model_var=varlist[i]

        best_model=models[i]

print(best_model)



best_model.fit(X_train, y_train)

z=12345

predofprod.append(z)

predofprod=np.reshape(predofprod, (1,-1))

prod=best_model.predict(predofprod)

print("Production will be",prod[0])
x='Summer'

y='Rice'

cultivation_data=crops_data[['Season', 'Crop_Year', 'Crop', 'Area', 'Production']]

cultivation_data=cultivation_data.groupby('Season').get_group(x)

cultivation_data=cultivation_data.groupby('Crop').get_group(y)

cultivation_data=cultivation_data.groupby('Crop_Year')[['Production', 'Area']].sum()

predofprod=list()

X = cultivation_data['Area'].values.reshape(-1, 1)

Y = cultivation_data['Production'].ravel()

validation_size = 0.30

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=validation_size)

varlist=list()

for model in models:

    model.fit(X_train, y_train)

    y_pred=model.predict(X_test)

    #print(explained_variance_score(y_test, y_pred))

    r1=explained_variance_score(y_test, y_pred)

    varlist.append(r1)

print("Explained variance score for different models")

print('LR : ', varlist[0])

print('DTR : ', varlist[1])

print('KNR : ', varlist[2])

print('SVR : ', varlist[3])

sns.barplot(x=varlist, y=names)  

best_model_var=varlist[0]

best_model=models[0]

for i in range(len(names)):

    if varlist[i]>best_model_var:

        best_model_var=varlist[i]

        best_model=models[i]

print(best_model)



best_model.fit(X_train, y_train)

z=12345

predofprod=list()

predofprod.append(z)

predofprod=np.reshape(predofprod, (1,-1))

prod=best_model.predict(predofprod)

print("Production will be",prod[0])