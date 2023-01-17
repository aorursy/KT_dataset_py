import pandas as pd

import seaborn as sn

df = pd.read_csv("../input/House_prediction.csv")

df
df.rename(columns={'parking spaces':'parking_spaces','hoa (R$)':'hoa','rent amount (R$)':'rent',

                   'property tax (R$)':'property_tax','fire insurance (R$)':'fire_insurance','total (R$)':'total'},inplace = True)
df['floor'].replace(to_replace='-',value='0',inplace=True)

df
df.sort_values(by=['city'],inplace=True)

df.reset_index(drop=True, inplace=True)

df
import matplotlib.pyplot as plt

import numpy as np
df2=df.groupby(by=['city'])

type(df2)
from scipy import stats
fig, ax = plt.subplots(figsize=(15,7))

#df.groupby(['city']).apply(lambda x: x.mode())['rent'].unstack().plot(ax=ax,label='mode',legend=True)

df.groupby(['city']).median()['rent'].plot(ax=ax,label="median",legend=True)

df.groupby(['city']).mean()['rent'].plot(ax=ax,label='mean',legend=True)

fig, ax = plt.subplots(figsize=(15,7))

#df.groupby(['city']).apply(lambda x: x.mode())['rooms'].unstack().plot(ax=ax,label='mode',legend=True)

df.groupby(['city']).median()['rooms'].plot(ax=ax,label="median",legend=True)

df.groupby(['city']).mean()['rooms'].plot(ax=ax,label='mean',legend=True)
fig, ax = plt.subplots(figsize=(15,7))

df.groupby(['city']).apply(lambda x: x.mode())['hoa'].unstack().plot(ax=ax,label='mode',legend=True)

df.groupby(['city']).median()['hoa'].plot(ax=ax,label="median",legend=True)

df.groupby(['city']).mean()['hoa'].plot(ax=ax,label='mean',legend=True)
fig, ax = plt.subplots(figsize=(15,7))

#df.groupby(['city']).apply(lambda x: x.mode())['property_tax'].unstack().plot(ax=ax,label='mode',legend=True)

df.groupby(['city']).median()['property_tax'].plot(ax=ax,label="median",legend=True)

df.groupby(['city']).mean()['property_tax'].plot(ax=ax,label='mean',legend=True)
fig, ax = plt.subplots(figsize=(15,7))

df.groupby(['city']).apply(lambda x: x.mode())['fire_insurance'].unstack().plot(ax=ax,label='mode',legend=True)

df.groupby(['city']).median()['fire_insurance'].plot(ax=ax,label="median",legend=True)

df.groupby(['city']).mean()['fire_insurance'].plot(ax=ax,label='mean',legend=True)
fig, ax = plt.subplots(figsize=(15,7))

df.groupby(['city']).apply(lambda x: x.mode())['area'].unstack().plot(ax=ax,label='mode',legend=True)

df.groupby(['city']).median()['area'].plot(ax=ax,label="median",legend=True)

df.groupby(['city']).mean()['area'].plot(ax=ax,label='mean',legend=True)
fig, ax = plt.subplots(figsize=(15,7))

df.groupby(['city']).apply(lambda x: x.mode())['parking_spaces'].unstack().plot(ax=ax,label='mode',legend=True)

df.groupby(['city']).median()['parking_spaces'].plot(ax=ax,label="median",legend=True)

df.groupby(['city']).mean()['parking_spaces'].plot(ax=ax,label='mean',legend=True)
fig, ax = plt.subplots(figsize=(15,7))

df.groupby(['city']).apply(lambda x: x.mode())['bathroom'].unstack().plot(ax=ax,label='mode',legend=True)

df.groupby(['city']).median()['bathroom'].plot(ax=ax,label="median",legend=True)

df.groupby(['city']).mean()['bathroom'].plot(ax=ax,label='mean',legend=True)
df['floor']=df['floor'].astype('int')

fig, ax = plt.subplots(figsize=(15,7))

df.groupby(['city']).apply(lambda x: x.mode())['floor'].unstack().plot(ax=ax,label='mode',legend=True)

df.groupby(['city']).median()['floor'].plot(ax=ax,label="median",legend=True)

df.groupby(['city']).mean()['floor'].plot(ax=ax,label='mean',legend=True)
fig, ax = plt.subplots(figsize=(15,7))

df.groupby(['city']).apply(lambda x: x.mode())['total'].unstack().plot(ax=ax,label='mode',legend=True)

df.groupby(['city']).median()['total'].plot(ax=ax,label="median",legend=True)

df.groupby(['city']).mean()['total'].plot(ax=ax,label='mean',legend=True)
#correlation between hoa,pt,fi with rent

df_cor_rent = df.filter(['hoa','property_tax','fire_insurance','rent'], axis=1)

corrMatrix=df_cor_rent.corr()

sn.heatmap(corrMatrix, annot=True)

plt.show()
#correlation between hoa,pt,fi with area

df_cor_rent = df.filter(['hoa','property_tax','fire_insurance','area'], axis=1)

corrMatrix=df_cor_rent.corr()

sn.heatmap(corrMatrix, annot=True)

plt.show()
#correlation between hoa,pt,fi with floor

df_cor_rent = df.filter(['hoa','property_tax','fire_insurance','floor'], axis=1)

corrMatrix=df_cor_rent.corr()

sn.heatmap(corrMatrix, annot=True)

plt.show()
#correlation between hoa,pt,fi with rooms

df_cor_rent = df.filter(['hoa','property_tax','fire_insurance','rooms'], axis=1)

corrMatrix=df_cor_rent.corr()

sn.heatmap(corrMatrix, annot=True)

plt.show()
#correlation between hoa,pt,fi with bathroom

df_cor_rent = df.filter(['hoa','property_tax','fire_insurance','bathroom'], axis=1)

corrMatrix=df_cor_rent.corr()

sn.heatmap(corrMatrix, annot=True)

plt.show()
#correlation between hoa,pt,fi with parking_spaces

df_cor_rent = df.filter(['hoa','property_tax','fire_insurance','parking_spaces'], axis=1)

corrMatrix=df_cor_rent.corr()

sn.heatmap(corrMatrix, annot=True)

plt.show()
#correlation between hoa,pt,fi with total

df_cor_rent = df.filter(['hoa','property_tax','fire_insurance','total'], axis=1)

corrMatrix=df_cor_rent.corr()

sn.heatmap(corrMatrix, annot=True)

plt.show()
# Manually catogerizing the 'city', 'furniture', 'animal' columns similar to ohe



citydummies=pd.get_dummies(df.city)

cityjoined=pd.concat([df,citydummies],axis='columns')

cityfinal=cityjoined.drop(['city','Campinas'], axis='columns')

animaldummies=pd.get_dummies(cityfinal.animal)

animalsjoined=pd.concat([cityfinal,animaldummies],axis='columns')

animalfinal = animalsjoined.drop(['animal','acept'],axis='columns')

furnituredummies = pd.get_dummies(animalfinal.furniture)

furniturejoined = pd.concat([animalfinal,furnituredummies],axis='columns')

df_final = furniturejoined.drop(['furniture','not furnished'],axis = 'columns')

df_final
df_final.rename(columns={'Belo Horizonte':'Belo','Porto Alegre':'Porto','Rio de Janeiro':'Rio','São Paulo':'Sao',

                         'not acept':'not_acept'}, inplace = True)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

model = LinearRegression()
#X = df_final[['hoa','fire_insurance','property_tax']].values

X = df_final[['fire_insurance']].values

Y = df_final[['rent']].values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state = 0)
model.fit(X_train, Y_train)
model.score(X_test,Y_test)
Y_pred = model.predict(X_test) 

df_compare = pd.DataFrame({'actual':Y_test.flatten(), 'predicted':Y_pred.flatten()})

df_compare

df_compare.head(20).plot(kind='bar',figsize=(10,8))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
X = df_final[['hoa','fire_insurance','property_tax','floor','rooms','bathroom','area','floor','Rio','Sao','Porto','Belo','furnished','not_acept']].values

Y = df_final[['rent']].values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state = 0)

model.fit(X_train, Y_train)

model.score(X_test,Y_test)
Y_pred = model.predict(X_test) 

df_compare = pd.DataFrame({'actual':Y_test.flatten(), 'predicted':Y_pred.flatten()})

df_compare

df_compare.head(20).plot(kind='bar',figsize=(10,8))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()