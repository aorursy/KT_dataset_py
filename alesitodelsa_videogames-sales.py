# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
salesdf = pd.read_csv("../input/Video_Games_Sales_as_at_22_Dec_2016.csv")
pd.to_numeric(salesdf["User_Score"], errors='coerce')
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
salesdf['Critic_Score'] = imp.fit_transform(salesdf[['Critic_Score']])
salesdf['Critic_Score'].describe()
#Imputo los nulos de la variable Critic_Score.




salesdf=salesdf[salesdf["User_Score"]!="tbd"]

#La variable User_Score contenía valores tbd que dificultan el preprocesamiento, por lo que en primera lugar los elimino.A continuación, imputos los nulos
from sklearn.impute import SimpleImputer
imp_mean =SimpleImputer(missing_values=np.nan,strategy="mean")
salesdf['User_Score'] = imp.fit_transform(salesdf[['User_Score']])
salesdf["User_Score"].describe()
salesdf.drop(salesdf.columns[[15]],axis=1,inplace=True)
salesdf.drop(salesdf.columns[[14]],axis=1,inplace=True)
salesdf.drop(salesdf.columns[[13]],axis=1,inplace=True)
salesdf.drop(salesdf.columns[[11]],axis=1,inplace=True)
#Elimino las columnas de Critic_count, User_count, Developer y rating, ya que tengo nulos y ademas son columnas que no voy a necesitar(Developer y Publisher se solapan bastantes).

salesdf = salesdf[salesdf.Name.notnull()]
salesdf = salesdf[salesdf.Year_of_Release.notnull()]
salesdf = salesdf[salesdf.Publisher.notnull()]
#Elimino filas de los nulos que haya en estas variables. No puedo hacer medias de ninguna de ellas pero puede que sean variables que necesite.
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
salesdfScaled = mms.fit_transform(salesdf[["NA_Sales","EU_Sales","JP_Sales","Other_Sales","Global_Sales",'Critic_Score',"User_Score"]])
salesdf[["NA_Sales","EU_Sales","JP_Sales","Other_Sales","Global_Sales",'Critic_Score',"User_Score"]] = salesdfScaled 
salesdf[["NA_Sales","EU_Sales","JP_Sales","Other_Sales","Global_Sales","Critic_Score","User_Score"]].describe()
#Normalización de las variables continuas
#¿Tengo que escalar la variable objetivo?
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
salesdf["Platform"] = le.fit_transform(salesdf['Platform'].values)
salesdf["Genre"] = le.fit_transform(salesdf["Genre"].values)


features = ['NA_Sales', 'EU_Sales']
salesdf['Global_Sales'].value_counts()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(salesdf[features],
                                                    salesdf["Global_Sales"],
                                                    test_size=0.3)

print("Lineal Regression score (Train): {0:.2}".format(lr.score(X_train, y_train)))
print("Lineal Regression score (Test): {0:.2}".format(lr.score(X_test, y_test)))
features = ['Genre', 'Platform']
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, y_train)
print("Lineal Regression score (Train): {0:.2}".format(lr.score(X_train, y_train)))
print("Lineal Regression score (Test): {0:.2}".format(lr.score(X_test, y_test)))
features = ['User_Score', 'Critic_Score']
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, y_train)
print("Lineal Regression score (Train): {0:.2}".format(lr.score(X_train, y_train)))
print("Lineal Regression score (Test): {0:.2}".format(lr.score(X_test, y_test)))
features = ['User_Score', 'Critic_Score',"Genre","Platform"]
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, y_train)
print("Lineal Regression score (Train): {0:.2}".format(lr.score(X_train, y_train)))
print("Lineal Regression score (Test): {0:.2}".format(lr.score(X_test, y_test)))
plt.figure(figsize=(12, 8))

sales_corr = salesdf.corr()
sns.heatmap(sales_corr, 
            xticklabels = sales_corr.columns.values,
            yticklabels = sales_corr.columns.values,
            annot = True);
plt.figure(figsize=(10, 8))
plt.scatter(salesdf['User_Score'], salesdf['Critic_Score'], c="red")
plt.title('Puntuación Críticas')
plt.xlabel('Puntuación de usuario')
plt.ylabel('Crítica profesional')
plt.show()
plt.figure(figsize=(10, 8))
plt.scatter(salesdf['Global_Sales'], salesdf['NA_Sales'], c="blue")
plt.title('Relacion Ventas')
plt.xlabel('Ventas Globales')
plt.ylabel('Ventas Norteamérica')
plt.show()
plt.figure(figsize=(10, 8))
plt.scatter(salesdf['Global_Sales'], salesdf['EU_Sales'], c="orange")
plt.title('Relacion Ventas')
plt.xlabel('Ventas Globales')
plt.ylabel('Ventas Europa')
plt.show()
plt.figure(figsize=(10, 8))
plt.scatter(salesdf['NA_Sales'], salesdf['EU_Sales'], c="yellow")
plt.title('Relacion Ventas')
plt.xlabel('Ventas Norteamerica')
plt.ylabel('Ventas Europa')
plt.show()
plt.figure(figsize=(10, 8))
plt.scatter(salesdf['JP_Sales'], salesdf['EU_Sales'], c="green")
plt.title('Relacion Ventas')
plt.xlabel('Ventas Japon')
plt.ylabel('Ventas Europa')
plt.show()
plt.figure(figsize=(10, 8))
plt.scatter(salesdf['Global_Sales'], salesdf['Critic_Score'], c="purple")
plt.title('Relacion Ventas/Crítica')
plt.xlabel('Ventas Globales')
plt.ylabel('Critica profesional')
plt.show()
platformSales = salesdf.groupby(['Platform']).Global_Sales.sum()
platformSales = platformSales[platformSales>150]
platformSales.plot(kind='bar',stacked=True,  colormap= 'Reds', 
                          grid=False, figsize=(13,11))
plt.title('Plataforma')
plt.ylabel('Ventas')
platformCount = salesdf.groupby('Platform').count()['Global_Sales'].reset_index()
platformCount = platformCount[platformCount.Global_Sales>200]
platformCount.plot.bar(x='Platform', y='Global_Sales', rot=30, figsize=(13,11))
plt.title('Plataforma')
plt.ylabel('Ventas')
Genre= salesdf.groupby('Genre').mean()
Genre

%matplotlib inline
from brewer2mpl import qualitative
plt.style.use('dark_background')
genreSales = salesdf.groupby(['Genre']).Global_Sales.sum()
genreSales.plot(kind='bar',stacked=True,  colormap= 'Reds', 
                          grid=False, figsize=(13,11))
plt.title('Genero')
plt.ylabel('Ventas')
genreCount = salesdf.groupby('Genre').count()['Global_Sales'].reset_index()
genreCount.plot.bar(x='Genre', y='Global_Sales', rot=30, figsize=(13,11))
plt.title('Genero')
plt.ylabel('Ventas')