import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

import plotly
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import matplotlib.pyplot as plt
import pylab as pl
sns.set(font_scale = 0.7)
import os
print(os.listdir("../input"))

#Cargamos el data set mediante Python y Pandas mediante read_csv
#Tener en cuenta indicadar el separador como recomendacion con delimiter se recomienda
#En caso tengamos un campo donde se guarden los ID unico podemos caolocar elnumero de columna en index_col para usarlo de indexacion
data = pd.read_csv("../input/avocado.csv",delimiter=",",index_col=0)

#Visualizamos el volumen de filas y columnas de nuestros datos
data.shape
#Podemos mediante el comando info identificar los tipos de campos 123
data.info()
#Vamos a ver un pequeño resumen de los datos
data.head(5)
#Vamos a validar que se cumple ambos formulas
#Todo aquel registro que no cumpla con la regla pasara a ser una inconsistencia la cual descartaremos.
dataIncorrecta = data[((data['4046'] + data['4225'] + data['4770'] + data['Total Bags']) != data['Total Volume'] ) | ((data['Small Bags'] + data['Large Bags'] + data['XLarge Bags']) != data['Total Bags'] )]
dataCorrecta =  data[((data['4046'] + data['4225'] + data['4770'] + data['Total Bags']) == data['Total Volume'] ) & ((data['Small Bags'] + data['Large Bags'] + data['XLarge Bags']) == data['Total Bags'] )]

dataIncorrecta.reset_index(inplace = True)
dataCorrecta.reset_index(inplace = True)

print("Data Total",data.shape)
print("Data Incorrecta",dataIncorrecta.shape)
print("Data Correcta",dataCorrecta.shape)
#Analizamos valores perdidos en caso de que existan
dataCorrecta.isnull().sum()

import itertools
plt.subplots(figsize=(28,20))
time_spent=['Small Bags','Large Bags','XLarge Bags','XLarge Bags']
length=len(time_spent)
for i,j in itertools.zip_longest(time_spent,range(length)):
    plt.subplot((length/2),2,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    dataCorrecta[i].hist(bins=18,edgecolor='black')
    plt.axvline(dataCorrecta[i].mean(),linestyle='dashed',color='r')
    plt.title(i,size=20)
    plt.xlabel('Tamaño de la bolsa')
    plt.ylabel('cantidad de bolsas')
plt.show()
dataCorrecta =dataCorrecta[(dataCorrecta['Small Bags']<300000.0) & (dataCorrecta['Large Bags']<150000.0) & (dataCorrecta['XLarge Bags']<10000.0)]
dataCorrecta.shape
import itertools
plt.subplots(figsize=(25,16))
time_spent=['Small Bags','Large Bags','XLarge Bags','XLarge Bags']
length=len(time_spent)
for i,j in itertools.zip_longest(time_spent,range(length)):
    plt.subplot((length/2),2,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    dataCorrecta[i].hist(bins=18,edgecolor='black')
    plt.axvline(dataCorrecta[i].mean(),linestyle='dashed',color='r')
    plt.title(i,size=20)
    plt.xlabel('Tamaño de la bolsa')
    plt.ylabel('cantidad de bolsas')
plt.show()
pl.figure(figsize=(12,5))
pl.title("Distribution Price")
ax = sns.distplot(dataCorrecta["AveragePrice"], color = 'g')
sns.boxplot(y="type", x="AveragePrice", data=dataCorrecta, palette = 'Set3')
conventional = dataCorrecta[dataCorrecta.type=="conventional"]
organic = dataCorrecta[dataCorrecta.type=="organic"]

groupBy1_price = conventional.groupby('Date').mean()
scatter1 = go.Scatter(x=groupBy1_price.AveragePrice.index, y=groupBy1_price.AveragePrice, name="Conventional")

groupBy2_price = organic.groupby('Date').mean()
scatter2 = go.Scatter(x=groupBy2_price.AveragePrice.index, y=groupBy2_price.AveragePrice, name="Organic")

data = [scatter1, scatter2]
layout=go.Layout(title="Time Series Plot for Mean Daily Price of Conventional and Organic Avocados", xaxis={'title':'Date'}, yaxis={'title':'Prices'})
figure=go.Figure(data=data,layout=layout)
iplot(figure)
#dataCorrecta['Date2']=pd.to_datetime(dataCorrecta['Date'], format="%Y/%m/%d")
dataCorrecta['Date'] =dataCorrecta['Date'].astype('datetime64[ns]')


#Tenemos que analizar los datos categoricos no numericos, para transformarlos en numericos, o ver si son utiles
dataCorrecta['type'].value_counts()
dataCorrecta.head()
dataCorrecta['year'].value_counts()
dataCorrecta['region'].value_counts()
fig, ax = plt.subplots(1, 1, figsize=(10,6))
sns.boxplot(x='year',y='AveragePrice',data=dataCorrecta,color='red')
#Analizamos el comportamiento de los procesio por region de las paltas organicos
mask = dataCorrecta['type']=='organic'
g = sns.factorplot('AveragePrice','region',data=dataCorrecta[mask],
                   hue='year',
                   height=13,
                   aspect=0.8,
                   palette='magma',
                   join=False,
              )
#Analizamos el comportamiento de los procesio por region de las paltas convencionales
mask = dataCorrecta['type']=='conventional'
g = sns.factorplot('AveragePrice','region',data=dataCorrecta[mask],
                   hue='year',
                   height=13,
                   aspect=0.8,
                   palette='magma',
                   join=False,
              )
label = LabelEncoder()
dicts = {}

label.fit(dataCorrecta.type.drop_duplicates()) 
dicts['type'] = list(label.classes_)
dataCorrecta.type = label.transform(dataCorrecta.type) 
dataCorrecta['type'].value_counts()
#Realizamos una tabla de correlacion, para conocer el nivel de relacion entre los campos y entre nuestro target
cols = ['AveragePrice','Total Volume','4046','4225','4770','Total Bags','Small Bags','Large Bags','XLarge Bags','year','type']
sns.set(font_scale = 1.5)
corr = dataCorrecta[cols].corr('spearman') 
plt.figure(figsize = ( 14 , 14 )) 
sns.heatmap(corr,annot=True,fmt='.2f',cmap="YlGnBu");
#Creamos las variables ficticias para region
region_dummi =pd.get_dummies(dataCorrecta['region'], prefix='reg')
region_dummi.head()
#Agregamos los datos fictios a la data original
data_nueva = pd.concat([dataCorrecta, region_dummi], axis=1)
data_nueva.head() 
#eliminamos las columna region
data_nueva = data_nueva.drop('region', 1)
data_nueva = data_nueva.drop('index', 1)
#corr = data_nueva.corr('spearman') 
#plt.figure(figsize = ( 35 , 20 )) 
#sns.heatmap(corr,annot=True,fmt='.2f',cmap="YlGnBu");
data_train = pd.DataFrame(index=data_nueva.index)
targer_train = pd.DataFrame(index=data_nueva.index)
data_train = data_nueva
targer_train = data_nueva['AveragePrice']
data_train['monthy'] = data_train['Date'].astype('datetime64[ns]').apply(lambda ts: ts.month)
dummi_month =pd.get_dummies(data_train['monthy'], prefix='month')
data_train = pd.concat([data_train, dummi_month], axis=1)
data_train = data_train.drop(['monthy'],axis=1)
#data_train = data_train.drop(['Date'],axis=1)
data_train.head(5)
data_train['year'].value_counts()
region_dummi_fecha =pd.get_dummies(data_train['year'], prefix='year')
data_train = pd.concat([data_train, region_dummi_fecha], axis=1)
data_train.head(5)
data_train = data_train.drop(['year'],axis=1)
data_train = data_train.drop(['Date'], axis=1)
data_train = data_train.drop(['AveragePrice'],axis=1)

data_train.head(5)
targer_train.mean()
%config InlineBackend.figure_format = 'svg'
sns.set(font_scale = 1)
sns.distplot(targer_train);
sns.set(font_scale = 1)
sns.distplot(np.log1p(targer_train));
data_train.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data_train,targer_train,test_size=0.2)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)
#Probamos con un modelo simple Ridge
from sklearn.linear_model import Ridge
ridge = Ridge(random_state=17)
ridge.fit(X_train, y_train);
ridge_pred = ridge.predict(X_test)
sns.set(font_scale = 1)
plt.hist(y_test, bins=50, alpha=.5, color='red', label='true', range=(0,4));
plt.hist(ridge_pred, bins=50, alpha=.5, color='green', label='pred', range=(0,4));
plt.legend();
mean_absolute_error(y_test, ridge_pred)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso,Ridge,BayesianRidge,ElasticNet,HuberRegressor,LinearRegression,LogisticRegression,SGDRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# print X_train.shape, X_test.shape

classifiers = [['DecisionTree :',DecisionTreeRegressor()],
               ['RandomForest :',RandomForestRegressor()],
               ['KNeighbours :', KNeighborsRegressor(n_neighbors = 2)],
               ['SVM :', SVR()],
               ['AdaBoostClassifier :', AdaBoostRegressor()],
               ['GradientBoostingClassifier: ', GradientBoostingRegressor()],
               ['Xgboost: ', XGBRegressor()],
               ['CatBoost: ', CatBoostRegressor(logging_level='Silent')],
               ['Lasso: ', Lasso()],
               ['Ridge: ', Ridge(random_state=17)],
               ['BayesianRidge: ', BayesianRidge()],
               ['ElasticNet: ', ElasticNet()],
               ['HuberRegressor: ', HuberRegressor()]]

print("Accuracy Results...")


for name,classifier in classifiers:
    classifier = classifier
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    print(name, (np.sqrt(mean_squared_error(y_test, predictions))))
classifier = RandomForestRegressor()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

mean_absolute_error(y_test, predictions)
sns.set(font_scale = 1)
data = pd.DataFrame({'Y Test':y_test , 'Pred':predictions},columns=['Y Test','Pred'])
sns.lmplot(x='Y Test',y='Pred',data=data,palette='rainbow')
data.head()
