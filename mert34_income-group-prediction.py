import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import os

import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
#Göstergelerin yüklenmesi

df = pd.read_csv('../input/Indicators.csv')
#Ülkelerin yüklenmesi

mf=pd.read_csv('../input/Country.csv')
#Veri kümelerinin birleştirilmesi

cf=pd.merge(df,mf[["CountryCode","IncomeGroup"]], on="CountryCode")
#Ülke isimlerinin index olarak atanması

cf.set_index("CountryName",inplace=True)
#2012 yılının filtrelenmesi

tf=cf[cf["Year"]==2012]
#Gelir grubundaki Nan değerlerin silinmesi

tf.dropna(subset=["IncomeGroup"],inplace=True)
tf.head()
#Kullanmayacağım değişkenlerin veri kümesinden çıkarılması

tf.drop(['CountryCode','IndicatorCode','Year'], axis=1,inplace=True)
#Kullanacağım değişkenlerin veri kümesinden çekilmesi

gdp=tf[(tf["IndicatorName"]=="GDP per capita (current US$)")]

inflation=tf[(tf["IndicatorName"]=="Inflation, GDP deflator (annual %)")]

export=tf[(tf["IndicatorName"]=="Exports of goods and services (% of GDP)")]

imports=tf[(tf["IndicatorName"]=="Imports of goods and services (% of GDP)")]

population=tf[(tf["IndicatorName"]=="Population growth (annual %)")]

manufac=tf[(tf["IndicatorName"]=="Manufacturing, value added (% of GDP)")]

industry=tf[(tf["IndicatorName"]=="Industry, value added (% of GDP)")]

agriculture=tf[(tf["IndicatorName"]=="Agriculture, value added (% of GDP)")]

data = {'gdp': gdp.Value,'inflation': inflation.Value,'export':export.Value,'imports':imports.Value,'population':population.Value,

     'agriculture':agriculture.Value,

     'industry':industry.Value,

     'manufac':manufac.Value}

variables=pd.DataFrame(data=data)
variables["IncomeGroup"]=population["IncomeGroup"]
variables.dropna(axis=0,inplace=True)
variables.head()
X = variables.iloc[:,0:-1].values

X.shape
y = variables.iloc[:,-1].values

y.shape
sc_X = StandardScaler()

X = sc_X.fit_transform(X)
labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
classifier = LogisticRegression(random_state=25, solver='lbfgs')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
predictions = pd.DataFrame(data=y_pred,    # values

                index=range(len(y_pred)),    # 1st column as index

                   columns=['y_pred'])  # 1st row as the column names



# Sadece y_pred'den oluşan df'e test(gerçek) y_test'i sütun olarak ekleme

predictions['y_test'] = y_test

predictions.head()
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)

print(cm)
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: % {:10.2f}".format(accuracy*100)) 

print(classification_report(y_test, y_pred))
sns_plot = sns.pairplot(variables,hue="IncomeGroup",size=1.5)
sns.jointplot(x="gdp",y="inflation",data=variables,kind="reg")
sns.set(style="darkgrid",font_scale=1.5)



f, axes = plt.subplots(4,2,figsize=(16,20))



sns.distplot(variables["gdp"],color="#d7191c",ax=axes[0,0])



sns.distplot(variables["inflation"],color="#fdae61",ax=axes[0,1])



sns.distplot(variables["export"],color="#abd9e9",ax=axes[1,0])



sns.distplot(variables["imports"],color="#2c7bb6",ax=axes[1,1])



sns.distplot(variables["population"],color="#018571",ax=axes[2,0])



sns.distplot(variables["agriculture"],color="r",ax=axes[2,1])



sns.distplot(variables["industry"],color="g",ax=axes[3,0])



sns.distplot(variables["manufac"],color="#a6611a",ax=axes[3,1])