
#Importacao de pacotes

import numpy as np
import pandas as pd 
import sklearn
import matplotlib.pyplot as plt

import sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import seaborn


datatreino = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",
        sep=',',
        engine='python',
        na_values="?")
datatreino.shape
datatreino.head()
datatreino.info()
datatreino.isnull().sum()
datatreino = datatreino.dropna()
datatreino.info()
datatreino = datatreino.replace(to_replace=['<=50K', '>50K'], value=[0, 1])
datatreino = datatreino.replace(to_replace=['Male','Female'], value=[0, 1])
datatreino = datatreino.replace(to_replace=['White','Black','Asian-Pac-Islander','Amer-Indian-Eskimo','Other'],value=[0,1,2,3,4])
datatreino = datatreino.replace(to_replace=['Husband','Not-in-family','Own-child','Unmarried','Wife','Other-relative'], value=[0,1,2,3,4,5])
datatreino = datatreino.replace(to_replace=['Private','Self-emp-not-inc','Local-gov','State-gov','Self-emp-inc','Federal-gov','Without-pay'], value=[0, 1,2,3,4,5,6])
datatreino = datatreino.replace(to_replace=['Married-civ-spouse','Never-married','Divorced','Separated','Widowed','Married-spouse-absent','Married-AF-spouse'], value=[0, 1,2,3,4,5,6])
datatreino = datatreino.replace(to_replace=['Prof-specialty','Craft-repair','Exec-managerial','Adm-clerical','Sales','Other-service','Machine-op-inspct','Transport-moving','Handlers-cleaners','Farming-fishing','Tech-support','Protective-serv','Priv-house-serv','Armed-Forces'], value=[0, 1,2,3,4,5,6,7,8,9,10,11,12,13])
outros = ['Mexico','Philippines','Germany','Puerto-Rico','Canada','El-Salvador','Cuba','India','England','Jamaica','South','Italy','China','Dominican-Republic','Vietnam','Guatemala','Japan','Columbia','Poland','Taiwan','Haiti','Iran','Portugal','Nicaragua','Peru','Greece','Ecuador','France','Ireland','Hong','Cambodia','Trinadad&Tobago','Laos','Thailand','Yugoslavia','Outlying-US(Guam-USVI-etc)','Hungary','Honduras','Scotland','Holand-Netherlands']
for i in outros:
                   datatreino = datatreino.replace(to_replace=['United-States',i], value=[0, 1])
datatreino.head()
datatreino['education.num'].value_counts()
datatreino['education'].value_counts()
datatreino = datatreino.drop('education', axis = 1)
datatreino.head()
plt.figure(figsize=(10,8))
adultft = datatreino.copy()
seaborn.heatmap(adultft.dropna().corr(), vmin=-1, vmax=1, annot=True, cmap='inferno')
plt.show()

Xdata = datatreino[['Id', 'age', 'education.num','marital.status', 'relationship', 'capital.gain', 'hours.per.week']]
Xdata = Xdata.set_index('Id')
Ydata = datatreino.income

Xdata.head()
melhorK = -1
melhormedia = -1

for i in range(25,32):
    knn = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(knn, Xdata, Ydata, cv=9, n_jobs=-1)
    media = np.mean(scores)
    print(i,":",media)
    if(media > melhormedia):
        melhormedia = media
        melhorK = i
classificacao = KNeighborsClassifier(n_neighbors = melhorK)
classificacao.fit(Xdata,Ydata)
datateste = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",sep=r'\s*,\s*', engine='python', na_values="?")

datateste = datateste.dropna()

dropcolumn = ['Id','workclass', 'education','fnlwgt','race','sex','capital.loss','native.country', 'occupation']

for i in dropcolumn:
    datateste = datateste.drop(i, axis = 1)
    
datateste = datateste.replace(to_replace=['Married-civ-spouse','Never-married','Divorced','Separated','Widowed','Married-spouse-absent','Married-AF-spouse'], value=[0, 1,2,3,4,5,6])
datateste = datateste.replace(to_replace=['Husband','Not-in-family','Own-child','Unmarried','Wife','Other-relative'], value=[0,1,2,3,4,5])
datateste.head()
results = classificacao.predict(datateste)
envio = pd.DataFrame(data = results)
envio = envio.replace(to_replace=[0,1], value=['<=50K','>50K'])
envio['income']=envio[0]
envio = envio.drop(0,axis=1)
envio.head()
envio['income'].value_counts().plot(kind='pie',autopct='%.2f')
envio.to_csv("envio.csv", index = False)