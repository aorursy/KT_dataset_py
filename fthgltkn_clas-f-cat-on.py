# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#DATAMIZI İNCELEMELİYİZ(we need to review our data)
data=pd.read_csv('../input/winequality-red.csv')

data.isnull().sum()

data.info()
data.columns=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',

       'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',

       'pH', 'sulphates', 'alcohol', 'quality']
data.head()
#SINIFLANDIRMA MODELLERİMİZE GEÇEBİLİRİZ(WE CAN GO TO OUR CLASSIFICATION MODELS)
#KNN CLASİFİCATİON
data.quality=[1 if each > 6 else 0 for each in data.quality]

y=data.quality

x=data.drop(['quality'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#VERİMİZİ NORMALİZE EDİYORUZ(NORMALİZE OF DATA)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train=sc.fit_transform(x_train)

x_test=sc.fit_transform(x_test)
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train)

print('{} nn değeri için {} '.format(3,knn.score(x_test,y_test)))
y_pred1=knn.predict(x_test)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred1)



f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_test")

plt.ylabel("y_pred1")

plt.show()
#SUPER VECTOR CLASİFİCATİON
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.svm import SVC

svm=SVC(random_state=42)

svm.fit(x_train,y_train)

print('svm score :',svm.score(x_test,y_test))
y_pred2=svm.predict(x_test)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred2)



f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_test")

plt.ylabel("y_pred2")

plt.show()
#NAİVE BAYES CLASİFİCATİON
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

nb.fit(x_train,y_train)

print('nb score :',nb.score(x_test,y_test))
y_pred3=nb.predict(x_test)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred3)



f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_test")

plt.ylabel("y_pred3")

plt.show()
#DECİSİON TREE CLASİFİCATİON
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()

dt.fit(x_train,y_train)

print('dt score',dt.score(x_test,y_test))
y_pred4=dt.predict(x_test)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred4)



f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_test")

plt.ylabel("y_pred4")

plt.show()
#RANDOM FOREST CLASİFİCATİON
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=100,random_state=42)

rf.fit(x_train,y_train)

print('rf score :',rf.score(x_test,y_test))
y_pred5=rf.predict(x_test)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred5)



f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_test")

plt.ylabel("y_pred5")

plt.show()
#TÜM CLASİFİCATİON METODLARIMIZIN SONUÇLARINI BİR PLOTTA GÖSTERELİM

#(SHOWING THE RESULTS OF OUR CLASIFICATİON METHODS ON A PLOT)
dicti={'knn_score':knn.score(x_test,y_test),'svm_score':svm.score(x_test,y_test),

           'nb_score':nb.score(x_test,y_test),'dt_score': dt.score(x_test,y_test),

          'rf_score':rf.score(x_test,y_test)}
for key,value in dicti.items():

    print(key," : ",value)

    

print('')


key_lis=list(('knn_score','svm_score','nb_score','dt_score','rf_score'))

value_lis=list((0.85,0.85,0.853125,0.865625,0.9))
plt.plot(key_lis,value_lis)

plt.show()
#CONCLUSİON



#Bu verimiz için en uygun methodumuz RANDOM FOREST 

#çünkü en doğru tahmin yukarıda görüldüğü gibi RF methodunda elde edildi.

#(Our best method for this data is RANDOM FOREST

# Because the most accurate estimate was obtained in the RF method as seen above)