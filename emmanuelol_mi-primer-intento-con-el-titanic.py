# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# cargar datos de entrenamiento
titanicDF=pd.read_csv('../input/train.csv')
print(titanicDF.columns)
features=titanicDF[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
features.head(5)
# convertir todo a un solo tipo de dato numerico

featuresDummies=pd.get_dummies(features,columns=['Pclass','Sex','Embarked'])
featuresDummies.head(5)
np.isnan(featuresDummies).any()
# etiquetas 
labels=titanicDF[['Survived']]
#separar set de entrenamiento y prueba
from sklearn.model_selection import train_test_split
train_data,test_data, train_labels, test_labels=train_test_split(featuresDummies, labels, random_state=0)
from sklearn.preprocessing import Imputer
imp = Imputer()
imp.fit(train_data)
train_data_finite=imp.transform(train_data)
test_data_finite=imp.transform(test_data)
np.isnan(train_data_finite).any()
# random forest walker

from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(max_depth=10,random_state=0)
classifier.fit(train_data_finite,train_labels)
classifier.predict(test_data_finite)
classifier.score(test_data_finite, test_labels)
#intento con regresion logistica
from sklearn.linear_model import LinearRegression
clf=LinearRegression()
clf.fit(train_data_finite,train_labels)

clf.predict(test_data_finite)
clf.score(test_data_finite,test_labels)
#cargar datos de prueba para competencia
titanicTest=pd.read_csv('../input/test.csv')
featuresTest=titanicTest[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
featuresTestDummies=pd.get_dummies(featuresTest,columns=['Pclass','Sex','Embarked'])
#testData=imp.transform(featuresTestDummies)

np.isnan(featuresTestDummies).any()
impTest=Imputer(missing_values='NaN',strategy='mean',axis=0)
impTest.fit(featuresTestDummies)
testData=imp.transform(featuresTestDummies)
np.isnan(testData).any()
survived=classifier.predict(testData)
ID=titanicTest['PassengerId'].values
Results=pd.DataFrame({'PassengerId':ID,'Survived':survived})
Results
Results.to_csv('submission.csv',sep=',',index=False)


elim=['art', 'thou']
elim
for word in elim:
    print(word)

mapa={'a':2,'b':3,'c':1,'d':3}
mapa

max(mapa,key=mapa.get)
texto='hola gola gola'
for word in texto:
    print(word)

literaturre='romeo romeo wherefore art thou romeo'
exclude=['art','thou']
dictionary={}
for word in literaturre.split():
    if word not in dictionary:
        dictionary[word]=1
    else: 
        dictionary[word]+=1

for word in exclude:
    if word in dictionary:
        del dictionary[word]

dictionary
maxValue=max(dictionary.values())
maxValue
resultado=[]
for key in dictionary:
    if dictionary[key]==maxValue:
        resultado.append(key)
resultado
loglines=[['mi2','jog','mid','pet'],['wz3',34,54,398],['a1','alps','cow','bar']]
temp=[]
for i in range(3):
    temp.append(loglines[i][0])
temp
print(temp.sort())
lista=sorted(temp)
indices=[]

            
indices
