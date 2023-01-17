# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Load the digits dataset
digits= datasets.load_digits()
data=digits.data #datos
target=digits.target#etiquetas
print("data:\n  tipo:%s   shape:%s"%(str(type(data)),str(data.shape)))
print("targets:\n  tipo:%s   shape:%s"%(str(type(target)),str(target.shape)))

columns=[]
for i in range(64):
    columns.append("(%i,%i)"%(i%8,i//8))
df = pd.DataFrame(data, columns=columns)
df['digito']=target


from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.2,random_state=2)
print(train.shape)
print(test.shape)
train_X = train[columns]
train_Y=train.digito
test_X= test[columns] 
test_Y =test.digito  
df.head(2)
promedios=df.mean()
intervalo=(1, 15)
promedios=promedios[promedios>=intervalo[0]]
promedios=promedios[promedios<=intervalo[1]]
dSel= promedios.index #descriptores seleccionados
print(len(dSel))
print(type(dSel))
df[dSel].head(2)
from sklearn.naive_bayes import GaussianNB
modeloClasificador = GaussianNB()
modeloClasificador.fit(train_X, train_Y)
from sklearn import metrics #for checking the model accuracy
prediccion=modeloClasificador.predict(test_X) 
print('The accuracy of the GaussianNB is:',metrics.accuracy_score(prediccion,test_Y))
#Display the first digit
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[41], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

dsMean=df.mean()#Serie con los promedios
intervalo=(1,15)
dsMean=dsMean[dsMean>=intervalo[0]]
dsMean=dsMean[dsMean<=intervalo[1]]
print("Intervalo de promedios utilizado: [ %i, %i ]"%intervalo)
print("# descriptores\n  eliminados: %i \n  utilizados: %i"%(65-len(dsMean),len(dsMean)-1))

intervalo=(5,11) #Intervalo de valores promedio a tener en cuenta
dsMean=df.mean()#Serie con los promedios
dsMean=dsMean[dsMean>=intervalo[0]]
dsMean=dsMean[dsMean<=intervalo[1]]
print("Intervalo de promedios utilizado: [ %i, %i ]"%intervalo)
print("# descriptores\n  eliminados: %i \n  utilizados: %i"%(65-len(dsMean),len(dsMean)-1))
columns=list(dsMean.index)
if not('digito' in columns): columns.append('digito')
train, test = train_test_split(df[columns], test_size = 0.25,random_state=2)
columns.pop() #Elimina la columna de la etiqueta del digito
train_X = train[columns]
train_Y=train.digito
test_X= test[columns] 
test_Y =test.digito  
modeloClasificador = GaussianNB()
modeloClasificador.fit(train_X, train_Y)
from sklearn import metrics #for checking the model accuracy
prediccion=modeloClasificador.predict(test_X) 
print('Exactitud GaussianNB :',metrics.accuracy_score(prediccion,test_Y))
cols=list(df.columns.values)
imPixDescartados=np.zeros(64)
for pos in columns:
    imPixDescartados[cols.index(pos)]=1
imPixDescartados=imPixDescartados.reshape((8,8))
plt.imshow(imPixDescartados,cmap=plt.cm.gray_r)
plt.title('Pixeles descartados en blanco')
plt.show()
distancia=5 #desviación mínima sobre la cual si se toman los descriptores
dsDesv=df.std()#Serie con los promedios
dsDesv=dsDesv[dsDesv>=distancia]
print("Desviación típica minima:  %i"%distancia)
print("# descriptores\n  eliminados: %i \n  utilizados: %i"%(65-len(dsDesv),len(dsDesv)-1))
columns=list(dsDesv.index)
if not('digito' in columns): columns.append('digito')
train, test = train_test_split(df[columns], test_size = 0.25,random_state=2)
columns.pop() #Elimina la columna de la etiqueta del digito
train_X , train_Y= train[columns],train.digito
test_X , test_Y = test[columns] , test.digito 
modeloClasificador = GaussianNB()
modeloClasificador.fit(train_X, train_Y)
prediccion=modeloClasificador.predict(test_X) 
print('Exactitud GaussianNB :',metrics.accuracy_score(prediccion,test_Y))
cols=list(df.columns.values)
imPixDescartados=np.zeros(64)
for pos in columns:
    imPixDescartados[cols.index(pos)]=1
imPixDescartados=imPixDescartados.reshape((8,8))
plt.imshow(imPixDescartados,cmap=plt.cm.gray_r)
plt.title('Pixeles descartados en blanco')
plt.show()
    