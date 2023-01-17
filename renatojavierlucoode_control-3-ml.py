# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn import datasets
x, y = datasets.load_iris(return_X_y=True)
x = x[:100]
y = y[:100]
y.shape
num_neuronas = 15
num_output = 2 #en caso de One hot
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

MLP = Sequential()
#capa oculta
MLP.add(Dense(units= num_neuronas, input_dim= x.shape[1],activation='sigmoid'))
#salida
MLP.add(Dense(units= 1, activation='sigmoid'))
#usando gradiente desendente con LR = 0.1
opt = SGD(lr = 0.1)

MLP.compile(loss='mse',optimizer=opt,metrics=['accuracy','mean_squared_error'])
historia_mlp = MLP.fit(x,y,
                  epochs = 14
                  )
import matplotlib.pyplot as plt
plt.figure(figsize=(12,10))
plt.plot(historia_mlp.history['mean_squared_error'])
plt.xlabel('epoch')
plt.ylabel('mse')
plt.figure(figsize=(12,10))
plt.plot(historia_mlp.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
from sklearn.naive_bayes import GaussianNB

Bayes = GaussianNB()
historia_bayes = Bayes.fit(x,y)
media_0 = np.zeros(4) # Vector con la media de cada entrada en su respectiva posicion
std_0 = np.zeros(4)   # Vector con la desviacion estandar de

for i in range(50):
    media_0[0] += x[i][0]
    media_0[1] += x[i][1]
    media_0[2] += x[i][2]
    media_0[3] += x[i][3]
media_0 = media_0/100

for i in range(50):
    std_0[0] += (x[i][0] - media_0[0])**2
    std_0[1] += (x[i][1] - media_0[1])**2
    std_0[2] += (x[i][2] - media_0[2])**2
    std_0[3] += (x[i][3] - media_0[3])**2
std_0 = std_0/100.

media_1 = np.zeros(4) # Vector con la media de cada entrada en su respectiva posicion
std_1 = np.zeros(4)   # Vector con la desviacion estandar de

for i in range(50,100):
    media_1[0] += x[i][0]
    media_1[1] += x[i][1]
    media_1[2] += x[i][2]
    media_1[3] += x[i][3]
media_1 = media_1/100

for i in range(50,100):
    std_1[0] += (x[i][0] - media_1[0])**2
    std_1[1] += (x[i][1] - media_1[1])**2
    std_1[2] += (x[i][2] - media_1[2])**2
    std_1[3] += (x[i][3] - media_1[3])**2
std_1 = std_1/100

media
import math
for i in range(4):
    std_0[i] = math.sqrt(std_0[i])
    std_1[i] = math.sqrt(std_1[i])
    
print("Media del 0",media_0)
print("desviacion estandar del 0",std_0)

print("Media del 1",media_1)
print("desviacion estandar del 1",std_1)
import math

def GNB(x,mean,std):
    return math.exp(-((x-mean)**2)/(2*std**2))/math.sqrt(2*math.pi*std**2)
def prediccion( x, mean_0, std_0, mean_1, std_1, y):
    P_0 = 0.5
    P_1 = 0.5
    
    Valor_0 = P_0*GNB(x[0],mean_0[0],std_0[0])*GNB(x[1],mean_0[1],std_0[1])*GNB(x[2],mean_0[2],std_0[2])*GNB(x[3],mean_0[3],std_0[3])
    
    Valor_1 = P_1*GNB(x[0],mean_1[0],std_1[0])*GNB(x[1],mean_1[1],std_1[1])*GNB(x[2],mean_1[2],std_1[2])*GNB(x[3],mean_1[3],std_1[3])
    
    Evidencia = Valor_0 + Valor_1
    
    posterior_0 = Valor_0/Evidencia
    posterior_1 = Valor_1/Evidencia
    
    if(posterior_0 > posterior_1):
        if(y == 0):
            print("Acertado")
            return 1 #acertado
        print("Falla")
        return 0 #no acertado
    else:
        if(y == 1):
            print("Acertado")
            return 1 #acertado
        print("Falla")
        return 0 #no acertado
print("x = ",x[10])
print("Valor correspondiente y = ",y[10])

prediccion(x[10],media_0,std_0,media_1,std_1,y[10])

acc = 0
aux = 0
for i in range(100):
    acc += prediccion(x[i],media_0,std_0,media_1,std_1,y[i])
acc = acc/100

print("La precision del modelo es: ", acc)