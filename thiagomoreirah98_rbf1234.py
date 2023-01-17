# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,accuracy_score
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_excel('/kaggle/input/arquivos1/Treinamento.xls')
X = np.array(data.drop('d', axis=1))
y = np.array(data['d'])
X
W1 = X[0]
W2 = X[1]
print(W1)
print(W2)
def grafDist(omg1,omg2,w1,w2, distXW1, distXW2):
    circle1 = plt.Circle(w1, np.max(distXW1), color='r', fill=False)
    circle2 = plt.Circle(w2, np.max(distXW2), color='blue', fill=False)
    fig, ax = plt.subplots()
    ax.plot(omg1[:,0],omg1[:,1], "yo",omg2[:,0],omg2[:,1], "co",w1[0],w1[1],"ro",w2[0],w2[1],"bo")
    ax.add_artist(circle1)
    ax.add_artist(circle2)
#inicia os omegas com valores zeros
omega_Anterior1=np.array([])
omega_Anterior2=np.array([])

dxW1=[]
dxW2=[]

while(True):
    
    omega1 = []
    omega2 = []
    
    dxW1.clear()
    dxW2.clear()
    
    for i in range(len(X)):
        dW1 = math.sqrt(sum((X[i]-W1)**2))
        dW2 = math.sqrt(sum((X[i]-W2)**2))
        
        if(dW1 < dW2):
            omega1.append(X[i])
            dxW1.append(dW1)
        else:
            omega2.append(X[i])
            dxW2.append(dW2)
            
    omega1 = np.array(omega1)
    omega2 = np.array(omega2)
    
    if(np.array_equal(omega_Anterior1, omega1) and np.array_equal(omega_Anterior2, omega2)):
        grafDist(omega1, omega2, W1, W2, dxW1, dxW2)
        break
    else:
        omega_Anterior1 = np.copy(omega1)
        omega_Anterior2 = np.copy(omega2)
        
        W1[0] = omega1[:, 0].mean()
        W1[1] = omega1[:, 1].mean()

        W2[0] = omega2[:, 0].mean()
        W2[1] = omega2[:, 1].mean()

W1
W2
def variancia(v,w):
    
    return np.mean((v-w)**2)
print("omega: ",omega1)
print("peso: ",W1)
var1=variancia(omega1,W1)
print("Variancia1: ",var1)
print("-")
print("omega: ",omega2)
print("peso: ",W2)
var2=variancia(omega2,W2)
print("Variancia2: ",var2)
W1
W2
def Somatorio(var, w):
    return sum((var-w)**2)
def FuncG(v,w,var):  
    g=[]
    for i in range(len(v)):        
        #somatorio=sum((v[i]-w)**2) #Faz o somátorio 
        g.append(math.exp(-Somatorio(v[i], w)/(2*var)))          
          
    return g
g1 = FuncG(X, W1, var1)
g2 = FuncG(X, W2, var2)
#Função para achar o EQM
def eqmFun(X,y,w):       
    yp=[]
    for i in range(len(X)):
        yp.append(np.dot(X[i],w))                        
    return mean_squared_error(y,np.array(yp))
X1 = np.array([g1, g2]).T
X1 = np.hstack((np.ones([len(X1), 1]), X1))
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.30, random_state=42)
_W = np.random.uniform(0, 1, len(X1[0]))
tolerancia=0.0000001
l_rate=0.01
num_epocas=100000
epocas=0
eqm=0.1
eqmIni=0.1
while(eqm > tolerancia):
           
    for i in range(len(X_train)): 
        yp= np.dot(X_train[i], _W)             #Previsao para calcular o erro         
        erro = y_train[i]-yp                   #Calculo do Erro
        _W += l_rate * erro * X_train[i]     #Atualiza os pesos
        
    eqmFim = eqmFun(X_train, y_train , _W)         #EQM "final" que sera utilisado para o calculo do EQM de parada do loop                           
    eqm = abs(eqmFim-eqmIni)           #Calculo do EQM que para o loop
    #print("EQM: {} | epocas: {}".format(eqm,epocas))
    
    eqmIni = eqmFim               #Copias o EQM "final" para o "inicial"
    epocas+=1                  #Adicona 1 ao numero de epocas
    if(epocas==num_epocas):     #Parada para quando atingir o maximo de epocas
        break

print("EQM: {} | epocas: {}".format(eqm,epocas))
def predict(X,w):
    if(sum(w*X)>=0):
        return 1#,sum(w*X)
    else:
        return -1#,sum(w*X)
    
def predict1(X,w):    
        return sum(w*X)
_W
#Teste com dados conhecidos
def test(x, y, w):

    saida=np.array([])

    for i in range(len(x)):
        saida=np.append(saida,predict(x[i], w))
    
    acc=accuracy_score(y,saida)
    print("Acurácia de {} %.".format(acc*100))
test(X_train, y_train, _W)
test(X_test, y_test, _W)
#Teste com dados desconhecidos
data_teste = pd.read_excel('/kaggle/input/arquivos1/Resultados.xls')
xt = np.array(data_teste.drop('d', axis=1))
yt = np.array(data_teste['d'])
xt
g1t = FuncG(xt, W1, var1)
g2t = FuncG(xt, W2, var2)
xt = np.array([g1t, g2t]).T
xt = np.hstack((np.ones([len(xt), 1]), xt))
test(xt, yt, _W)
#Teste com dados conhecidos
def valores(x, y, w):

    saida=np.array([])
    s=np.array([])

    for i in range(len(x)):
        saida=np.append(saida,predict(x[i], w))
        s=np.append(s,predict1(x[i], w))
    
    acc=accuracy_score(y,saida)
    print("Acurácia de {} %.".format(acc*100))
    print("\nsaida pre",s)
    print("\nsaida pos",saida)
    print("\nW",w)
valores(xt, yt, _W)
