# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import math

from sklearn.metrics import mean_squared_error,accuracy_score

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



base=pd.read_excel("/kaggle/input/tabela/Tabel_Treinamento_RNA.xls")

x=np.array(base.drop(["d"],1))

y=np.array(base.d)

base_teste=pd.read_csv("/kaggle/input/basevalidacao/teste.csv")

base_teste



xtest=np.array(base_teste.drop(['d'],1))

ytest=np.array(base_teste.d)

xtest
def graf(x,w1,w2):

    plt.plot(x[:,0],x[:,1], "yo",w1[0],w1[1],"ro",w2[0],w2[1],"bo")

    plt.ylabel('x2')

    plt.xlabel('x1')

    plt.show()
def grafDist(omg1,omg2,w1,w2):

    circle1 = plt.Circle(w1, np.max(distXW1), color='r', fill=False)

    circle2 = plt.Circle(w2, np.max(distXW2), color='blue', fill=False)

    fig, ax = plt.subplots()

    ax.plot(omg1[:,0],omg1[:,1], "yo",omg2[:,0],omg2[:,1], "co",w1[0],w1[1],"ro",w2[0],w2[1],"bo")

    ax.add_artist(circle1)

    ax.add_artist(circle2)
plt.plot(x[:,0],x[:,1], "ro")

plt.ylabel('x2')

plt.xlabel('x1')

plt.show()
w1=x[0]

w2=x[1]

graf(x,w1,w2)

omg1I=np.array([[1]])

omg2I=np.array([[1]])

omg1F=np.array([[]])

omg2F=np.array([[]])

distXW1=np.array([[]])

distXW2=np.array([[]])

#omg1I=np.append(omg1I,1)

#omg1I=np.append(omg1I,2)
def media(vet):

    media1=0

    media2=0

    for i in range(len(vet)):        

        if (i%2==0):

            media1+=vet[i]           

        else:

            media2+=vet[i]           

    return [media1/(len(vet)/2),media2/(len(vet)/2)]
while(np.array_equal(omg1I,omg1F)==False and np.array_equal(omg2I,omg2F)==False):    

    omg1I=np.copy(omg1F)

    omg2I=np.copy(omg2F)

    omg1F=np.delete(0,0)

    omg2F=np.delete(0,0)    

    distXW1=np.delete(0,0)

    distXW2=np.delete(0,0)

    aux1=[]

    aux2=[]

    aux1.clear()

    aux2.clear()

    

    for i in range(len(x)):

        distW1=math.sqrt(sum((x[i]-w1)**2))     

        distW2=math.sqrt(sum((x[i]-w2)**2)) 

     

        if(distW1<distW2):

            aux1.append(x[i])

            distXW1=np.append(distXW1,distW1)



        else:

            aux2.append(x[i])

            distXW2=np.append(distXW2,distW2)



    omg1F=np.array(aux1)

    omg2F=np.array(aux2)

    w1=np.array([np.mean(omg1F[:, 0]), np.mean(omg1F[:, 1])])

    w2=np.array([np.mean(omg2F[:, 0]), np.mean(omg2F[:, 1])])   

    grafDist(omg1F,omg2F,w1,w2)

    

    
def variancia(vet,w):

    

    return np.mean((vet-w)**2)


print("-----------------------------------------------------------")

print("omega: ",omg1F)

print("peso: ",w1)

var1=variancia(omg1F,w1)

print("Variancia1: ",var1)

print("-----------------------------------------------------------")

print("omega: ",omg2F)

print("peso: ",w2)

var2=variancia(omg2F,w2)

print("Variancia2: ",var2)

print("-----------------------------------------------------------")
def getG(vet,w,var):  

    g=np.array([])

    aux=[]

    for i in range(len(vet)):        

        div=sum((vet[i]-w)**2) 

        aux.append(math.exp(-div/(2*var)))

    g=np.append(g,aux)           

          

    

    return g
def x_to_g(x,w1,w2,var1,var2):

    bias=np.ones([len(x),1]) 

    g1=getG(x,w1,var1)

    g2=getG(x,w2,var2)

    z=np.array([g1,g2]).T

    

    return np.hstack((bias,z)) 

#['{:.3f}'.format(float(x)) for x in g1]

def gerarPesos(tam):       

    return np.random.uniform(0,1,tam)



#Função para achar o EQM

def eqmFun(X,y,w):       

    yp=[]

    for i in range(len(X)):

        yp.append(np.dot(X[i],w))                        

    return mean_squared_error(y,np.array(yp))
z=x_to_g(x,w1,w2,var1,var2)

w=gerarPesos(len(x[0])+1)     

wInicio=np.copy(w)              

tolerancia=0.0000001

l_rate=0.01

num_epocas=100000     

epocas=0 

eqm=0.1  

eqmIni=0.1  
    

while (eqm>tolerancia):

           

    for i in range(len(x)): 

        yp= np.dot(z[i],w)    #Previsao para calcular o erro         

        erro = y[i]-yp          #Calculo do Erro

        w+=l_rate*erro*z[i]    #Atualiza os pesos

        

    eqmFim=eqmFun(z,y,w)       #EQM "final" que sera utilisado para o calculo do EQM de parada do loop                           

    eqm=abs(eqmFim-eqmIni)      #Calculo do EQM que para o loop

    print("EQM: {} | epocas: {}".format(eqm,epocas))

    #v_eqm.append(eqm)

    eqmIni=eqmFim               #Copias o EQM "final" para o "inicial"

    epocas+=1                   #Adicona 1 ao numero de epocas

    if(epocas==num_epocas):     #Parada para quando atingir o maximo de epocas

        break
def predict(X,w):

    if(sum(w*X)>=0):

        return 1#,sum(w*X)

    else:

        return -1#,sum(w*X)

    

def predict1(X,w):    

        return sum(w*X)

    



#print(predict(Zb[2],w))
ztrei=x_to_g(x,w1,w2,var1,var2)

saida=np.array([])



for i in range(len(ztrei)):

    saida=np.append(saida,predict(ztrei[i],w))

    

acc=accuracy_score(y,saida)

print('O modelo obteve {}% de acurácia.'.format(acc*100))

ztest=x_to_g(xtest,w1,w2,var1,var2)

saida=np.array([])

s=np.array([])

for i in range(len(ztest)):

    saida=np.append(saida,predict(ztest[i],w))

    s=np.append(s,predict1(ztest[i],w))

    

acc=accuracy_score(ytest,saida)

print('O modelo obteve {}% de acurácia.'.format(acc*100))

print("saida pre\n",s)

print("saida pos\n",saida)

print("W",w)