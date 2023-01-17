# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



dataTrain=pd.read_csv("/kaggle/input/projeto/train.csv")

X=np.array(dataTrain)
X
mapa=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
def vizinhos(mapa,raio):

    vizinhos=[]

    for x1 in range(len(mapa)):

        for y1 in range(len(mapa)):

            vizin = []

            for x2 in range(len(mapa)):

                for y2 in range(len(mapa)):

                    if(mapa[x2][y2] != mapa[x1][y1]):

                        distEucli = math.sqrt(pow(x1-x2, 2) + pow(y1-y2, 2))

                        if(distEucli <= raio):

                            vizin.append(mapa[x2][y2])

            vizinhos.append(vizin)

    return vizinhos

def train(X,l_rate,max_epocas):

    w=np.random.uniform(0,1,(16, 3))

    #X=X/np.max(X)

    norm = np.linalg.norm(X)

    X = X/norm

   # w=w/np.max(X)

    norm = np.linalg.norm(w)

    w = w/norm

    vizin=vizinhos(mapa,1)

    epocas=0

    

    mudancaMin=0.0000001

    mudanca=0.02

    while(abs(mudanca)>=mudancaMin):

        mudanca=0

        vencedor=np.array([])

        for i in range(len(X)):

            dists = np.array([])

            for n in range(len(w)):

                dists = np.append(dists, np.linalg.norm(X[i]-w[n]))   

            vencedor=np.argmin(dists)                     

            w[vencedor]+= l_rate*(X[i]-w[vencedor])

            mudanca+= l_rate*(X[i]-w[vencedor])        

            for j in vizin[vencedor]:

                w[j-1]+= (l_rate/2)*(X[i]-w[j-1])  

            for j in vizin[vencedor]:  

                norm = np.linalg.norm(w[j-1])

                w[j-1] = w[j-1]/norm                

        

        epocas+=1     

        mudanca=sum(mudanca)

        print(epocas)

        if(epocas==max_epocas):           

            

            print(abs(mudanca))

            break

    return w





    

   



            

w=train(X, 0.001,1000)          

vencedor=np.array([])



for i in range(len(X)):

    dists = np.array([])

    for n in range(len(w)):

        dists = np.append(dists, np.linalg.norm(X[i]-w[n]))              

    vencedor=np.append(vencedor,np.argmin(dists))

dic={"classe A": vencedor[:20],

    "classe B": vencedor[20:60],

    "classe C":vencedor[61:]}

print("Classe A:\n",dic["classe A"])

print("Classe B:\n",dic["classe B"])

print("Classe C:\n",dic["classe C"])
print(set(dic["classe A"]))

print(set(dic["classe B"]))

print(set(dic["classe C"]))
def predict(X,w):

    dists=np.array([])

    for n in range(len(w)):

        dists = np.append(dists, np.linalg.norm(X-w[n]))

    vencedor=np.argmin(dists)

    if(vencedor in dic["classe A"]):

        return "classe A"

    elif(vencedor in dic["classe B"]):

        return "classe B"

    elif(vencedor in dic["classe C"]):

        return "classe C"

    
for i in range(len(X)):

    print(i,":",predict(X[i],w))
dataTest=pd.read_csv("/kaggle/input/projeto/test.csv")

xTest=np.array(dataTest)

print(xTest)
for i in range(len(xTest)):

    print(i,":",predict(xTest[i],w))