# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import random

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
nb_patterns = 4

pattern_width = 5

pattern_height = 9
X = np.zeros((nb_patterns, pattern_width * pattern_height))



#Número 1

X[0] = [

    -1, -1,  1,  1, -1,

    -1,  1,  1,  1, -1,

    -1, -1,  1,  1, -1,

    -1, -1,  1,  1, -1,

    -1, -1,  1,  1, -1,

    -1, -1,  1,  1, -1,

    -1, -1,  1,  1, -1,

    -1, -1,  1,  1, -1,

    -1, -1,  1,  1, -1

]



#Número 2

X[1] = [

     1,  1,  1,  1,  1,

     1,  1,  1,  1,  1,

    -1, -1, -1,  1,  1,

    -1, -1, -1,  1,  1,

     1,  1,  1,  1,  1,

     1,  1, -1, -1, -1,

     1,  1, -1, -1, -1,

     1,  1,  1,  1,  1,

     1,  1,  1,  1,  1

]



#Número 3

X[2] = [

     1,  1,  1,  1,  1,

     1,  1,  1,  1,  1,

    -1, -1, -1,  1,  1,

    -1, -1, -1,  1,  1,

     1,  1,  1,  1,  1,

    -1, -1, -1,  1,  1,

    -1, -1, -1,  1,  1,

     1,  1,  1,  1,  1,

     1,  1,  1,  1,  1

]



#Número 4

X[3] = [

     1,  1, -1,  1,  1,

     1,  1, -1,  1,  1,

     1,  1, -1,  1,  1,

     1,  1,  1,  1,  1,

     1,  1,  1,  1,  1,

    -1, -1, -1,  1,  1,

    -1, -1, -1,  1,  1,

    -1, -1, -1,  1,  1,

    -1, -1, -1,  1,  1

]
def plot(X):

    fig, ax = plt.subplots(1, nb_patterns, figsize=(10, 5))



    fig.suptitle('Padrões Inseridos Para Treino', fontsize=16)



    for i in range(nb_patterns):

        ax[i].matshow(X[i].reshape((pattern_height, pattern_width)), cmap='binary')

        ax[i].set_xticks([])

        ax[i].set_yticks([])



    fig.tight_layout()

    plt.show()

plot(X)
def treinamento(X):                

    return np.dot(X.T,X)/len(X[0]) -(np.identity(pattern_width*pattern_height)*(len(X)/len(X[0])))

        
W=treinamento(X)



W
def gerarRuido(num, porc):

    

    vet = np.copy(num)

    size = len(vet)

    qtdPixels = int((size * porc) / 100)

    pixelsToModify = random.sample(range(size), qtdPixels)

    for i in range(qtdPixels):

        p = pixelsToModify[i]

        vet[p] = -1 if vet[p] == 1 else 1

            

    return vet
def funAtivacao(X,w):

    u = X.copy()

    for i in range(len(u)):

         if (np.dot(X,w[i])>0):

                u[i]=1

         else:

                u[i]= -1

    return u
def predict(X,w):

    vAtual=X.copy()

    epocas=0

    vAnterior=0

    while(np.array_equal(vAtual,vAnterior)==False):

        print(epocas+1)

        epocas+=1

        vAnterior=vAtual.copy() 

        vAtual=funAtivacao(X,w)

        

    return vAtual

        
def showPatterns(titlePattern, patternOriginal, patternCorrupted, patternRecovered):

    

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    

    fig.suptitle(titlePattern, fontsize=16)



    ax[0].matshow(patternOriginal.reshape(pattern_height, pattern_width), cmap='binary')

    ax[0].set_title('Imagem Transmitida \n (sem ruídos)')

    ax[0].set_xticks([])

    ax[0].set_yticks([])

    

    ax[1].matshow(patternCorrupted.reshape(pattern_height, pattern_width), cmap='binary')

    ax[1].set_title('Imagem Recebida \n (com ruídos)')

    ax[1].set_xticks([])

    ax[1].set_yticks([])



    ax[2].matshow(patternRecovered.reshape(pattern_height, pattern_width), cmap='binary')

    ax[2].set_title('Imagem Recuperada \n (sem ruídos)')

    ax[2].set_xticks([])

    ax[2].set_yticks([])

    

    fig.tight_layout()

    fig.subplots_adjust(top=0.75)

    

    plt.show()
def testar(titlePattern, padrao, ruidoPorc):    

    original = padrao.copy()    

    ruido = gerarRuido(original, ruidoPorc)    

    recuperado = predict(ruido,W)    

    showPatterns(titlePattern, original, ruido, recuperado)
for i in range(4):

    for j in range(3):

        testar('Padrão Número {} com 20% de ruído'.format(i+1), X[i], 20)
for i in range(4):

    p=20

    for j in range(10):

        testar('Padrão Número {} com {}% de ruído'.format(i+1,p+5), X[i], p)

        p+=5