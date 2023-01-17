import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder # Change values of labels to 0 and 1

import seaborn as sns  

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import scale

import random

import operator
table = pd.read_csv('../input/genre-dataset/genre_dataset.txt')

table = table[(table.genre.str.contains("jazz and blues"))|

        (table.genre.str.contains("soul and reggae"))]



for i in range (4,len(table.columns)):

    col = table.iloc[:,[i]].values

    table.iloc[:,[i]] = scale(col)

    

# Jazz es -1 en tanto Reggae es 1

le = LabelEncoder()

table['genre'] = le.fit_transform(table[['genre']])

table['genre'].values[table['genre'].values == 0] = -1 # Cambiar las etiquetas 0 por -1

table.head()
X_train ,X_test = train_test_split(table,test_size=0.2)



x_train = X_train.iloc[:,4:].values

y_train = X_train.iloc[:,0]

x_test = X_test.iloc[:,4:].values

y_test = X_test.iloc[:,0]
# Inicializar los alpha como con 0.5

alpha = np.full((1, y_train.shape[0]), 0.5)

print ('Dimensiones matriz de alfas',alpha.shape)

# Inicializar b como 0

b=0

# Definir C (cte regularización)

C=2

# Cambiar la forma del vector y_train

y_trainT = np.atleast_2d(y_train)

y_trainT = np.transpose(y_trainT)

print ('Dimensiones matriz de etiquetas de entrenamiento',y_trainT.shape)

# Cambiar la forma del vector y_test

y_testT = np.atleast_2d(y_test)

y_testT = np.transpose(y_testT)
# Definir la función del kernel. Para kernel polinomial:

def kernel(xi,xj):

    d = 3

    gamma = 0.01

    k = gamma*(np.dot(xi,xj))**d

    return k
from IPython.display import Image

import os

Image("../input/polykernel/Capture.PNG")
# Matriz K

K = np.zeros((y_train.shape[0], y_train.shape[0]))



for i in range (y_train.shape[0]-1):

    for j in range (y_train.shape[0]-1):

        K[i,j]=kernel(x_train[i],x_train[j])
# Definir la función f(xi) para definir con ello el márgen de clasificación

def f(i):

    for j in range (y_trainT.shape[0]-1):

        total = alpha[0,i]*y_trainT[i]*K[i,j]

    return total+b



# Definir el márgen de clasificación

def E(i):

    return f(i)-y_trainT[i]
# Escoger el alpha i entre aquellos que no cumplan las condiciones de KKT.

def select_xi():

    KKT=1 # KKT=1 significa que se cumple la condición de KKT para este alpha

    x=0

    error = 0.01 # Error aceptado

    while KKT==1:

        i=random.randint(0, y_train.shape[0]-2)

        if alpha[0,i]==0:

            if f(i)*y_trainT[i]<1+error:

                KKT=0

        if alpha[0,i]>0-error and alpha[0,i]<C+error:

            if f(i)*y_trainT[i]!=1:

                KKT=0

        if alpha[0,i]==C:

            if f(i)*y_trainT[i]>1-error:

                KKT=0

        x+=1

        if x==y_train.shape[0]*1.5:

            i=-1

            break

    return i
# Se decide emplear este ya que la función select_xj toma demasiado tiempo

def select_xj(i):

    M = np.zeros((20,2))

    for x in range (20):

        j=random.randint(0, y_train.shape[0]-2)

        M[x,0] = j

        M[x,1] = abs(E(i)-E(j))

    index = max(enumerate(M[:,1]), key=operator.itemgetter(1))

    return int(M[index[0],0])
# Límite inferior de la optimización

def L(i,j):

    if y_trainT[i]==y_trainT[j]:

        x = max(0, alpha[0,j]+alpha[0,i]-C)

    if y_trainT[i]!=y_trainT[j]:

        x = max(0, alpha[0,j]-alpha[0,i])

    return x



# Límite superior de la optimización

def H(i,j):

    if y_trainT[i]==y_trainT[j]:

        x = min(C, alpha[0,j]+alpha[0,i])

    if y_trainT[i]!=y_trainT[j]:

        x = min(C, C+alpha[0,j]-alpha[0,i])

    return x
# Determinar alpha j nuevo

def new_alphaj(i,j):

    nu = 2*kernel(x_train[i],x_train[j])-kernel(x_train[i],x_train[i])-kernel(x_train[j],x_train[j])

    a_j = alpha[0,j]-(y_trainT[j]*E(i)-E(j))/nu

    h = H(i,j)

    l = L(i,j)

    if a_j >= h:

        a_j = h

    if a_j <= l:

        a_j = l

    return float(a_j)  
# Calcular nuevo alpha i

def new_alphai(i, j, a_j):

    a_i = alpha[0,i]+y_trainT[i]*y_trainT[j]*(alpha[0,i]-a_j)

    return float(a_i)
def iteracion(alpha):

    x=0

    while x<2000:

        # Seleccionar los alphas

        i = select_xi()

        if i==-1:

            print ('broken')

            break

        j = select_xj(i)

        a_j = new_alphaj(i,j)

        a_i = new_alphai(i,j,a_j)

        # Reemplazar a_i y a_j

        #print ('i = ', i)

        #print ('j = ', j)

        #print (alpha[0,i], ' --> ', a_i)

        #print (alpha[0,j], ' --> ', a_j)

        alpha[0,i] = a_i

        alpha[0,j] = a_j

        total = alpha[0,i]*y_trainT[i]*K[i,j]

        b = total - y_trainT[i]

        x+=1

        if x%500==0:

            print(x)

    return alpha
alpha_final = iteracion(alpha)
df=pd.DataFrame(alpha_final)
df.loc[:, (df != 0.5).any(axis=0)]
# Definir la función f(xi) para definir con ello el márgen de clasificación

def f_test(i):

    for j in range (y_testT.shape[0]-1):

        total = alpha[0,i]*y_testT[i]*K[i,j]

    return total+b



# Definir el márgen de clasificación

def E_test(i):

    return f(i)-y_testT[i]
# Calcular el error con el set de prueba

# Conteo de errores

suma=0

for i in range (y_testT.shape[0]-1):

    if f_test(i)>0:

        ftest=1

    if f_test(i)<0:

        ftest=-1

    if ftest!=int(y_testT[i]):

        suma+=1



print ('Fraccion de errores sin tener en cuenta márgen: ', suma/y_testT.shape[0] )