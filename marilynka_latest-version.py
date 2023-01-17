import numpy as np  

#import pandas as pd  

#import sys

#np.set_printoptions(threshold=sys.maxsize)

from sympy import KroneckerDelta as kron

import tensorflow as tf

 

import matplotlib.pyplot as plt

#import pydot

import time

import random

from multiprocessing import Process, Queue
from tensorflow import keras

from tensorflow.keras import backend as K

from tensorflow.keras.layers import  SimpleRNNCell
n = 20 #общее число работников в организации (кол-во вершин графа)

C = np.random.randint(0, 11, (n, n)) #затраты ребер графа (целые значения от 0 до 10)

#np.around(np.random.uniform(0, 1, (n,n)),decimals=2)  

C = np.tril(C) + np.tril(C, -1).T #матрица затрат должна быть симметричной

np.fill_diagonal(C, 0, wrap=False) 

#print(C)
P = np.random.randint(0, 2, (n, n)) #несуществующие ребра графа

P = np.tril(P) + np.tril(P, -1).T

#P = (P + P.T)/2

np.fill_diagonal(P, 1, wrap=False)

print('матрица несуществующих ребер')

print(P)
for i in range(n):

    for j in range(n):

        if P[i][j] == 1:

            C[i][j] = 10

print('матрица затрат')

print(C)
#константы:

mu1 =950#3000

mu2 = 475#480

mu3 = 1500#50000

mu4 = 2500#15000

mu5 = 2500#50000

A = 0.0001

B = 0.00001

D = 0.00001

ex_number = np.around(n*2/3, decimals=0) # 2/3 - кол-во исполнителей относительно общего числа рработников

director = 0 #номер узла, соответствующего топ-менеджеру в иерархии

executors = random.sample(range(1,n), int(ex_number)) #случайно выбранные индексы исполнителей  

print(executors)
weights = np.zeros((n*n,n*n))  
def init_weights():

    start = time.time()

      

    for x in range(n):

        for i in range(n):

            for y in range(n):

                for j in range(n):

                 

                    weights[x*n+i,y*n+j] = mu2*kron(x, y)*kron(i, j) - mu4*kron(x, y) - mu4*kron(i, j) + mu4*kron(j, x) + mu4*kron(i, y)

    

    finish = time.time()

    result = finish - start

    print("Время на инициализацию матрицы весов: " + str(result) + " секунд")

    return weights
weights = init_weights()
def weights_init(shape, dtype='float32'): #инициализатор весов нейронов               

    return K.variable(value=weights, dtype=dtype)
def energy(m,OUT, ftype,out_from_others = []):

    OUT = K.reshape(OUT, (n,n))

    #ftype = ftype + OUT

     

    E1 = 0 

    E2 = 0

    E3 = 0

    E4 = 0

    E5 = 0

    for x in range(n):

        for i in range(n):

            if (i != x) and (x!=m and i!=director):

                E1 = E1 + C[x][i]*(1/(1+ftype[x][i]))*OUT[x][i]  

                E3 = E3+P[x][i]*OUT[x][i]

            if i != x:

                E2 = E2 + OUT[x][i]*(1-OUT[x][i])

                 

                

    out1=0

    out2=0

    #square=0

    for x in range(n):

        for i in range(n):

            if i != x:

                out1 = out1+OUT[x][i]

                out2 = out2+OUT[i][x]

        #square = (out1-out2)*(out1-out2)

        E4 = E4 + (out1-out2)*(out1-out2)

        

    E5 = 1-OUT[m][director]

    

    E = mu1*E1 + mu2*E2 + mu3*E3 + mu4*E4 + mu5*E5

    

    OUT = K.reshape(OUT, (n*n,1))

    return E

    
def bias_init(shape, dtype='float32'): #инициализатор смещения нейронов (bias)

        bias = np.zeros((n,n))

        for x in range (n):

            for i in range (n):

                if (x==m and i==director):

                    bias[x][i] = mu5/2-mu2/2

                else:

                    bias[x][i] = -mu1/2*C[x][i]*ftype[x][i]-mu3/2*P[x][i]-mu2/2 

     

        bias=np.concatenate(bias)

        b = np.zeros((n*n,1))

 

        for x in range (n*n):

            b[x][0]= bias[x]

        #b = b.shape((n*n,1))

        return K.variable(value=b, dtype=dtype)
def input_tensor():  

        IN = np.full((n,n), 0.5 + np.random.uniform(-0.00001, 0.00001, (n,n)))  # 0.00001 

        IN.shape = (n*n,1)

        return K.variable(value=IN, dtype='float32')
class RNN(SimpleRNNCell): #сеть Хопфилда для конкретного исполнителя

    def __init__(self,**kwargs):

        super(RNN, self).__init__(**kwargs)

        #self.weights = weights

        self.prev_in = [] #входные значения с предыдущих итераций

        self.energies = [] #значения функции энергии

        self.prev_energy = np.Infinity #значение энергии на предыдущей итерации

        self.worker = -1 #номер узла исполнителя, для которого используется сеть

        self.stop = False #для остановки в случае увеличения энергии

        self.result = input_tensor() #выходной сигнал после определенной итерации

        

    def build(self,input_shape):

        self.kernel = weights_init((n*n,n*n)) #self.add_weight(shape=(input_shape[-1], self.units),

                        #              name='kernel',

                         #             initializer=self.kernel_initializer  )

          

        self.bias =  bias_init((n,n))

        

        self.built = True

        #super(RNN, self).build(input_shape)

        

    def threshold(self, out, threshold_value=0.5): #пороговая функция сети

        return K.cast(K.greater(K.clip(out, 0, 1), threshold_value), dtype='float32')

     

     

    def call(self,inputs,states=[], training=None ): #функция, совершающая 1 итерацию сети

        inputs = K.reshape(inputs, (1,n*n))

         

        output = K.dot(  K.reshape( inputs,(1,n*n)),self.kernel) 

        #output = output + self.bias

        

        output = K.bias_add(output,  K.reshape(self.bias,(n*n,)))

        output = self.activation(output)

        inputs = output

          

        if len(self.prev_in) ==1:

            inputs = self.prev_in[0] + D*output

             

        elif len(self.prev_in) ==2:

            inputs =  self.prev_in[1] - A*self.prev_in[0] + D*output

            

        elif len(self.prev_in) > 2:

            inputs = self.prev_in[len(self.prev_in)-1] - A*self.prev_in[len(self.prev_in)-2] - B*self.prev_in[len(self.prev_in)-3]+ D*output

             

        self.prev_in.append(inputs)

        

        return  inputs 
def create_RNN(m):

    mRNN = RNN(                         #создание экземпляра нейронной сети для конкретного исполнителя

               units=n,

               activation='sigmoid',

               use_bias=True,

               kernel_initializer=weights_init,

               recurrent_initializer=weights_init,

               bias_initializer=bias_init,

               kernel_regularizer=None,

               recurrent_regularizer=None,

               bias_regularizer=None,

               kernel_constraint=None,

               recurrent_constraint=None,

               bias_constraint=None,

               dropout=0.,

               recurrent_dropout=0.,)



    mRNN.build((n*n,)) #формирование сети

    #all_RNN.append(mRNN)

    mRNN.worker = m

    return mRNN
def one_iteration(i,q): #

    all_RNN[i].bias = bias_init(shape = (n,n)) 

    all_RNN[i].prev_in.append( all_RNN[i].call(inputs=all_RNN[i].result)) #

    #mOUT = all_RNN[i].threshold(all_RNN[i].call(inputs=all_RNN[i].result))

    #global ftype 

    

    if (K.eval(all_RNN[i].prev_energy) - K.eval(energy(all_RNN[i].worker,all_RNN[i].call(inputs=all_RNN[i].result),ftype)))<1e-5:

                                                       #all_RNN[i].prev_in[len(all_RNN[i].prev_in)-1],ftype)):

        

        ''' print('stop')

        print(K.eval(all_RNN[i].prev_energy))

        print(K.eval(energy(all_RNN[i].worker,mOUT,ftype)))'''

        all_RNN[i].stop = True

        all_RNN[i].result = all_RNN[i].threshold(all_RNN[i].prev_in[len(all_RNN[i].prev_in)-2])

         

    else:

        all_RNN[i].prev_energy =  energy(all_RNN[i].worker,all_RNN[i].call(inputs=all_RNN[i].result),ftype)

                                         #all_RNN[i].prev_in[len(all_RNN[i].prev_in)-1], ftype) #(iter < 10) or

        all_RNN[i].energies.append(all_RNN[i].prev_energy)

         

        all_RNN[i].result = all_RNN[i].call(inputs=all_RNN[i].result)#all_RNN[i].prev_in[len(all_RNN[i].prev_in)-1]

         

    q.put([all_RNN[i].result,all_RNN[i].prev_energy,all_RNN[i].stop,all_RNN[i].prev_in,all_RNN[i].energies]) 

     

 

    
all_start = time.time()
if __name__ == '__main__':

    ftype = np.zeros((n,n)) #Матрица ftype нейронов

    all_RNN=[]

    for m in executors:

        all_RNN.append(create_RNN(m))

    iterat = 0

     

    #while (K.eval(mRNN.prev_energy) >= K.eval(energy(mRNN.worker,mRNN.result,ftype))for mRNN in all_RNN):

    while not all(mRNN.stop == True for mRNN in all_RNN) and iterat<10:

        #print(iterat)

        if iterat>0:

            for i in range(len(executors)):

                ftype = ftype + K.eval(all_RNN[i].result).reshape(n,n)

        iterat = iterat + 1

        #param = np.Infinity

        queues = []

        #queue = Queue()

        procs = []

        for i in range(len(executors)):

            if (all_RNN[i].stop == False):

                queue = Queue()

                queues.append(queue)

                p = Process(target=one_iteration, args=(i,queue,))

                procs.append(p)     

                p.start()

             

           

                #[all_RNN[i].result,all_RNN[i].prev_energy,all_RNN[i].stop] = queue.get()

                #all_RNN[i].energies.append(all_RNN[i].prev_energy)

            

        for proc in procs:

            proc.join()

        

        j=0

        for i in range(len(executors)):

            if (all_RNN[i].stop == False) :

                 

                [all_RNN[i].result,all_RNN[i].prev_energy,all_RNN[i].stop,all_RNN[i].prev_in,all_RNN[i].energies] = queues[j].get()

                j = j+1

            

            #ftype = ftype + K.eval(all_RNN[i].result).reshape(n,n)

            #print(ftype)

            

        ftype = np.zeros((n,n))

             

     
print(iterat) #сколько было итераций
for i in range(len(executors)):

    print(all_RNN[i].stop)
for i in range(len(executors)):

    OUT = K.eval(all_RNN[i].result)

    OUT.shape = (n,n)

    print("Результаты для исполнителя "+str(executors[i])+":") 

    print(OUT)

    

for mRNN in  all_RNN:

    iterations = np.arange(0,len(mRNN.energies))

#print(iterations)

#print(len(energies))

    e =  np.zeros((len(mRNN.energies),))

#print(e)

    print("energy values")

    for i in range(len(mRNN.energies)):

        e[i] = K.eval(mRNN.energies[i])

         

        print(e[i])

    #print(K.eval(energies[i]))

#j=0

#for i in range(0,len(mRNN.energies),10):

    #energy[j] = K.eval(mRNN.energies[i])

    #j=j+1

        

    plt.plot(iterations,e, marker='o')

    plt.title("График функции энергии для исполнителя "+str(mRNN.worker))

    plt.xlabel("количество итераций")

    plt.ylabel("значение функции энергии")

    plt.show()
all_finish = time.time()

all_result = all_finish - all_start

print("Общее время: " + str(all_result) + " секунд")