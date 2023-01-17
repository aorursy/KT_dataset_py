 

import numpy as np  

import pandas as pd  

import sys

np.set_printoptions(threshold=sys.maxsize)

from sympy import KroneckerDelta as kron

from keras import backend as K

from keras.layers import  SimpleRNNCell

import tensorflow as tf

import matplotlib.pyplot as plt

import pydot

 


n = 6 #общее число работников в организации (кол-во вершин графа)

 

C = np.array([[0,0.5,0.25,0.75,0,1], #симметричная матрица 6х6, содержащая затраты каждого из ребер графа

              [0.5,0,0,0.5,0.2,0],

              [0.25,0,0,0,0.1,0.25],

              [0.75,0.5,0,0,0,0],

              [0,0.2,0.1,0,0,0],

              [1,0,0.25,0,0,0]])



P = np.array([[1,0,0,0,1,0], #симметричная матрица 6х6, содержащая несуществующие ребра графа

              [0,1,1,0,0,1],

              [0,1,1,1,0,0],

              [0,0,1,1,1,1],

              [1,0,0,1,1,1],

              [0,1,0,1,1,1]])

 

#константы:

mu1 =950#3000

mu2 = 475#480

mu3 = 1500#50000

mu4 = 2500#15000

mu5 = 2500#50000

A = 0.0001

B = 0.00001

D = 0.00001



director = 0 #номер узла, соответствующего топ-менеджеру в иерархии

executors = [3,4,5] #номера узлов исполнителей, до которых необходимо найти оптимальный путь 

 
def print_graph(): #функция, визуализирующая исходный исходные данные (все вершины графа и связи между ними)

    graph = pydot.Dot(graph_type='graph')

    node0 = pydot.Node("0")

    node1 = pydot.Node("1")

    node2 = pydot.Node("2")

    node3 = pydot.Node("3")

    node4 = pydot.Node("4")

    node5 = pydot.Node("5")

 

    graph.add_node(node0)

    graph.add_node(node1)

    graph.add_node(node2)

    graph.add_node(node3)

    graph.add_node(node4)

    graph.add_node(node5)

    

    graph.add_edge(pydot.Edge(node0,node3, label = "0.75"))

    graph.add_edge(pydot.Edge(node0,node1, label = "0.5"))

    graph.add_edge(pydot.Edge(node0,node2, label = "0.25"))

    graph.add_edge(pydot.Edge(node0,node5, label = "1"))

    graph.add_edge(pydot.Edge(node1,node3, label = "0.5"))

    graph.add_edge(pydot.Edge(node1,node4, label = "0.2"))

    graph.add_edge(pydot.Edge(node2,node4, label = "0.1"))

    graph.add_edge(pydot.Edge(node2,node5, label = "0.25"))

    

    graph.write_png('initial_graph.png')
print_graph()
IN = np.full((n,n), 0.5 + np.random.uniform(-0.00001, 0.00001, (n,n))) #вектор входных значений нейронов

print(IN)

IN.shape = (n*n,1)
weights = np.zeros((n*n,n*n)) #матрица весовых коэффициентов нейронов

for x in range(n):

    for i in range(n):

        for y in range(n):

            for j in range(n):

                p = x*n+i

                q = y*n+j

                weights[p,q] = mu2*kron(x, y)*kron(i, j) - mu4*kron(x, y) - mu4*kron(i, j) + mu4*kron(j, x) + mu4*kron(i, y)

    

print(weights)                 

                    
def weights_init(shape, dtype='float32'): #инициализатор весов нейронов

    return K.variable(value=weights, dtype=dtype)
def input_tensor(IN): #инициализатор входных значений нейронов

    return K.variable(value=IN, dtype='float32')
class RNN(SimpleRNNCell): #сеть Хопфилда для конкретного исполнителя

    def __init__(self,**kwargs):

        super(RNN, self).__init__(**kwargs)

        self.prev_in = [] #входные значения с предыдущих итераций

        self.energies = [] #значения функции энергии

        

    def build(self,input_shape):

        super(RNN, self).build(input_shape)

        

    def threshold(self, out, threshold_value=0.5): #пороговая функция сети

        return K.cast(K.greater(K.clip(out, 0, 1), threshold_value), dtype='float32')

    

    def calcul_energy(self,m,OUT, out_from_others = []): #функция вычисления значения энергии

        ftype = np.ones((n,n))

        if out_from_others != []:

            sum = np.zeros((n,n))

            for out in out_from_others:

                sum = sum + out

            ftype = 1/(1+sum)    

        OUT = K.reshape(OUT, (n,n))

        E1 = 0 

        E2 = 0

        E3 = 0

        E4 = 0

        E5 = 0

        for x in range(n):

            for i in range(n):

                if (i != x) and (x!=m and i!=director):

                    E1 = E1 + C[x][i]*ftype[x][i]*OUT[x][i]  

                    E3 = E3+P[x][i]*OUT[x][i]

                if i != x:

                    E2 = E2 + OUT[x][i]*(1-OUT[x][i])

                 

                

        out1=0

        out2=0

        square=0

        for x in range(n):

            for i in range(n):

                if i != x:

                    out1 = out1+OUT[x][i]

                    out2 = out2+OUT[i][x]

            square = (out1-out2)*(out1-out2)

            E4 = E4 +square

        

        E5 = 1-OUT[m][director]

    

        E = mu1*E1 + mu2*E2 + mu3*E3 + mu4*E4 + mu5*E5

        return E

        

    def call(self,inputs, training=None ): #функция, совершающая 1 итерацию сети

        if self.prev_in==[]:

            out = self.activation(inputs)

         

            h = K.dot(  self.kernel,out)

     

            h = h+ self.bias

            new_inputs = inputs + 0.00001*h

            self.prev_in.append(inputs)

            self.prev_in.append(new_inputs)

            return new_inputs

        else:

            out = self.activation(inputs)

            h = K.dot(  self.kernel,out)

            h = h+ self.bias

            if len(self.prev_in)<3:

                new_inputs = self.prev_in[len(self.prev_in)-1] -A*self.prev_in[len(self.prev_in)-2]+ D*h

            else:

                new_inputs = self.prev_in[len(self.prev_in)-1] -A*self.prev_in[len(self.prev_in)-2] - B*self.prev_in[len(self.prev_in)-3]+ D*h

            self.prev_in.append(new_inputs)

            return new_inputs

         

             

        return new_out

    

    def iterate(self, m,inputs,  num_iters=100,out_from_others = []): #вызов функции call несколько раз

        for iter in range(num_iters):

            out = self.call(inputs)

            energy = self.calcul_energy(m,out,out_from_others)

            self.energies.append(energy)

            inputs = out

             

        return self.threshold(out)
out_from_others = [] #результаты, получившиеся для всех исполнителей кроме текущего

 
for m in executors:

    def bias_init(shape, dtype='float32'): #инициализатор смещения нейронов (bias)

        bias = np.zeros((n,n))

        for x in range (n):

            for i in range (n):

                if (x==m and i==director):

                    bias[x][i] = mu5/2-mu2/2

                else:

                    bias[x][i] = -mu1/2*C[x][i]-mu3/2*P[x][i]-mu2/2 

     

        bias=np.concatenate(bias)

        b = np.zeros((n*n,1))

 

        for x in range (n*n):

            b[x][0]= bias[x]

        return K.variable(value=b, dtype=dtype)





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

    mOUT = K.eval(mRNN.iterate(m=m,inputs=input_tensor(IN), out_from_others=out_from_others)) #работа сети

    mOUT.shape = (n,n)

    print("Результаты для исполнителя "+str(m)+":") 

    print(mOUT)

    

    out_from_others.append(mOUT)

    

    #визуализация функции энергии

    iterations = np.arange(0,100,10)

    energy = np.zeros((10,))

    j=0

    for i in range(0,len(mRNN.energies),10):

        energy[j] = K.eval(mRNN.energies[i])

        j=j+1

        

    plt.plot(iterations,energy, marker='o')

    plt.title("График функции энергии для исполнителя "+str(m))

    plt.xlabel("количество итераций")

    plt.ylabel("значение функции энергии")

    plt.show()

    

     

     
def print_result_graph(): #визуализация получившегося графа

    graph = pydot.Dot(graph_type='graph')

      

    node0 = pydot.Node("0")

    

    node2 = pydot.Node("2")

    node3 = pydot.Node("3")

    node4 = pydot.Node("4")

    node5 = pydot.Node("5")

 

    graph.add_node(node0)

     

    graph.add_node(node2)

    graph.add_node(node3)

    graph.add_node(node4)

    graph.add_node(node5)

     

    graph.add_node(pydot.Node("0"))

    graph.add_node(pydot.Node("2"))

    graph.add_node(pydot.Node("3"))

    graph.add_node(pydot.Node("4"))

    graph.add_node(pydot.Node("5"))

     

    

    graph.add_edge(pydot.Edge(node0,node3, label = "0.75"))

     

    graph.add_edge(pydot.Edge(node0,node2, label = "0.25"))

     

    graph.add_edge(pydot.Edge(node2,node4, label = "0.1"))

    graph.add_edge(pydot.Edge(node2,node5, label = "0.25"))

    

    graph.write_png('result_graph.png')

    
print_result_graph()