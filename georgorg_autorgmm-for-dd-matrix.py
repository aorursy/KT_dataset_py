# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import matplotlib.ticker as ticker



plot_size = (20, 10)

large = 22; med = 16; small = 12

params = {'axes.titlesize': large,

          'legend.fontsize': med,

          'figure.figsize': plot_size,

          'axes.labelsize': med,

          'axes.titlesize': med,

          'xtick.labelsize': med,

          'ytick.labelsize': med,

          'figure.titlesize': large}

plt.rcParams.update(params)
D_0 = np.array([[1, 3, 0.333, 5, 1, 0.5, 2], [0.333, 1, 0.1111, 2, 0.3333, 0.167, 0.5], [3, 9, 1, 0.111, 3, 1, 7], [0.2, 0.5, 9, 1, 0.2, 0.142, 0.5], [1, 3, 0.333, 5, 1, 0.5, 2], [2, 6, 1, 7, 2, 1, 4], [0.5, 2, 0.142, 7, 0.5, 0.25, 1]])

pd.DataFrame(D_0)
import copy

def RGMM(D):

    w = []

    for i in D:

        element = 1

        for j in i:

            element = element * j

        w.append(element**(1/7))

    return w/np.sum(w)



def CR(D):

    return float((np.max(np.linalg.eig(D)[0]) - np.size(D[0]))/(np.size(D[0])-1))



def uzgod(D, l):

    w = RGMM(D)

    D_ = D

    for i in range(np.size(D[0])):

        for j in range(np.size(D[0])):

            D_[i][j] = D_[i][j]**l*(w[i]/w[j])**(1-l)

    return D

def efectivity(D_0, D_1):

    return np.max(abs(D_1 - D_0)), (np.sum((D_1 - D_0)**2))**0.5/np.size(D_1[0])
l = 0.8

w = RGMM(D_0)

print(CR(D_0))

D_1 = uzgod(copy.copy(D_0), l)

print(efectivity(D_0, D_1))

print(pd.DataFrame(D_1 - D_0))

print(efectivity(D_0, D_1))
def lab(D_0, l=0.1):

    Ds = []

    W = []

    Cr = []

    sigma = []

    delta = []

    x = []

    D = copy.copy(D_0)

    for i in range(1000):

        x.append(i)

        print('===============================')

        print('K = ', i)

        D_1 = uzgod(copy.copy(D), l)

        Ds.append(D_1)

        print(pd.DataFrame(D_1))

        cr = CR(D_1)

        Cr.append([cr])

        print('CR = ',cr)

        w = RGMM(D_1)

        W.append(w)

        print('----------W-------------')

        print(w)

        sg, dl = efectivity(D_0, D_1)

        sigma.append(sg)

        delta.append(dl)

        print('---------sigma,delta----')

        print(sg)

        print(dl)

        if cr<0.1:

            break

        D = D_1

    return Ds, W, Cr, sigma, delta, x

#          0    1   2    3    4     5

def ploter(l_, name):

    f = plt.figure()

    plt.plot(l_[5], l_[4], label='Дельта')

    plt.plot(l_[5], l_[3], label='сигма')

    plt.plot(l_[5], l_[2], label='CR')

    

    legend = plt.legend(loc='best', shadow=True, fontsize='x-large')



    # Put a nicer background color on the legend.

    #legend.get_frame().set_facecolor('C1')



    plt.title(name)

    plt.show()
log = []

counter = 0

ll = [0.5, 0.6, 0.7, 0.8, 0.9]

for i in range(np.size(ll)):

    log.append(lab(D_0, l=ll[i]))

    ploter(log[counter], name='alfa = '+str(ll[i]))

    counter = counter+1
log = []

counter = 0

ll = [0.5, 0.6, 0.7, 0.8, 0.9]

for i in range(np.size(ll)):

    log.append(lab(D_0, l=ll[i]))
(np.max(np.linalg.eig(D_0)[0]) - np.size(D_0[0]))/(np.size(D_0[0])-1)
import random

random.seed(9001)

def create_D(size=3, state='random1', k = -1):

    D = np.zeros((size,size))

    for i in range(size):

        D[i][i] = 1

    for i in range(size):

        for j in range(i):

            if state == 'random1':

                D[j][i] = random.uniform(0, 10)

            if state == 'random2':

                D[j][i] = random.randint(1, 10)

            D[i][j] = 1/D[j][i]

    if k>-1:

        for i in range(size):

            D[k][i] = 1

            D[i][k] = 1

    return D

create_D(size=3, state='random1', k = 1)        
CR_1_n = []

CR_2_n = []

CR_one_miss_n_int = []

CR_one_miss_n_float = []

N = 700

Nn = 25

for j in range(2,Nn):

    CR_1 = []

    CR_2 = []

    CR_one_int = []

    CR_one_float = []

    for i in range(N+N*j):

        D = create_D(size=j, state='random1')

        CR_1.append(CR(D))

        D = create_D(size=j, state='random2')

        CR_2.append(CR(D))

        D = create_D(size=j, state='random1', k = random.randint(1,j-1))

        CR_one_int.append(CR(D))

        D = create_D(size=j, state='random2', k = random.randint(1,j-1))

        CR_one_float.append(CR(D))

    CR_1_n.append(CR_1)

    CR_2_n.append(CR_2)

    CR_one_miss_n_int.append(CR_one_int)

    CR_one_miss_n_float.append(CR_one_float)



fig, ax = plt.subplots()



plt.plot([j for j in range(2,Nn)], [np.mean(k) for k in CR_1_n], label='CR1 changes by size')  



#  Устанавливаем интервал основных делений:

ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

#  Устанавливаем интервал вспомогательных делений:

ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))



#  Тоже самое проделываем с делениями на оси "y":

ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))



legend = plt.legend(loc='best', shadow=True, fontsize='x-large')





fig, ax = plt.subplots()



plt.plot([j for j in range(2,Nn)], [np.mean(k) for k in CR_2_n], label='CR2 changes by size')  



#  Устанавливаем интервал основных делений:

ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

#  Устанавливаем интервал вспомогательных делений:

ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))



#  Тоже самое проделываем с делениями на оси "y":

ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))



legend = plt.legend(loc='best', shadow=True, fontsize='x-large')





fig, ax = plt.subplots()



plt.plot([j for j in range(2,Nn)], [np.mean(k) for k in CR_2_n], label='CR2 changes by size') 

plt.plot([j for j in range(2,Nn)], [np.mean(k) for k in CR_1_n], label='CR1 changes by size')  



#  Устанавливаем интервал основных делений:

ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

#  Устанавливаем интервал вспомогательных делений:

ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))



#  Тоже самое проделываем с делениями на оси "y":

ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))



legend = plt.legend(loc='best', shadow=True, fontsize='x-large')







print('============================================================')

fig, ax = plt.subplots()



plt.plot([j for j in range(2,Nn)], [np.mean(k) for k in CR_one_miss_n_int], label='CR1 changes by size with one missing')  



#  Устанавливаем интервал основных делений:

ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

#  Устанавливаем интервал вспомогательных делений:

ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))



#  Тоже самое проделываем с делениями на оси "y":

ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))



legend = plt.legend(loc='best', shadow=True, fontsize='x-large')





fig, ax = plt.subplots()



plt.plot([j for j in range(2,Nn)], [np.mean(k) for k in CR_one_miss_n_float], label='CR2 changes by size with one missing')  



#  Устанавливаем интервал основных делений:

ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

#  Устанавливаем интервал вспомогательных делений:

ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))



#  Тоже самое проделываем с делениями на оси "y":

ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))



legend = plt.legend(loc='best', shadow=True, fontsize='x-large')





fig, ax = plt.subplots()



plt.plot([j for j in range(2,Nn)], [np.mean(k) for k in CR_one_miss_n_float], label='CR2 changes by size with one missing') 

plt.plot([j for j in range(2,Nn)], [np.mean(k) for k in CR_one_miss_n_int], label='CR1 changes by size with one missing')  



#  Устанавливаем интервал основных делений:

ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

#  Устанавливаем интервал вспомогательных делений:

ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))



#  Тоже самое проделываем с делениями на оси "y":

ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))



legend = plt.legend(loc='best', shadow=True, fontsize='x-large')



print('==============================comperison=====================================')



fig, ax = plt.subplots()



plt.plot([j for j in range(2,Nn)], [np.mean(k) for k in CR_one_miss_n_float], label='CR2 changes by size with one missing') 

plt.plot([j for j in range(2,Nn)], [np.mean(k) for k in CR_one_miss_n_int], label='CR1 changes by size with one missing')

plt.plot([j for j in range(2,Nn)], [np.mean(k) for k in CR_2_n], label='CR2 changes by size') 

plt.plot([j for j in range(2,Nn)], [np.mean(k) for k in CR_1_n], label='CR1 changes by size')



#  Устанавливаем интервал основных делений:

ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

#  Устанавливаем интервал вспомогательных делений:

ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))



#  Тоже самое проделываем с делениями на оси "y":

ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))



legend = plt.legend(loc='best', shadow=True, fontsize='x-large')
[np.mean(k) for k in CR_1_n]
fig, ax = plt.subplots()



plt.plot([j for j in range(2,25)], [np.mean(k) for k in CR_1_n], label='CR1 changes by size')  



#  Устанавливаем интервал основных делений:

ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

#  Устанавливаем интервал вспомогательных делений:

ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))



#  Тоже самое проделываем с делениями на оси "y":

ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))



legend = plt.legend(loc='best', shadow=True, fontsize='x-large')
fig, ax = plt.subplots()



plt.plot([j for j in range(2,25)], [np.mean(k) for k in CR_2_n], label='CR2 changes by size')  



#  Устанавливаем интервал основных делений:

ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

#  Устанавливаем интервал вспомогательных делений:

ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))



#  Тоже самое проделываем с делениями на оси "y":

ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))



legend = plt.legend(loc='best', shadow=True, fontsize='x-large')
min_CR_1 = [[j for j in range(2,25)], [np.mean(k) for k in CR_1_n]]

min_CR_2 = [[j for j in range(2,25)], [np.mean(k) for k in CR_2_n]]
pd.DataFrame(min_CR_1)
pd.DataFrame(min_CR_2)