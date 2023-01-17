# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rd

from matplotlib import pyplot as plt

!pip install array2gif

from array2gif import write_gif



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def brownscheBew(a, i, j):

    if rd.choice([True, False]):

        if rd.choice([True, False]):

            b = moveup(a, i, j).copy()

            a = b[0]

            i = b[1]

            j = b[2]

        else:

            b = movedown(a, i, j).copy()

            a = b[0]

            i = b[1]

            j = b[2]

    else:

        if rd.choice([True, False]):

            b = moverigth(a, i, j).copy()

            a = b[0]

            i = b[1]

            j = b[2]

        else:

            b = moveleft(a, i, j).copy()

            a = b[0]

            i = b[1]

            j = b[2]

    return [a, i, j]



def moveup(a, i, j):

    if i == 0:

        return [a, i, j]

    else:

        a[i - 1, j] = a[i, j]

        a[i, j] = 0

        return [a, i - 1, j]



def movedown(a, i, j):

    L = a.shape[0]

    if i == L - 1:

        return [a, i, j]

    else:

        a[i + 1, j] = a[i, j]

        a[i, j] = 0

        return [a, i + 1, j]

    

def moverigth(a, i, j):

    L = a.shape[1]

    if j == L - 1:

        return [a, i, j]

    else:

        a[i, j + 1] = a[i, j]

        a[i, j] = 0

        return [a, i, j + 1]



def moveleft(a, i, j):

    if j == 0:

        return [a, i, j]

    else:

        a[i, j - 1] = a[i, j]

        a[i, j] = 0

        return [a, i, j - 1]



def randWalk(a_start, x_start, y_start, K):

    outofBounds = True

    while(outofBounds):

        outofBounds = False

        a = a_start.copy()

        steps = np.empty((K + 1, a.shape[0], a.shape[1]))

        steps[0] = a

        pos = np.empty((K + 1, 2))

        pos[0] = [x_start, y_start]

        i = x_start

        j = y_start

        for k in range(K):

            b = brownscheBew(a, i, j).copy()

            a = b[0]

            i = b[1]

            j = b[2]

            steps[k + 1] = a

            pos[k + 1] = [i, j]

            if np.array_equal(pos[k], pos[k + 1]):

                outofBounds = True

                break

    return [steps, pos]



        
L = 40

K = 1000

a = np.zeros((L, L), dtype = int)

x_start = int(L/2)

y_start = int(L/2)

a[x_start, y_start] = 1

b = randWalk(a, x_start, y_start, K).copy()

steps = b[0]

dataset = np.ndarray((K + 1, L, L, 3), dtype = int)

for n in range(K + 1):

    for i in range(L):

        for j in range(L):

            if steps[n, i, j] == 1:

                dataset[n, i, j] = [255, 0, 0].copy()

            else:

                dataset[n, i, j] = [0, 255, 0].copy()

write_gif(dataset, 'RandWalk.gif', fps=50)
L = 80

N = 1000

K = np.arange(N)

Q = 2000

R = np.zeros(N)

for q in range(Q):

    a = np.zeros((L, L), dtype = int)

    x_start = int(L/2)

    y_start = int(L/2)

    a[x_start, y_start] = 1

    pos = randWalk(a, x_start, y_start, N)[1].copy()

    for i in range(N):

        R[i] = R[i] + (pos[0][0] - pos[i][0]) ** 2 + (pos[0][1] - pos[i][1]) ** 2

R = R / Q

np.save("R.npy", R)

np.save("K.npy", K)
R = np.load("R.npy")

K = np.load("K.npy")



plt.plot(K, R)

plt.show()
def hasNeighbours(a, i, j):

    for h in range(i - 1, i + 2):

        for k in range(j - 1, j + 2):

            if not(h == i and k == j):

                if exists(a, h, k):

                   if a[h, k] == 1:

                    return True

    return False



def exists(a, i, j):

    L = a.shape[0]

    return ((not i < 0) and (not i > L - 1) and (not j < 0) and (not j > L - 1))



def crystal(a, i, j):

    bound = False

    while(not bound):

        if hasNeighbours(a, i, j):

            bound = True

            break    

        b = brownscheBew(a, i, j).copy()

        a = b[0]

        i = b[1]

        j = b[2]

    return a



def randEdge(a):

    L = a.shape[0]

    free = False

    while(not free):

        k = rd.randint(0, L - 1)

        if rd.choice([True, False]):

            if rd.choice([True, False]):

                i = 0

                j = k

            else:

                i = L - 1

                j = k

        else:

            if rd.choice([True, False]):

                i = k

                j = 0

            else:

                i = k

                j = L - 1

        if a[i][j] != 1:

            free = True

    return [i, j]



def crystalSize(a):

    unique, counts = np.unique(a, return_counts=True)

    return counts[1]
L = 101

N = 500

a = np.zeros((L, L), dtype = int)

a[int(L / 2)][int(L / 2)] = 1

while(crystalSize(a) < N):

    pos = randEdge(a)

    i = pos[0]

    j = pos[1]

    a[i][j] = 1

    a = crystal(a, i, j)

np.save("Crystal.npy", a)    
a = np.load("Crystal.npy")

fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis

ax.imshow(a)

plt.show()

fig.savefig('Crystal.png')   # save the figure to file

plt.close(fig)