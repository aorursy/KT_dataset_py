# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

from fastai.basics import *
n=100

#sütu ekle

x = torch.ones(n,2) 

x[:,0].uniform_(-1.,1)

x[:10]
a = tensor(3.,2); a

#type(a
y = x@a + torch.rand(n)



plt.scatter(x[:,0], y);
#(x-y)2/n

def mse(y_hat, y):

    return ((y_hat-y)**2).mean()
a = tensor(-1.,1)
y_hat = x@a

mse(y_hat, y)
plt.scatter(x[:,0],y)

plt.scatter(x[:,0],y_hat);
# a ne olmalı
a = nn.Parameter(a); 
def update():

    y_hat = x@a

    loss = mse(y, y_hat)

    if t % 10 == 0: print(loss)

    loss.backward()

    with torch.no_grad():

        a.sub_(lr * a.grad)

        a.grad.zero_()
lr = 1e-1

for t in range(100): update()
plt.scatter(x[:,0],y)

plt.scatter(x[:,0],x@a);


from matplotlib import animation, rc

rc('animation', html='jshtml')


a = nn.Parameter(tensor(-1.,1))



fig = plt.figure()

plt.scatter(x[:,0], y, c='orange')

line, = plt.plot(x[:,0], x@a)

plt.close()



def animate(i):

    update()

    line.set_ydata(x@a)

    return line,



animation.FuncAnimation(fig, animate, np.arange(0, 100), interval=20)