import numpy as np

import random

import matplotlib.pyplot as plt
def diff(ss,ee):

    y=[]

    for i in ss:

        if i not in ee:

            y.append(i)

    return y
c=[(0,0)]

xl=[0]

yl=[0]

for i in range(1,1000):

    s=[(xl[i-1]-1,yl[i-1]),(xl[i-1]+1,yl[i-1]),(xl[i-1],yl[i-1]-1),(xl[i-1],yl[i-1]+1)]

    ss=diff(s,c)

    if ss==[]:

        print("Possibilities have ended. Total steps move = ",i)

        break

    else:

        st=random.choice(ss)

        y = st[1]

        x = st[0]

        c.append(st)

        xl.append(x)

        yl.append(y)

        #print(x,y,s)



plt.plot(xl, yl)

plt.title('Biased random Walk in 2D')

plt.ylabel('y')

plt.xlabel('x')

plt.show()