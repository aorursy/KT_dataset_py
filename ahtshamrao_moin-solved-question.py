import numpy as np

#equation to determine change in AA and BB

AA=150

BB=150

delta=0

print("the initial value of A molecule : ",AA, "and B molecule: ",BB)

#array initially

initial=np.array([[BB],[AA]],np.int64)

delta=(AA*20)/100

AA=AA-delta

BB=BB+delta

delta=(BB*10)/100

AA=AA+delta

BB=BB-delta

#array after 1 hour

dummy=np.array([[AA],[BB]])

print("after 1 hour, the value of A molecule : ",AA, "and B molecule: ",BB)

pop=np.concatenate((initial, dummy),axis=1)

p=np.array([150,150])

final_array=pop*p

#determining if the matrix is hermitian or not 

if final_array.all==(final_array.T).all:

     print("the final array is hermitan matrix")

else:

     print("mpt a hermitian matrix")



import numpy as np

import matplotlib.pylab as plot

import math as m

#initialize variables

#velocity, gravity

v = 30

g = 9.8

#increment theta 25 to 60 then find  t, x, y

#define x and y as arrays



theta = np.arange(m.pi/6, m.pi/3, m.pi/36)



t = np.linspace(0, 5, num=100) # Set time as 'continous' parameter.



for i in theta: # Calculate trajectory for every angle

    x1 = []

    y1 = []

    for k in t:

        x = ((v*k)*np.cos(i)) # get positions at every point in time

        y = ((v*k)*np.sin(i))-((0.5*g)*(k**2))

        x1.append(x)

        y1.append(y)

    p = [i for i, j in enumerate(y1) if j < 0] # Don't fall through the floor                          

    for i in sorted(p, reverse = True):

        del x1[i]

        del y1[i]



    plot.plot(x1, y1) # Plot for every angle



plot.show() # And show on one graphic
import matplotlib.pyplot as plt,random,numpy as np

def find_fix_point(xx,aa):

    x1=aa*(1-xx)

    xx=x1

    x2=aa*xx*(a-xx)

    r=x2-x1

    convergence=aa*(1-x1)/(1-r)

    return convergence



def making_graph(list_a,g):

    y=np.linspace(0,1,num=50)

    list_convergence=[ ]

    for i in list_a:

         x=y[random.randint (0,49)]

         list_convergence.append(round(find_fix_point(x,a),3))

    plt.scatter(list_a,list_convergence,label='bla bla bla', color='blue');

    plt.xlabel('converginf points')

    plt.ylabel('o')

    plt.title("graph: "+str(g))

    plt.show()



a=1

# part a, making a ranging in (1,5,4)

list_a=np.linspace(1.5, 4,num=1000)

making_graph(list_a,1)








