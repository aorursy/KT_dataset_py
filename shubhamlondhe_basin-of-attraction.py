import numpy as np

import matplotlib.pyplot as plt



#plt.xlim(-2.5,2.5)

#plt.ylim(-2.5,2.5)

plt.xticks(np.arange(-2.5,2.55 , 0.05))

plt.yticks(np.arange(-2.5,2.55, 0.05))

#plt.scatter(x, y)

plt.xticks(fontsize=1,rotation=90)

plt.yticks(fontsize=1)

plt.grid()

plt.show()
print(len(np.arange(-2.5,2.55 , 0.05)))
x=np.arange(-2.5,2.55 , 0.05)

y=np.arange(-2.5,2.55 , 0.05)



def HenonMap(a,b,x,y):



    return y + 1.0 - a *x*x, b * x



a = 1.4



b = 0.3



iterates = 1000



x_arr = []

y_arr = []



xconv=[]

yconv=[]

xnonconv=[]

ynonconv=[]



for i in range(100):

    for j in range(100):

        x_arr.append( x[i] )

        y_arr.append( y[i] )

        for n in range(0,iterates):

            xtemp, ytemp = HenonMap(a,b,x[i],y[i])

            x_arr.append( xtemp )

            y_arr.append( ytemp )

        for k in range(100):

            if(-2.5<=x_arr[k]<=2.5 and -2.5<=y_arr[k]<=2.5):

                    xconv.append(x[i])

                    yconv.append(y[j])

            elif(pow(10,5)<=x_arr[k] or -pow(10,5)<=x_arr[k]  or pow(10,5)<=y_arr[k] or -pow(10,5)<=y_arr[k]):

                    xnonconv.append(x[i])

                    ynonconv.append(y[j])

print(len(xconv),len(yconv),len(xnonconv),len(ynonconv))
print(xnonconv)
#from matplotlib import pyplot as plt

#from matplotlib.patches import Rectangle

#w,h=0.05,0.05

#plt.xlim(-2.5,2.5)

#plt.ylim(-2.5,2.5)

#plt.gca()

#for i in range(len(xconv)):

    #x=round(xconv[i]+0.05,1)

    #y=round(yconv[i]+0.05,1)

    #rectangle = plt.Rectangle((x, y), 0.1, 0.1, fc='b',fill=True)

    #plt.gca().add_patch(rectangle)

#for i in range(len(xnonconv)):

    #x=round(xnonconv[i]+0.05,1)

    #y=round(ynonconv[i]+0.05,1)

    #rectangle = plt.Rectangle((x, y), 0.1, 0.1, fc='r',fill=True)

    #plt.gca().add_patch(rectangle)
