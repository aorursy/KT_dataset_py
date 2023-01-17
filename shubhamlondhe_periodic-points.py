import numpy as np

import matplotlib.pyplot as plt
def HenonMap(a,b,x,y):

    return y + 1.0 - a *x*x, b * x







def GetPerPoint(a,b,iterates,xtemp,ytemp):

    xpoints=[xtemp]

    ypoints=[ytemp]

    for i in range(0,iterates-1):

        x, y = HenonMap(a,b,xpoints[i],ypoints[i])

        xpoints.append(x)

        ypoints.append(y)

    return xpoints, ypoints

#DataSet 1A

X2PerPoint,Y2PerPoint=GetPerPoint(1.38813,0.0497109,2,1.27433,-0.224616)

X4PerPoint,Y4PerPoint=GetPerPoint(1.38813,0.0497109,4,0.0194026,1.17796)
print(X4PerPoint,Y4PerPoint)
plt.xlim(-2,2)

plt.ylim(-2,2)

plt.plot(-1.81466,-1.81466,marker='o',color='k',alpha=1)

plt.plot(0.764951,0.764951,marker='o',label="Fixed Point",color='k',alpha=1)

plt.plot(X2PerPoint[0],Y2PerPoint[0],marker='o',color='r',alpha=1)

plt.plot(X2PerPoint[1],Y2PerPoint[1],marker='o',label="Period 2 Point",color='r',alpha=1)        

plt.plot(X4PerPoint[0],Y4PerPoint[0],marker='o',color='g',alpha=1)

plt.plot(X4PerPoint[1],Y4PerPoint[1],marker='o',color='g',alpha=1)

plt.plot(X4PerPoint[2],Y4PerPoint[2],marker='o',color='g',alpha=1)

plt.plot(X4PerPoint[3],Y4PerPoint[3],marker='o',label="Period 4 Point",color='g',alpha=1)

plt.title("Period 2 and Period 4 points")       

plt.legend()

plt.show()
#DataSet 1B

X2PerPoint1,Y2PerPoint1=GetPerPoint(1.90954,0.318205,2,1.43775,-0.119548)

X4PerPoint1,Y4PerPoint1=GetPerPoint(1.90954,0.318205,4,0.0951981,1.37722)
plt.xlim(-2.5,2.5)

plt.ylim(-2.5,2.5)

plt.plot(-2.1901,-2.1901,marker='o',color='k',alpha=1)

plt.plot(0.871898,0.871898,marker='o',label="Fixed Point",color='k',alpha=1)

plt.plot(X2PerPoint1[0],Y2PerPoint1[0],marker='o',color='r',alpha=1)

plt.plot(X2PerPoint1[1],Y2PerPoint1[1],marker='o',label="Period 2 Point",color='r',alpha=1)        

plt.plot(X4PerPoint1[0],Y4PerPoint1[0],marker='o',color='g',alpha=1)

plt.plot(X4PerPoint1[1],Y4PerPoint1[1],marker='o',color='g',alpha=1)

plt.plot(X4PerPoint1[2],Y4PerPoint1[2],marker='o',color='g',alpha=1)

plt.plot(X4PerPoint1[3],Y4PerPoint1[3],marker='o',label="Period 4 Point",color='g',alpha=1)

plt.title("Period 2 and Period 4 points")       

plt.legend()

plt.show()
#DataSet 2A

X2PerPoint2,Y2PerPoint2=GetPerPoint(1.43528,0.0350756,2,-0.277288,1.31236)

X4PerPoint2,Y4PerPoint2=GetPerPoint(1.43528,0.0350756,4,0.11221,1.15753)

X8PerPoint2,Y8PerPoint2=GetPerPoint(1.43528,0.0350756,8,0.013414,1.19879)
plt.xlim(-2.5,2.5)

plt.ylim(-2.5,2.5)

plt.plot(0.787502,0.787502,marker='o',color='k',alpha=1)

plt.plot(-1.82258,-1.82258,marker='o',label="Fixed Point",color='k',alpha=1)

plt.plot(X2PerPoint2[0],Y2PerPoint2[0],marker='o',color='r',alpha=1)

plt.plot(X2PerPoint2[1],Y2PerPoint2[1],marker='o',label="Period 2 Point",color='r',alpha=1)        

plt.plot(X4PerPoint2[0],Y4PerPoint2[0],marker='o',color='g',alpha=1)

plt.plot(X4PerPoint2[1],Y4PerPoint2[1],marker='o',color='g',alpha=1)

plt.plot(X4PerPoint2[2],Y4PerPoint2[2],marker='o',color='g',alpha=1)

plt.plot(X4PerPoint2[3],Y4PerPoint2[3],marker='o',label="Period 4 Point",color='g',alpha=1)

plt.plot(X4PerPoint2[0],Y4PerPoint2[0],marker='o',color='g',alpha=1)

plt.plot(X4PerPoint2[1],Y4PerPoint2[1],marker='o',color='g',alpha=1)

plt.plot(X4PerPoint2[2],Y4PerPoint2[2],marker='o',color='g',alpha=1)

plt.plot(X4PerPoint2[3],Y4PerPoint2[3],marker='o',label="Period 8 Point",color='b',alpha=1)



plt.title("Period 2,Period 4 and period 8 points")       

plt.legend()

plt.show()