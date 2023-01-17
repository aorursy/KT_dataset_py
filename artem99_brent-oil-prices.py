

import numpy as np 

import pandas as pd 



data =pd.read_csv("../input/brent/Brent.csv")

y=np.asfarray(data.iloc[7262:])

y=y.reshape(953)

x=np.array(range(953))#DAYS
import matplotlib.pyplot as plt



plt.figure(figsize=(40,10))

plt.scatter(x,y,s=8)

plt.rc('xtick', labelsize=20)

plt.rc('ytick', labelsize=20)

plt.xticks([0,255,501,753,955],

           [2015,2016,2017,2018,2019])

plt.show()
x_train=x[:800]

x_test=x[800:]

y_train=y[:800]

y_test=y[800:]

import scipy as sp

def model(x,y,x_t,y_t):

    l=0

    for i in range(10):

        fp1,residuals,rank,sv,rcond=sp.polyfit(x,y,i,full=True)

        f=sp.poly1d(fp1)

        error=int(sp.sum((f(x)-y)**2))

        error1=int(sp.sum((f(x_t)-y_t)**2))

        print('polynom :',l,'error1:',error)

        print('polynom :',l,'error2:',error1)

        l+=1

#  DEGREE THREE, LEAST ERROR

      

model(x_train,y_train,x_test,y_test)


fp1,residuals,rank,sv,rcond=sp.polyfit(x,y,3,full=True)

f=sp.poly1d(fp1)

plt.figure(figsize=(40,10))

plt.rc('xtick', labelsize=20)

plt.rc('ytick', labelsize=20)

plt.plot(x,f(x),'r--',linewidth=2)

plt.scatter(x,y,s=10)

plt.xticks([0,255,501,753,955],

           [2015,2016,2017,2018,2019])

plt.show()
x_future=np.array(range(953,1010))



plt.figure(figsize=(40,10))

plt.rc('xtick', labelsize=20)

plt.rc('ytick', labelsize=20)

plt.plot(x_future,f(x_future),'r--',linewidth=6)

plt.scatter(x,y,s=10)

plt.xticks([0,255,501,753,955],

           [2015,2016,2017,2018,2019])

plt.show()


mean_x=np.mean(x)

mean_y=np.mean(y)

cov_x=np.cov(x)

cov_y=np.cov(y)

def covXY(x,y):

    sum_xy=[]

    for i in range(len(x)):

        a=(x[i]-mean_x)*(y[i]-mean_y)

        sum_xy.append(a)

    return sum(sum_xy)

cov_xy=covXY(x,y)/len(x)-1

cor_xy=cov_xy/(np.sqrt(cov_x)*np.sqrt(cov_y))

print('CORRELATION:',cor_xy)

RSS=[]#residual amount squares

TSS=[]#total amount squares

ESS=[]#sum of squares models

def fun(y,y_future,y_mean):

    for i in range(len(y)):

        r=(y[i]-y_future[i])**2

        RSS.append(r)

        t=(y[i]-y_mean)**2

        TSS.append(t)

        e=(y_future[i]-y_mean)**2

        ESS.append(e)

fun(y,f(x),mean_y)   



#FORCE, LINEAR COMMUNICATION

R2=sum(ESS)/sum(TSS)

print('R2:',R2)
average_price=sum(f(x_future))/len(x_future)

print('$',average_price)