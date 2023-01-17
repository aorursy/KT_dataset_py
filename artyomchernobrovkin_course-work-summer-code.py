import numpy as np

import matplotlib.pyplot as plt

import scipy.integrate as spint

from mpl_toolkits.mplot3d import axes3d

import matplotlib as mpl
a=10

b1=5

b=10

G1=8.0*10**10#Сталь углеродистая

G2=4.0*10**10#Бронза марганцевая

A=100

eps=10**(-3)
pi=np.pi

alpha=pi/a

G_plus=G1+G2

G_minus=G1-G2

p_ksi= lambda xi: np.sin((np.pi*xi)/a)
def SymC_W1(pi,a,b,y,xi,x,G_plus):

    t=[(pi/a)*(b+y),(pi/a)*(b-y)]

    x=[(pi/a)*(xi-x),(pi/a)*(xi+x)]

    C=[0,0,0,0]

    C[0]=-np.log(np.cosh(t[1])-np.cos(x[0]))

    C[1]=-np.log(np.cosh(t[0])-np.cos(x[0]))

    C[2]=np.log(np.cosh(t[0])-np.cos(x[1]))

    C[3]=np.log(np.cosh(t[1])-np.cos(x[1]))

    sum_C_W1=0

    for w in range(len(C)):

        sum_C_W1+=(a/(4*pi*G_plus))*C[w]

    return sum_C_W1
def SymSeries_W1(A,pi,a,xi,x,b,b1,y,alpha,G_plus,G_minus):

    Sym_Series_W1=0

    x=[(pi/a)*(xi-x),(pi/a)*(xi+x)]

    exp=[-alpha*(b+y),-alpha*(b-y),-2*alpha*b,-2*alpha*(b-b1),-2*alpha*b1]

    for n in range(1,A):

        Q1=(np.exp(n*exp[0])+np.exp(n*exp[1]))/(alpha*n*((1-np.exp(n*exp[2]))* G_plus+G_minus*(np.exp(n*exp[3])-np.exp(n*exp[4]))))

        Q2=(np.exp(n*exp[0])+np.exp(n*exp[1]))/(alpha*n*G_plus)

        Sym_Series_W1+=(Q1-Q2)*(np.cos(n*x[0])-np.cos(n*x[1]))

    Sym_Series_W1=Sym_Series_W1/2    

    return Sym_Series_W1
def Int_W1(a,p_ksi,x,y,pi,b,A,alpha,b1,G_plus,G_minus):

    F=4/a

    #intSum=lambda xi :p_ksi(xi)*(  SymC_W1(pi,a,b,y,xi,x,G_plus))

    intSum=lambda xi :p_ksi(xi)*( SymC_W1(pi,a,b,y,xi,x,G_plus) + SymSeries_W1(A,pi,a,xi,x,b,b1,y,alpha,G_plus,G_minus))

    W1res=spint.fixed_quad(intSum,0,a,n=50)[0]

    return W1res*F
def SymC_W2(pi,a,b,b1,y,xi,x,G_plus,G_minus):

    t=[(pi/a)*(b+2*b1-y),(pi/a)*(b-2*b1+y),(pi/a)*(b+y),(pi/a)*(b-y)]

    x=[(pi/a)*(xi-x),(pi/a)*(xi+x)]

    C=[0,0,0,0,0,0,0,0]

    C[0]=-np.log(np.cosh(t[0])-np.cos(x[0]))

    C[1]=-np.log(np.cosh(t[1])-np.cos(x[0]))

    C[2]=np.log(np.cosh(t[2])-np.cos(x[0]))

    C[3]=np.log(np.cosh(t[3])-np.cos(x[0]))

    

    C[4]=-np.log(np.cosh(t[2])-np.cos(x[1]))

    C[5]=np.log(np.cosh(t[0])-np.cos(x[1]))

    C[6]=np.log(np.cosh(t[1])-np.cos(x[1]))

    C[7]=-np.log(np.cosh(t[3])-np.cos(x[1]))

    sum_C_W2=0

    sum_C_W2=((a**2)/(4*pi*G_plus))*( G_minus*(C[0]+C[1]+C[5]+C[6])+ G_plus*(C[2]+C[3]+C[4]+C[7]))

    return (-1)*sum_C_W2
def SymSeries_W2(A,pi,a,xi,x,b,b1,y,alpha,G_plus,G_minus):

    Sym_Series_W2=0

    x=[(pi/a)*(xi-x),(pi/a)*(xi+x)]

    exp=[-alpha*(b+2*b1-y),-alpha*(b-2*b1+y),-alpha*(b+y),-alpha*(b-y),-2*alpha*b,-2*alpha*(b-b1),-2*alpha*b1]

    for n in range(1,A):

        Q3=(G_minus*(np.exp(n*exp[0])+np.exp(n*exp[1]))-G_plus*(np.exp(n*exp[2])+np.exp(n*exp[3])))/(alpha*n*((-1+np.exp(n*exp[4]))* G_plus+G_minus*(-np.exp(n*exp[5])-np.exp(n*exp[6]))))

        Q4=(G_minus*(np.exp(n*exp[0])+np.exp(n*exp[1]))-G_plus*(np.exp(n*exp[2])+np.exp(n*exp[3])))/(alpha*n*G_plus)

        Sym_Series_W2+=(Q3-Q4)*(np.cos(n*x[0])-np.cos(n*x[1]))

    Sym_Series_W2=Sym_Series_W2/2    

    return Sym_Series_W2
def Int_W2(a,p_ksi,x,y,pi,b,A,alpha,b1,G_plus,G_minus,G2):

    F=2/(G2*a)

    #intSum_W2=lambda xi :p_ksi(xi)*( SymC_W2(pi,a,b,b1,y,xi,x,G_plus,G_minus))

    intSum_W2=lambda xi :p_ksi(xi)*( SymC_W2(pi,a,b,b1,y,xi,x,G_plus,G_minus) + SymSeries_W2(A,pi,a,xi,x,b,b1,y,alpha,G_plus,G_minus))

    W2res2=spint.fixed_quad(intSum_W2,0,a,n=50)[0]

    return W2res2*F
y_W1=np.linspace(0,b1,50)

x_W1=[0,a]
resultW1_0= np.zeros((len(y_W1),3)) 

resultW1_1= np.zeros((len(y_W1),3)) 

for i in range(len(y_W1)):

    resultW1_0[i]=Int_W1(a,p_ksi,x_W1[0],y_W1[i],pi,b,A,alpha,b1,G_plus,G_minus)

    resultW1_1[i]=Int_W1(a,p_ksi,x_W1[1],y_W1[i],pi,b,A,alpha,b1,G_plus,G_minus)
#Краевые условия W2|x=0,W2|x=a

y_W2=np.linspace(b1,b,50)

x_W2=[0,a]



resultW2_0= np.zeros((len(y_W2),3)) 

resultW2_1= np.zeros((len(y_W2),3)) 

for i in range(len(y_W2)):

    resultW2_0[i]=Int_W2(a,p_ksi,x_W2[0],y_W2[i],pi,b,A,alpha,b1,G_plus,G_minus,G2)

    resultW2_1[i]=Int_W2(a,p_ksi,x_W2[1],y_W2[i],pi,b,A,alpha,b1,G_plus,G_minus,G2)
#W1

fig,ax=plt.subplots(1,2)

ax[0].plot(y_W1,resultW1_0[:,0],label=r'$W_{1}$(x,y) |x=0', lw=3) 

ax[0].set_title(r'$W_{1}$(x,y)')

ax[1].plot(y_W1,resultW1_1[:,0],label=r'$W_{1}$(x,y) |x=a', lw=3)

ax[1].set_title(r'$W_{1}$(x,y)') 

ax[0].set_xlabel('x')   

ax[1].set_xlabel('x')   

ax[0].legend(loc=0, fontsize=10)

ax[1].legend(loc=0, fontsize=10)
#W2



fig,ax=plt.subplots(1,2)

ax[0].plot(y_W2,resultW2_0[:,0],label=r'$W_{2}$(x,y) |x=0', lw=3) 

ax[0].set_title(r'$W_{2}$(x,y)')

ax[1].plot(y_W2,resultW2_1[:,0],label=r'$W_{2}$(x,y) |x=a', lw=3)

ax[1].set_title(r'$W_{2}$(x,y)') 

ax[0].set_xlabel('x')   

ax[1].set_xlabel('x')   

ax[0].legend(loc=0, fontsize=10)

ax[1].legend(loc=0, fontsize=10)
#Условия сопряжения

x_pairing_W1=np.linspace(0,a,50)

y_pairing_W1=b1

resultW1_2= np.zeros((len(x_pairing_W1),3))     

for j in range(len(x_pairing_W1)):

    resultW1_2[j]=Int_W1(a,p_ksi,x_pairing_W1[j],y_pairing_W1,pi,b,A,alpha,b1,G_plus,G_minus)

    
#Условия сопряжения

x_pairing_W2=np.linspace(0,a,50)

y_pairing_W2=b1

resultW2_2= np.zeros((len(x_pairing_W2),3))     

for j in range(len(x_pairing_W2)):

    resultW2_2[j]=Int_W2(a,p_ksi,x_pairing_W2[j],y_pairing_W2,pi,b,A,alpha,b1,G_plus,G_minus,G2)
fig,ax=plt.subplots(1,2)

ax[0].plot(x_pairing_W1,resultW1_2[:,0],label=r'$W_{1}$(x,y) |y=b1', lw=3) 

ax[0].set_title(r'$W_{1}$(x,y)|y=b1-0')

ax[1].plot(x_pairing_W2,resultW2_2[:,0],label=r'$W_{2}$(x,y) |y=b1', lw=3)

ax[1].set_title(r'$W_{2}$(x,y) |y=b1+0') 

ax[0].set_xlabel('x')   

ax[1].set_xlabel('x')   

ax[0].legend(loc=0, fontsize=10)

ax[1].legend(loc=0, fontsize=10) 
#W1

x=np.linspace(0,a,50)

y=np.linspace(0,b1,50)

resultmatrix= np.zeros((len(x),len(y))) 

for i in range(len(x)):

    for j in range(len(y)):

        resultmatrix[i][j]=Int_W1(a,p_ksi,x[i],y[j],pi,b,A,alpha,b1,G_plus,G_minus)





xy=np.linspace(0,a)

X,Y=np.meshgrid(xy, xy)

Z=resultmatrix.T

fig=plt.figure(figsize=(14,14))

ax=fig.add_subplot(1,1,1,projection='3d')  

p=ax.plot_surface(X,Y,Z,rstride=3,cstride=3,cmap=mpl.cm.plasma,linewidth=0)

cb=fig.colorbar(p,shrink=0.8)

#ax.view_init(30,30)

ax.legend(loc=0)

ax.set_xlabel('x')

ax.set_ylabel('y')

ax.set_title(r'$W_{1}$(x,y)')
#W2

x=np.linspace(0,a,50)

y=np.linspace(b1,b,50)

resultmatrix= np.zeros((len(x),len(y))) 

for i in range(len(x)):

    for j in range(len(y)):

        resultmatrix[i][j]=Int_W2(a,p_ksi,x[i],y[j],pi,b,A,alpha,b1,G_plus,G_minus,G2)





xy=np.linspace(0,a)

X,Y=np.meshgrid(xy, xy)

Z=resultmatrix.T

fig=plt.figure(figsize=(14,14))

ax=fig.add_subplot(1,1,1,projection='3d')  

p=ax.plot_surface(X,Y,Z,rstride=3,cstride=3,cmap=mpl.cm.plasma,linewidth=0)

cb=fig.colorbar(p,shrink=0.8)

#ax.view_init(30,30)

ax.legend(loc=0)

ax.set_xlabel('x')

ax.set_ylabel('y')

ax.set_title(r'$W_{2}$(x,y)')