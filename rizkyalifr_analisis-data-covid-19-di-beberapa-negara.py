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

import pandas as pd

import numpy as np

from datetime import datetime,timedelta

from sklearn.metrics import mean_squared_error

from scipy.optimize import curve_fit

from scipy.optimize import fsolve

import matplotlib.pyplot as plt

%matplotlib inline
def linear(x,y) :

    x=np.array(x)

    y=np.array(y)

    n = len(y) #jumlah data

    xy=x*y #xy

    xx=x**2 #xx

    b=(n*xy.sum()-x.sum()*y.sum())/(n*xx.sum()-(x.sum())**2)

    a=y.mean()-b*x.mean()

    print("Persamaan regresi linearnya adalah : y = {:.4f}x + {:.4f}".format(b,a))

    yreg=b*x+a #yregresi

    return yreg
def pangkat(x,y) :

    x=np.array(x)

    y=np.array(y)

    xp=np.log10(x)

    yp=np.log10(y)

    n = len(y) #jumlah data

    xy=xp*yp #xy

    xx=xp**2 #xx

    bp=(n*xy.sum()-xp.sum()*yp.sum())/(n*xx.sum()-(xp.sum())**2)

    aa=yp.mean()-bp*xp.mean()

    ap=10**aa

    print("Persamaan regresi pangkatnya adalah : y = {}x^{}".format(ap,bp))

    yreg=ap*(x**bp)

    return yreg
def eksp(x,y):

    x=np.array(x)

    y=np.array(y)

    x=x

    ye=np.log(y)

    n = len(y) #jumlah data

    xye=x*ye #xy

    xx=x**2 #xx

    b=(n*xye.sum()-x.sum()*ye.sum())/(n*xx.sum()-(x.sum())**2)

    import math

    #Menghitung a

    A=ye.mean()-b*x.mean()

    a=math.exp(A)

    print("Persamaan regresi eksponensialnya adalah : y = {:.4f}e^{:.4f}x".format(a,b))

    yreg=np.array(a*np.exp(b*x)) #yregresi

    return yreg
def orde3(x,y):

    x=np.array(x)

    y=np.array(y)

    #Matriks A

    A=np.zeros((4,4),dtype=float)

    for i in range(0,4):

        for j in range(0,4):

            A[i][j]=(x**(j+i)).sum()

    

    #Matriks B

    B=np.zeros((1,4), dtype=float)

    for i in range(0,4) :

        B[0][i]=((x**i)*y).sum()

    

    #Eliminasi Gauss

    A=np.array(A)

    b=np.array([ B[0][0],  B[0][1],  B[0][2],  B[0][3]])



    Ab = np.hstack([A, b.reshape(-1, 1)])



    n = len(b)



    for i in range(n):

        a = Ab[i]

        for j in range(i + 1, n):

            b = Ab[j]

            m = a[i] / b[i]

            Ab[j] = a - m * b

            

    for i in range(n - 1, -1, -1):

        Ab[i] = Ab[i] / Ab[i, i]

        a = Ab[i]

        for j in range(i - 1, -1, -1):

            b = Ab[j]

            m = a[i] / b[i]

            Ab[j] = a - m * b

    X = Ab[:, 4]

    a0=X[0]

    a1=X[1]

    a2=X[2]

    a3=X[3]

    print("Persamaan regresi polinomial orde 3 nya adalah : y = {:.4f} + {:.4f}x + {:.4f}x^2 + {:.4f}x^3".format(a0,a1,a2,a3))

    yreg=a0+a1*x+a2*(x**2)+a3*(x**3)

    return yreg
def r2_score_linear(y,ypred) :

    s1=(np.subtract(y,ypred)**2).sum()

    s2=(np.subtract(y,np.mean(y))**2).sum()

    r=1-s1/s2

    print("R^2 adalah {}".format(r))



def r2_score_pangkat(x,y) :

    x=np.array(x)

    y=np.array(y)

    xp=np.log10(x)

    yp=np.log10(y)

    n = len(y) #jumlah data

    xy=xp*yp #xy

    xx=xp**2 #xx

    bp=(n*xy.sum()-xp.sum()*yp.sum())/(n*xx.sum()-(xp.sum())**2)

    aa=yp.mean()-bp*xp.mean()

    ypp=aa+bp*xp

    s1=(np.subtract(yp,ypp)**2).sum()

    s2=(np.subtract(yp,np.mean(yp))**2).sum()

    r=1-s1/s2

    print("R^2 adalah {}".format(r))



def r2_score_exp(x,y) :

    x=np.array(x)

    y=np.array(y)

    ye=np.log(y)

    n = len(y) #jumlah data

    xye=x*ye #xy

    xx=x**2 #xx

    b=(n*xye.sum()-x.sum()*ye.sum())/(n*xx.sum()-(x.sum())**2)

    #Menghitung a

    A=ye.mean()-b*x.mean()

    ypp=A+b*x

    s1=(np.subtract(ye,ypp)**2).sum()

    s2=(np.subtract(ye,np.mean(ye))**2).sum()

    r=1-s1/s2

    print("R^2 adalah {}".format(r))

    

def r2_score_orde3(y,yreg) :

    s1=(np.subtract(y,yreg)**2).sum()

    s2=(np.subtract(y,np.mean(y))**2).sum()

    r=1-s1/s2

    print("R^2 adalah {}".format(r))
def facto(a):

    for i in df:

        df[a] = df[a].factorize()[0]+1
def active(x,y,yy,country):

    fig= plt.figure(figsize=(5,2))

    axes= fig.add_axes([2,2,4,4])

    plt.xlabel("\nDate",fontsize=28)

    plt.ylabel("Total Active Cases in" +country,fontsize=28)

    plt.xticks(rotation=90,fontsize=15)

    plt.yticks(fontsize=15)

    axes.set_title('Active Cases in '+country+'\n', fontdict={'fontsize': 35, 'fontweight': 'medium'})

    axes.plot(x,yy)

    plt.scatter(x,y)

    plt.show()
def death(x,y,yy,country):

    fig= plt.figure(figsize=(5,2))

    axes= fig.add_axes([2,2,4,4])

    plt.xlabel("\nDate",fontsize=28)

    plt.ylabel("Total Deaths in "+country,fontsize=28)

    plt.xticks(rotation=90,fontsize=15)

    plt.yticks(fontsize=15)

    axes.set_title('Total Deaths in '+country+'\n', fontdict={'fontsize': 35, 'fontweight': 'medium'})

    axes.plot(x,yy)

    plt.scatter(x,y)

    plt.show()
#Input Data

df=pd.read_csv('../input/china.csv')
x = list(df.iloc[:,0])

y = list(df.iloc[:,1])

date=x



fig= plt.figure(figsize=(5,2))

axes= fig.add_axes([2,2,4,4])

plt.xlabel("\nDate",fontsize=28)

plt.ylabel("Total Active Cases in China ",fontsize=28)

plt.xticks(rotation=90,fontsize=15)

plt.yticks(fontsize=15)

axes.set_title('Active Cases in China\n', fontdict={'fontsize': 35, 'fontweight': 'medium'})

axes.plot(x,y)

plt.scatter(x,y)

plt.show()
x = list(df.iloc[:26,0])

x1=x
facto("date")
x = list(df.iloc[:26,0])

y = list(df.iloc[:26,1])

ypred=linear(x,y)

r2_score_linear(y,ypred)
y = list(df.iloc[:26,1])



fig= plt.figure(figsize=(5,2))

axes= fig.add_axes([2,2,4,4])

plt.xlabel("\nDate",fontsize=28)

plt.ylabel("Total Active Cases in China ",fontsize=28)

plt.xticks(rotation=90,fontsize=15)

plt.yticks(fontsize=15)

axes.set_title('Active Cases in China(+)\n', fontdict={'fontsize': 35, 'fontweight': 'medium'})

axes.plot(x1,ypred)

plt.scatter(x1,y)

plt.show()
ypred=pangkat(x,y)

r2_score_pangkat(x,y)
y = list(df.iloc[:26,1])



fig= plt.figure(figsize=(5,2))

axes= fig.add_axes([2,2,4,4])

plt.xlabel("\nDate",fontsize=28)

plt.ylabel("Total Active Cases in China ",fontsize=28)

plt.xticks(rotation=90,fontsize=15)

plt.yticks(fontsize=15)

axes.set_title('Active Cases in China(+)\n', fontdict={'fontsize': 35, 'fontweight': 'medium'})

axes.plot(x1,ypred)

plt.scatter(x1,y)

plt.show()
ypred=eksp(x,y)

r2_score_exp(x,y)
y = list(df.iloc[:26,1])



fig= plt.figure(figsize=(5,2))

axes= fig.add_axes([2,2,4,4])

plt.xlabel("\nDate",fontsize=28)

plt.ylabel("Total Active Cases in China ",fontsize=28)

plt.xticks(rotation=90,fontsize=15)

plt.yticks(fontsize=15)

axes.set_title('Active Cases in China(+)\n', fontdict={'fontsize': 35, 'fontweight': 'medium'})

axes.plot(x1,ypred)

plt.scatter(x1,y)

plt.show()
ypred=orde3(x,y)

r2_score_orde3(y,ypred)
y = list(df.iloc[:26,1])



fig= plt.figure(figsize=(5,2))

axes= fig.add_axes([2,2,4,4])

plt.xlabel("\nDate",fontsize=28)

plt.ylabel("Total Active Cases in China ",fontsize=28)

plt.xticks(rotation=90,fontsize=15)

plt.yticks(fontsize=15)

axes.set_title('Active Cases in China(+)\n', fontdict={'fontsize': 35, 'fontweight': 'medium'})

axes.plot(x1,ypred)

plt.scatter(x1,y)

plt.show()
##Membuat fungsi yang menggambarkan penambahan kasus aktif di china

def china_active (x) :

    y = 848.1973  -399.5480*x + 174.5892*(x**2)  -2.7120*(x**3)

    return y
df["tanggal"]=date

x= list(df.iloc[26:65,3])

x1=x

y = list(df.iloc[26:65,1])
x = list(df.iloc[26:65,0])

y = list(df.iloc[26:65,1])

ypred=linear(x,y)

r2_score_linear(y,ypred)
y = list(df.iloc[26:65,1])



fig= plt.figure(figsize=(5,2))

axes= fig.add_axes([2,2,4,4])

plt.xlabel("\nDate",fontsize=28)

plt.ylabel("Total Active Cases in China (-) ",fontsize=28)

plt.xticks(rotation=90,fontsize=15)

plt.yticks(fontsize=15)

axes.set_title('Active Cases in China(-)\n', fontdict={'fontsize': 35, 'fontweight': 'medium'})

axes.plot(x1,ypred)

plt.scatter(x1,y)

plt.show()
ypred=pangkat(x,y)

r2_score_pangkat(x,y)
y = list(df.iloc[26:65,1])



fig= plt.figure(figsize=(5,2))

axes= fig.add_axes([2,2,4,4])

plt.xlabel("\nDate",fontsize=28)

plt.ylabel("Total Active Cases in China (-) ",fontsize=28)

plt.xticks(rotation=90,fontsize=15)

plt.yticks(fontsize=15)

axes.set_title('Active Cases in China(-)\n', fontdict={'fontsize': 35, 'fontweight': 'medium'})

axes.plot(x1,ypred)

plt.scatter(x1,y)

plt.show()
ypred=eksp(x,y)

r2_score_exp(x,y)
y = list(df.iloc[26:65,1])



fig= plt.figure(figsize=(5,2))

axes= fig.add_axes([2,2,4,4])

plt.xlabel("\nDate",fontsize=28)

plt.ylabel("Total Active Cases in China (-) ",fontsize=28)

plt.xticks(rotation=90,fontsize=15)

plt.yticks(fontsize=15)

axes.set_title('Active Cases in China(-)\n', fontdict={'fontsize': 35, 'fontweight': 'medium'})

axes.plot(x1,ypred)

plt.scatter(x1,y)

plt.show()
ypred=orde3(x,y)

r2_score_orde3(y,ypred)
y = list(df.iloc[26:65,1])



fig= plt.figure(figsize=(5,2))

axes= fig.add_axes([2,2,4,4])

plt.xlabel("\nDate",fontsize=28)

plt.ylabel("Total Active Cases in China (-) ",fontsize=28)

plt.xticks(rotation=90,fontsize=15)

plt.yticks(fontsize=15)

axes.set_title('Active Cases in China(-)\n', fontdict={'fontsize': 35, 'fontweight': 'medium'})

axes.plot(x1,ypred)

plt.scatter(x1,y)

plt.show()
x = list(df.iloc[:,0])

y = list(df.iloc[:,2])



fig= plt.figure(figsize=(5,2))

axes= fig.add_axes([2,2,4,4])

plt.xlabel("\nDate",fontsize=28)

plt.ylabel("Total Number Deaths in china ",fontsize=28)

plt.xticks(rotation=90,fontsize=15)

plt.yticks(fontsize=15)

axes.set_title('Total Deaths in China\n', fontdict={'fontsize': 35, 'fontweight': 'medium'})

axes.plot(date,y)

plt.scatter(date,y)

plt.show()
x = list(df.iloc[:,0])

y = list(df.iloc[:,2])

ypred=linear(x,y)

r2_score_linear(y,ypred)
x = list(df.iloc[:,0])

y = list(df.iloc[:,2])



fig= plt.figure(figsize=(5,2))

axes= fig.add_axes([2,2,4,4])

plt.xlabel("\nDate",fontsize=28)

plt.ylabel("Total Number Deaths in China ",fontsize=28)

plt.xticks(rotation=90,fontsize=15)

plt.yticks(fontsize=15)

axes.set_title('Total Deaths in China\n', fontdict={'fontsize': 35, 'fontweight': 'medium'})

axes.plot(date,ypred)

plt.scatter(date,y)

plt.show()
ypred=pangkat(x,y)

r2_score_pangkat(x,y)
x = list(df.iloc[:,0])

y = list(df.iloc[:,2])



fig= plt.figure(figsize=(5,2))

axes= fig.add_axes([2,2,4,4])

plt.xlabel("\nDate",fontsize=28)

plt.ylabel("Total Number Deaths in China ",fontsize=28)

plt.xticks(rotation=90,fontsize=15)

plt.yticks(fontsize=15)

axes.set_title('Total Deaths in China\n', fontdict={'fontsize': 35, 'fontweight': 'medium'})

axes.plot(date,ypred)

plt.scatter(date,y)

plt.show()
ypred=eksp(x,y)

r2_score_exp(x,y)
x = list(df.iloc[:,0])

y = list(df.iloc[:,2])



fig= plt.figure(figsize=(5,2))

axes= fig.add_axes([2,2,4,4])

plt.xlabel("\nDate",fontsize=28)

plt.ylabel("Total Number Deaths in china ",fontsize=28)

plt.xticks(rotation=90,fontsize=15)

plt.yticks(fontsize=15)

axes.set_title('Total Deaths in China\n', fontdict={'fontsize': 35, 'fontweight': 'medium'})

axes.plot(date,ypred)

plt.scatter(date,y)

plt.show()
ypred=orde3(x,y)

r2_score_orde3(y,ypred)
x = list(df.iloc[:,0])

y = list(df.iloc[:,2])



fig= plt.figure(figsize=(5,2))

axes= fig.add_axes([2,2,4,4])

plt.xlabel("\nDate",fontsize=28)

plt.ylabel("Total Number Deaths in China ",fontsize=28)

plt.xticks(rotation=90,fontsize=15)

plt.yticks(fontsize=15)

axes.set_title('Total Deaths in China\n', fontdict={'fontsize': 35, 'fontweight': 'medium'})

axes.plot(date,ypred)

plt.scatter(date,y)

plt.show()
def china_death(x) :

    y = -201.5653 + 33.4657*x + 2.3820*(x**2)  -0.0329*(x**3)

    return y
df=pd.read_csv('../input/south_korea.csv')
x = list(df.iloc[:,0])

y = list(df.iloc[:,1])

date=x

tgl=date



fig= plt.figure(figsize=(5,2))

axes= fig.add_axes([2,2,4,4])

plt.xlabel("\nDate",fontsize=28)

plt.ylabel("Total Active Cases in South Korea ",fontsize=28)

plt.xticks(rotation=90,fontsize=15)

plt.yticks(fontsize=15)

axes.set_title('Active Cases in South Korea\n', fontdict={'fontsize': 35, 'fontweight': 'medium'})

axes.plot(x,y)

plt.scatter(x,y)

plt.show()
x = list(df.iloc[:26,0])

y = list(df.iloc[:26,1])

date=x

facto('date')

x = list(df.iloc[:26,0])
ypred=linear(x,y)

r2_score_linear(y,ypred)

active(date,y,ypred,"South Korea")
ypred=pangkat(x,y)

r2_score_pangkat(x,y)

active(date,y,ypred,"South Korea")
ypred=eksp(x,y)

r2_score_exp(x,y)

active(date,y,ypred,"South Korea")
ypred=orde3(x,y)

r2_score_orde3(y,ypred)

active(date,y,ypred,"South Korea")
def korea_active(x) :

    y = 999.4408  -565.8759*x + 69.6722*(x**2)  -1.4701*(x**3)

    return y
x=list(df.iloc[26:,0])

y=list(df.iloc[26:,1])

date=tgl[26:]

facto('date')

x=list(df.iloc[26:,0])
active(date,y,y,"South Korea")
ypred=linear(x,y)

r2_score_linear(y,ypred)

active(date,y,ypred,"South Korea")
ypred=pangkat(x,y)

r2_score_pangkat(x,y)

active(date,y,ypred,"South Korea")
ypred=eksp(x,y)

r2_score_exp(x,y)

active(date,y,ypred,"South Korea")
ypred=orde3(x,y)

r2_score_orde3(y,ypred)

active(date,y,ypred,"South Korea")
y=list(df.iloc[10:,2])

x=list(df.iloc[10:,0])

tgl=tgl[10:]

death(tgl,y,y,"South Korea")
ypred=linear(x,y)

r2_score_linear(y,ypred)

death(tgl,y,ypred,"South Korea")
ypred=pangkat(x,y)

r2_score_pangkat(x,y)

death(tgl,y,ypred,"South Korea")
ypred=eksp(x,y)

r2_score_exp(x,y)

death(tgl,y,ypred,"South Korea")
ypred=orde3(x,y)

r2_score_orde3(y,ypred)

death(tgl,y,ypred,"South Korea")
def korea_death(x) :

    y = -56.4323 + 7.3958*x  -0.1889*(x**2) + 0.0029*(x**3)

    return y
df=pd.read_csv('../input/italy.csv')
x=list(df.iloc[:,0])

y=list(df.iloc[:,1])

date=x

facto('date')

x=list(df.iloc[:,0])
active(date,y,y,"Italy")
ypred=linear(x,y)

r2_score_linear(y,ypred)

active(date,y,ypred,"Itlay")
ypred=pangkat(x,y)

r2_score_pangkat(x,y)

active(date,y,ypred,"Itlay")
ypred=eksp(x,y)

r2_score_exp(x,y)

active(date,y,ypred,"Itlay")
ypred=orde3(x,y)

r2_score_orde3(y,ypred)

active(date,y,ypred,"Itlay")
def italy_active(x) :

    y = 242.8358 + 54.07488*x -23.9166*(x**2) + 1.4830*(x**3)

    return y
x=list(df.iloc[6:,0])

y=list(df.iloc[6:,2])

x=np.array(range(1,len(x)+1))

date=date[6:]

death(date,y,y,"Italy")
ypred=linear(x,y)

r2_score_linear(y,ypred)

death(date,y,ypred,"Italy")
ypred=pangkat(x,y)

r2_score_pangkat(x,y)

death(date,y,ypred,"Italy")
ypred=eksp(x,y)

r2_score_exp(x,y)

death(date,y,ypred,"Italy")
ypred=orde3(x,y)

r2_score_orde3(y,ypred)

death(date,y,ypred,"Italy")
def italy_death(x) :

    y = -67.0005 + 50.9006*x  -8.1316*(x**2) + 0.3887*(x**3)

    return y
df=pd.read_csv('../input/iran.csv')
tgl=list(df.iloc[:,0])

x=list(df.iloc[3:,0])

y=list(df.iloc[3:,1])

date=x

facto('date')

x=list(df.iloc[3:,0])

x=np.array(range(1,len(x)+1))
active(date,y,y,"Iran")
ypred=linear(x,y)

r2_score_linear(y,ypred)

active(date,y,ypred,"Iran")
ypred=pangkat(x,y)

r2_score_pangkat(x,y)

active(date,y,ypred,"Iran")
ypred=eksp(x,y)

r2_score_exp(x,y)

active(date,y,ypred,"Iran")
ypred=orde3(x,y)

r2_score_orde3(y,ypred)

active(date,y,ypred,"Iran")
def iran_active(x) :

    y = 2.7289*np.exp(0.2578*x)

    return y
x=list(df.iloc[1:,0])

x=np.array(range(1,len(x)+1))

y=list(df.iloc[1:,2])

date=tgl[1:]

death(date,y,y,"Iran")
ypred=linear(x,y)

r2_score_linear(y,ypred)

death(date,y,ypred,"Iran")
ypred=pangkat(x,y)

r2_score_pangkat(x,y)

death(date,y,ypred,"Iran")
ypred=eksp(x,y)

r2_score_exp(x,y)

death(date,y,ypred,"Iran")
ypred=orde3(x,y)

r2_score_orde3(y,ypred)

death(date,y,ypred,"Iran")
def iran_death(x) :

    y = 61.3416  -16.0411*x + 0.5321*(x**2) + 0.0422*(x**3)

    return y
df=pd.read_csv('../input/spain.csv')
tgl=list(df.iloc[:,0])

x=list(df.iloc[1:,0])

y=list(df.iloc[1:,1])

date=x

facto('date')

x=list(df.iloc[1:,0])

x=np.array(range(1,len(x)+1))
active(date,y,y,"Spain")
ypred=linear(x,y)

r2_score_linear(y,ypred)

active(date,y,ypred,"Spain")
ypred=pangkat(x,y)

r2_score_pangkat(x,y)

active(date,y,ypred,"Spain")
ypred=eksp(x,y)

r2_score_exp(x,y)

active(date,y,ypred,"Spain")
ypred=orde3(x,y)

r2_score_orde3(y,ypred)

active(date,y,ypred,"Spain")
def spain_active(x) :

    y = -1123.2841 + 630.7073*x  -84.2410*(x**2) + 3.4434*(x**3)

    return y
x=list(df.iloc[9:,0])

x=np.array(range(1,len(x)+1))

y=list(df.iloc[9:,2])

date=tgl[9:]

death(date,y,y,"Spain")
ypred=linear(x,y)

r2_score_linear(y,ypred)

death(date,y,ypred,"Spain")
ypred=pangkat(x,y)

r2_score_pangkat(x,y)

death(date,y,ypred,"Spain")
ypred=eksp(x,y)

r2_score_exp(x,y)

death(date,y,ypred,"Spain")
ypred=orde3(x,y)

r2_score_orde3(y,ypred)

death(date,y,ypred,"Spain")
def spain_death(x) :

    y = -213.5343 + 134.9023*x  -19.9782*(x**2) + 0.9258*x**3

    return y
df=pd.read_csv('../input/germany.csv')
tgl=list(df.iloc[:,0])

x=list(df.iloc[:,0])

y=list(df.iloc[:,1])

date=x

facto('date')

x=list(df.iloc[:,0])
active(date,y,y,"Germany")
ypred=linear(x,y)

r2_score_linear(y,ypred)

active(date,y,ypred,"Germany")
ypred=pangkat(x,y)

r2_score_pangkat(x,y)

active(date,y,ypred,"Germany")
ypred=eksp(x,y)

r2_score_exp(x,y)

active(date,y,ypred,"Germany")
ypred=orde3(x,y)

r2_score_orde3(y,ypred)

active(date,y,ypred,"Germany")
def germany_active(x) :

    y = -1722.0683 + 752.8608*x -72.2599*(x**2) + 1.9000*(x**3)

    return y
x=list(df.iloc[23:,0])

y=list(df.iloc[23:,2])

x=np.array(range(1,len(x)+1))

date=tgl[23:]

death(date,y,y,"Germany")
ypred=linear(x,y)

r2_score_linear(y,ypred)

death(date,y,ypred,"Germany")
ypred=pangkat(x,y)

r2_score_pangkat(x,y)

death(date,y,ypred,"Germany")
ypred=eksp(x,y)

r2_score_exp(x,y)

death(date,y,ypred,"Germnany")
ypred=orde3(x,y)

r2_score_orde3(y,ypred)

death(date,y,ypred,"Germany")
def germany_death(x) :

    y = -8.7712 + 8.1603*x -1.4059*(x**2) + 0.0987*(x**3)

    return y
df=pd.read_csv('../input/australia.csv')
tgl=list(df.iloc[:,0])

x=list(df.iloc[:,0])

y=list(df.iloc[:,1])

date=x

facto('date')

x=list(df.iloc[:,0])
active(date,y,y,"Australia")
ypred=linear(x,y)

r2_score_linear(y,ypred)

active(date,y,ypred,"Australia")
ypred=pangkat(x,y)

r2_score_pangkat(x,y)

active(date,y,ypred,"Australia")
ypred=eksp(x,y)

r2_score_exp(x,y)

active(date,y,ypred,"Australia")
ypred=orde3(x,y)

r2_score_orde3(y,ypred)

active(date,y,ypred,"Australia")
def aus_active(x) :

    y = -268.8942 + 102.5710*x  -8.3502*(x**2) + 0.1860*(x**3)

    return y
x=list(df.iloc[15:,0])

x=np.array(range(1,len(x)+1))

y=list(df.iloc[15:,2])

date=tgl[15:]

death(date,y,y,"Australia")
ypred=linear(x,y)

r2_score_linear(y,ypred)

death(date,y,ypred,"Australia")
ypred=pangkat(x,y)

r2_score_pangkat(x,y)

death(date,y,ypred,"Australia")
ypred=eksp(x,y)

r2_score_exp(x,y)

death(date,y,ypred,"Australia")
ypred=orde3(x,y)

r2_score_orde3(y,ypred)

death(date,y,ypred,"Australia")
def aus_death(x) :

    y = 0.1886 + 0.5217*x + -0.0360*(x**2) + 0.0013*(x**3)

    return y
df=pd.read_csv('../input/usa.csv')
tgl=list(df.iloc[:,0])

x=list(df.iloc[:,0])

y=list(df.iloc[:,1])

date=x

facto('date')

x=list(df.iloc[:,0])
active(date,y,y,"USA")
ypred=linear(x,y)

r2_score_linear(y,ypred)

active(date,y,ypred,"USA")
ypred=pangkat(x,y)

r2_score_pangkat(x,y)

active(date,y,ypred,"USA")
ypred=eksp(x,y)

r2_score_exp(x,y)

active(date,y,ypred,"USA")
ypred=orde3(x,y)

r2_score_orde3(y,ypred)

active(date,y,ypred,"USA")
def usa_active(x) :

    y=3.2136*np.exp(0.2367*x)

    return y
x=list(df.iloc[14:,0])

x=np.array(range(1,len(x)+1))

y=list(df.iloc[14:,2])

date=tgl[14:]

death(date,y,y,"USA")
ypred=linear(x,y)

r2_score_linear(y,ypred)

death(date,y,ypred,"USA")
ypred=pangkat(x,y)

r2_score_pangkat(x,y)

death(date,y,ypred,"USA")
ypred=eksp(x,y)

r2_score_exp(x,y)

death(date,y,ypred,"USA")
ypred=orde3(x,y)

r2_score_orde3(y,ypred)

death(date,y,ypred,"USA")
def usa_death(x) :

    y = -126.8023 + 69.1938*x -8.0705*(x**2) + 0.2714*(x**3)

    return y
df=pd.read_csv('../input/indonesia.csv')
tgl=list(df.iloc[1:,0])

x=list(df.iloc[1:,0])

y=list(df.iloc[1:,1])

date=x

facto('date')

x=list(df.iloc[1:,0])

x=np.array(range(1,len(x)+1))
active(date,y,y,"Indonesia")
ypred=linear(x,y)

r2_score_linear(y,ypred)
active(date,y,ypred,"Indonesia")
ypred=pangkat(x,y)

r2_score_pangkat(x,y)
active(date,y,ypred,"Indonesia")
ypred=eksp(x,y)

r2_score_exp(x,y)
active(date,y,ypred,"Indonesia")
ypred=orde3(x,y)

r2_score_orde3(y,ypred)
active(date,y,ypred,"Indonesia")
def indonesia_active(x) :

    y =3.9334 + 0.3479*x  -0.4818*(x**2) + 0.0691*(x**3)

    return y
x=list(df.iloc[10:,0])

y=list(df.iloc[10:,2])

x=np.array(range(1,len(x)+1))

date=tgl[9:]

death(date,y,y,"Indonesia")
ypred=linear(x,y)

r2_score_linear(y,ypred)

death(date,y,ypred,"Indonesia")
ypred=pangkat(x,y)

r2_score_pangkat(x,y)

death(date,y,ypred,"Indonesia")
ypred=eksp(x,y)

r2_score_exp(x,y)

death(date,y,ypred,"Indonesia")
ypred=orde3(x,y)

r2_score_orde3(y,ypred)

death(date,y,ypred,"Indonesia")
def indonesia_death(x) :

    y = 4.6538 -3.2099*x + 0.7487*(x**2)  -0.0177*(x**3)

    return y
x=np.array(range(1,25))

x1=np.array(range(1,15))
def kor(y1,y2,country,x) :

    r2_score_linear(y1,y2)

    plt.plot(x,y1,label="indonesia",color="red")

    plt.plot(x,y2,label=country,color="blue")

    plt.legend()

    plt.show()
kor(indonesia_active(x),china_active(x),"china",x)
kor(indonesia_death(x1),china_death(x1),"china",x1)
kor(indonesia_active(x),korea_active(x),"Korea Selatan",x)
kor(indonesia_death(x1),korea_death(x1),"Korea Selatan",x1)
kor(indonesia_active(x),italy_active(x),"Italy",x)
kor(indonesia_death(x1),italy_death(x1),"Italy",x1)
kor(indonesia_active(x),iran_active(x),"Iran",x)
kor(indonesia_death(x1),iran_death(x1),"Iran",x1)
kor(indonesia_active(x),spain_active(x),"Spanyol",x)
kor(indonesia_death(x1),spain_death(x1),"Spanyol",x1)
kor(indonesia_active(x),germany_active(x),"jerman",x)
kor(indonesia_death(x1),germany_death(x1),"Jerman",x1)
kor(indonesia_active(x),aus_active(x),"Australia",x)
kor(indonesia_death(x1),aus_death(x1),"Australia",x1)
kor(indonesia_active(x),usa_active(x),"USA",x)
kor(indonesia_death(x1),usa_death(x1),"USA",x1)