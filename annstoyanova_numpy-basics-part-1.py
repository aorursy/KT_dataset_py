import numpy as np
np.__version__
x[-3:]=2

x
x=np.zeros(10)

x
x[4]=1

x
z2 = z[(z>=z_mean-z_std)&(z<=z_mean+z_std)]

z3=z2.size/z.size

z3
z_std=z.std()

z_std
z_mean=z.mean()

z_mean
z=np.random.uniform(150,200,100)

z

a=np.linspace(0,np.pi/2,20)

a
coss=np.cos(a)

coss
sinn=np.sin(np.pi/2-a)

sinn
coss-sinn
def close (x,y,eps):

    if np.abs(x-y)<eps:

        return True

    else:

        return False
close(coss,sinn,1e-6)
np.abs(coss-sinn)<0.0005
close_v=np.vectorize(close)

close_v(coss,sinn,1e-6)

import numpy as np

A = np.random.normal(5, 2.5, (10, 5))

A
x=A[4:]

x_mean=x.mean()

x_mean
a=A[:2]

a_median=np.median(a)

a_median
import numpy as np

print(np.sum(A, axis=1))

print(np.sum(A, axis=0))
print(np.sum(A))
b=(A[:3, :3])

c=np.linalg.det(b)

c
sumM=np.sum(A, axis=1)

print(sumM)

mean=np.mean(sumM)

print(mean)



a=sumM[sumM < mean]

print(a)

kol=a.shape[0]

kol







    

a=(A[A[:,0]>A[:,-1], :])

print(a)

print(a.ndim,a.shape)
sumM=np.sum(A, axis=1)

print(sumM)

mean=np.mean(sumM)

print(mean)

X=A[sumM > mean,:]

print(X)



s=np.linalg.matrix_rank(X)

s





df = np.loadtxt('../input/cardio_train.csv', delimiter=';', skiprows=1)

print(df.shape)
x2=df[:,1]

print(x2)

y=x2/365.25

age=np.floor(y)

print(age)

df[:,1]=age

print(df)



n=age[age>50]

print(n)

print(n.size)
mean = np.mean(df[:, 3][df[:, 2] == 2]) > np.mean(df[:, 3][df[:, 2] == 1])

mean
wmax = np.max(df[:, 4])

wmin = np.min(df[:, 4])

print(wmax)

print(wmin)

Maxx = df[df[:,4] == np.max(df[:, 4])].size# кол-во макс веса



Maxx
a = df[df[:, -4] == 0][:, -1][df[df[:, -4] == 0][:, -1] == 1].size#кол-во некурящих

a