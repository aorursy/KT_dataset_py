import numpy as np
np.__version__
x=np.zeros(10, dtype='int8')

x
x[4]=1

x
x[-3:]=2

x
y = np.random.normal(150, 200, 100)
mu=np.mean(y)

mu
dlt=np.std(y)

dlt
z=y[(y >= mu-3*dlt) & (y <= mu+3*dlt)]

z.size/100
z[(z<mu-3*dlt)|(z>mu+3*dlt)]
b = np.linspace(0, np.pi/2, 20)

b
cos = np.cos(b)

cos
sin = np.sin(np.pi/2-b)

sin
def close(x, y, eps):

    if np.all(abs(x - y)<eps):

        return True

    else:

        return False
close(cos,sin,1e-6)
M = np.random.normal(5, 2.5, (10, 5))

M
np.mean(M[4,:])
np.median(M[:,2])
print(np.sum(M, axis=1))

print(np.sum(M, axis=0))

print(np.sum(M))
np.linalg.det(M[:3, :3])
np.sum(M, axis=1)[np.sum(M, axis=1)<np.mean(np.sum(M, axis=1))].size
N=M[M[:,0]>M[:,4],:]

print(N)

N.shape
K=M[np.sum(M, axis=1)>np.mean(np.sum(M, axis=1))]

print(K)

np.linalg.matrix_rank(K)
df = np.loadtxt('../input/cardio_train.csv', delimiter=';', skiprows=1)

df.shape
df[:,1] = np.floor(df[:,1] / 365.25)
np.count_nonzero((df[:, 1] > 50)==True)
f = np.mean(df[df[:, 2]==1, 3])

m = np.mean(df[df[:, 2]==2, 3])

if m > f:

    print('Верно')

else:

    print('Не верно')
print(np.min(df[:,4]))

print(np.max(df[:,4]))
K=df[df[:,4]==np.max(df[:,4])]

print(K)

K.shape[0]
P=(df[:,-4]==0)&(df[:,-1]==1)

np.count_nonzero(P==True)