import numpy as np
A = np.random.uniform(5, 2.5, (10, 5))

print(np.mean(A[5,:]))

print(np.median(A[:,3]))

print(np.sum(A, axis=1))

print(np.sum(A, axis=0))

print(np.sum(A))

print(np.linalg.det(A[:3,:3]))

y=np.sum(A, axis=1)

z=np.mean(y)

print(z)

d=y[(y<z)]

print(d)

print(d.size)

print(A[A[:, 0]>A[:,4], :])

print(A.shape)

a=np.sum(A, axis=1)

b=np.mean(a)

print(b)

c=a[(a>b)]

print(c)

rank = np.linalg.matrix_rank(A)

print(rank)
Z = np.loadtxt('../input/cardio_train.csv', delimiter=';', skiprows=1)

print(Z)

print(Z.shape)

Z[:, 1]=Z[:, 1]//365.25

print(Z[:, 1]>50)

print(Z.size)

print(np.max(Z[:, 4]), np.min(Z[:, 4]))

a=np.max(Z[:, 4])

print(a.size)

x=(Z[(Z[:, 3]==0) & (Z[:,-1]==1)])

print(x.size)

#y=(Z[(Z[:, 2]==1) & (Z[:,3])])

#k=(Z[(Z[:, 2]==2) & (Z[:,3])])

#print(np.mean(y)<np.mean(x))

d=Z[1:,2:4]

print(d[d[:,0]==2,:])

f=d[d[:,0]>=2,:]

print(np.mean(f[:,1]))

m=d[d[:,0]==1,:]

print(m)

print(np.mean(m[:,1]))

l=Z[1:,4]

print(l)