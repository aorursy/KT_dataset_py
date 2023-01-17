import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
v = [3,4]
u = [1,2,3]
v ,u
type(v)
w = np.array([9,5,7])
type(w)
w.shape[0]
w.shape
a = np.array([7,5,3,9,0,2])
a[0]
a[1:]
a[1:4]
a[-1]
a[-3]
a[-6]
a[-3:-1]
v = [3,4]
u = [1,2,3]
plt.plot (v)
plt.plot([0,v[0]] , [0,v[1]])
plt.plot([0,v[0]] , [0,v[1]])
plt.plot([8,-8] , [0,0] , 'k--')
plt.plot([0,0] , [8,-8] , 'k--')
plt.grid()
plt.axis((-8, 8, -8, 8))
plt.show()
v1 = np.array([1,2])
v2 = np.array([3,4])
v3 = v1+v2
v3 = np.add(v1,v2)
print('V3 =' ,v3)
plt.plot([0,v1[0]] , [0,v1[1]] , 'r' , label = 'v1')
plt.plot([0,v2[0]] , [0,v2[1]], 'b' , label = 'v2')
plt.plot([0,v3[0]] , [0,v3[1]] , 'g' , label = 'v3')
plt.plot([8,-8] , [0,0] , 'k--')
plt.plot([0,0] , [8,-8] , 'k--')
plt.grid()
plt.axis((-8, 8, -8, 8))
plt.legend()
plt.show()
plt.plot([0,v1[0]] , [0,v1[1]] , 'r' , label = 'v1')
plt.plot([0,v2[0]]+v1[0] , [0,v2[1]]+v1[1], 'b' , label = 'v2')
plt.plot([0,v3[0]] , [0,v3[1]] , 'g' , label = 'v3')
plt.plot([8,-8] , [0,0] , 'k--')
plt.plot([0,0] , [8,-8] , 'k--')
plt.grid()
plt.axis((-8, 8, -8, 8))
plt.legend()
plt.show()
u1 = np.array([3,4])
a = .5
u2 = u1*a
plt.plot([0,u1[0]] , [0,u1[1]] , 'r' , label = 'v1')
plt.plot([0,u2[0]] , [0,u2[1]], 'b--' , label = 'v2')
plt.plot([8,-8] , [0,0] , 'k--')
plt.plot([0,0] , [8,-8] , 'k--')
plt.grid()
plt.axis((-8, 8, -8, 8))
plt.legend()
plt.show()
u1 = np.array([3,4])
a = -.3
u2 = u1*a
plt.plot([0,u1[0]] , [0,u1[1]] , 'r' , label = 'v1')
plt.plot([0,u2[0]] , [0,u2[1]], 'b' , label = 'v2')
plt.plot([8,-8] , [0,0] , 'k--')
plt.plot([0,0] , [8,-8] , 'k--')
plt.grid()
plt.axis((-8, 8, -8, 8))
plt.legend()
plt.show()
a1 = [5 , 6 ,8]
a2 = [4, 7 , 9]
print(np.multiply(a1,a2))
a1 = np.array([1,2,3])
a2 = np.array([4,5,6])

dotp = a1@a2
print(" Dot product - ",dotp)

dotp = np.dot(a1,a2)
print(" Dot product usign np.dot",dotp)

dotp = np.inner(a1,a2)
print(" Dot product usign np.inner", dotp)

dotp = sum(np.multiply(a1,a2))
print(" Dot product usign np.multiply & sum",dotp)

dotp = np.matmul(a1,a2)
print(" Dot product usign np.matmul",dotp)

dotp = 0
for i in range(len(a1)):
    dotp = dotp + a1[i]*a2[i]
print(" Dot product usign for loop" , dotp)
v3 = np.array([1,2,3,4,5,6])
length = np.sqrt(np.dot(v3,v3))
length
v3 = np.array([1,2,3,4,5,6])
length = np.sqrt(sum(np.multiply(v3,v3)))
length
v3 = np.array([1,2,3,4,5,6])
length = np.sqrt(np.matmul(v3,v3))
length
v1 = [2,3]
length_v1 = np.sqrt(np.dot(v1,v1))
norm_v1 = v1/length_v1
length_v1 , norm_v1
v1 = [2,3]
norm_v1 = v1/np.linalg.norm(v1)
norm_v1
#First Method
v1 = np.array([8,4])
v2 = np.array([-4,8])
ang = np.rad2deg(np.arccos( np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))))
plt.plot([0,v1[0]] , [0,v1[1]] , 'r' , label = 'v1')
plt.plot([0,v2[0]]+v1[0] , [0,v2[1]]+v1[1], 'b' , label = 'v2')
plt.plot([16,-16] , [0,0] , 'k--')
plt.plot([0,0] , [16,-16] , 'k--')
plt.grid()
plt.axis((-16, 16, -16, 16))
plt.legend()
plt.title('Angle between Vectors - %s'  %ang)
plt.show()
#Second Method
v1 = np.array([4,3])
v2 = np.array([-3,4])
lengthV1 = np.sqrt(np.dot(v1,v1)) 
lengthV2  = np.sqrt(np.dot(v2,v2))
ang = np.rad2deg(np.arccos( np.dot(v1,v2) / (lengthV1 * lengthV2)))
print('Angle between Vectors - %s' %ang)
v1 = np.array([1,2,-3])
v2 = np.array([7,-4,2])
fig = plt.figure()
ax = Axes3D(fig)
ax.plot([0, v1[0]],[0, v1[1]],[0, v1[2]],'b')
ax.plot([0, v2[0]],[0, v2[1]],[0, v2[2]],'r')
ang = np.rad2deg(np.arccos( np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)) ))
plt.title('Angle between vectors: %s degrees.' %ang)
# https://www.youtube.com/watch?v=FCmH4MqbFGs

v1 = np.array([1,2,3])
v2 = np.array([4,5,6])
np.inner(v1,v2)

print("\n Inner Product ==>  \n", np.inner(v1,v2))
print("\n Outer Product ==>  \n", np.outer(v1,v2))
v1 = np.array([1,2,3])
v2 = np.array([4,5,6])
print("\nVector Cross Product ==>  \n", np.cross(v1,v2))
A = np.array([[1,2,3,4] , [5,6,7,8] , [10 , 11 , 12 ,13] , [14,15,16,17]])
A
type(A)
A.dtype
B = np.array([[1.5,2.07,3,4] , [5,6,7,8] , [10 , 11 , 12 ,13] , [14,15,16,17]])
B
type(B)
B.dtype
A.shape
A[0,]
A[:,0]
A[0,0]
A[0][0]
A[1:3 , 1:3]
np.zeros(9).reshape(3,3)
np.zeros((3,3))
np.ones(9).reshape(3,3)
np.ones((3,3))
X = np.random.random((3,3))
X
I = np.eye(9)
I
D = np.diag([1,2,3,4,5,6,7,8])
D
M = np.random.randn(5,5)
U = np.triu(M)
L = np.tril(M)
print("lower triangular matrix - \n" , M)
print("\n")


print("lower triangular matrix - \n" , L)
print("\n")

print("Upper triangular matrix - \n" , U)
A = np.array([[1,2] , [3,4] ,[5,6]])
B = np.array([[1,1] , [1,1]])
C = np.concatenate((A,B))
C , C.shape , type(C) , C.dtype
np.full((5,5) , 8)
M
M.flatten()
#********************************************************#
M = np.array([[1,2,3],[4,-3,6],[7,8,0]])
N = np.array([[1,1,1],[2,2,2],[3,3,3]])

print("\n First Matrix (M)  ==>  \n", M)
print("\n Second Matrix (N)  ==>  \n", N)

C = M+N
print("\n Matrix Addition (M+N)  ==>  \n", C)

# OR

C = np.add(M,N,dtype = np.float64)
print("\n Matrix Addition using np.add  ==>  \n", C)

#********************************************************#
#********************************************************#
M = np.array([[1,2,3],[4,-3,6],[7,8,0]])
N = np.array([[1,1,1],[2,2,2],[3,3,3]])

print("\n First Matrix (M)  ==>  \n", M)
print("\n Second Matrix (N)  ==>  \n", N)

C = M-N
print("\n Matrix Subtraction (M-N)  ==>  \n", C)

# OR

C = np.subtract(M,N,dtype = np.float64)
print("\n Matrix Subtraction using np.subtract  ==>  \n", C)

#********************************************************#
M = np.array([[1,2,3],[4,-3,6],[7,8,0]])

C = 10

print("\n Matrix (M)  ==>  \n", M)

print("\nMatrices Scalar Multiplication ==>  \n", C*M)

# OR

print("\nMatrices Scalar Multiplication ==>  \n", np.multiply(C,M))
M = np.array([[1,2,3],[4,-3,6],[7,8,0]])

print("\n Matrix (M)  ==>  \n", M)

print("\nTranspose of M ==>  \n", np.transpose(M))

# OR

print("\nTranspose of M ==>  \n", M.T)
M = np.array([[1,2,3],[4,-3,6],[7,8,0]])

print("\n Matrix (M)  ==>  \n", M)

print("\nDeterminant of M ==>  ", np.linalg.det(M))
M = np.array([[1,2,3],[4,-3,6],[7,8,0]])

print("\n Matrix (M)  ==>  \n", M)

print("\nRank of M ==> ", np.linalg.matrix_rank(M))
M = np.array([[1,2,3],[4,-3,6],[7,8,0]])

print("\n Matrix (M)  ==>  \n", M)

print("\nTrace of M ==> ", np.trace(M))
M = np.array([[1,2,3],[4,-3,6],[7,8,0]])

print("\n Matrix (M)  ==>  \n", M)

print("\nInverse of M ==> \n", np.linalg.inv(M))
M = np.array([[1,2,3],[4,-3,6],[7,8,0]])
N = np.array([[1,1,1],[2,2,2],[3,3,3]])

print("\n First Matrix (M)  ==>  \n", M)
print("\n Second Matrix (N)  ==>  \n", N)

print("\n Point-Wise Multiplication of M & N  ==> \n", M*N)

# OR

print("\n Point-Wise Multiplication of M & N  ==> \n", np.multiply(M,N))
M = np.array([[1,2,3],[4,-3,6],[7,8,0]])
N = np.array([[1,1,1],[2,2,2],[3,3,3]])

print("\n First Matrix (M)  ==>  \n", M)
print("\n Second Matrix (N)  ==>  \n", N)

print("\n Matrix Dot Product ==> \n", M@N)

# OR

print("\n Matrix Dot Product using np.matmul ==> \n", np.matmul(M,N))

# OR

print("\n Matrix Dot Product using np.dot ==> \n", np.dot(M,N))
M = np.array([[1,2,3],[4,-3,6],[7,8,0]])
N = np.array([[1,1,1],[2,2,2],[3,3,3]])

print("\n First Matrix (M)  ==>  \n", M)
print("\n Second Matrix (N)  ==>  \n", N)


print("\n Matrix Division (M/N)   ==> \n", M/N)

# OR

print("\n Matrix Division (M/N)   ==> \n", np.divide(M,N))
N = np.array([[1,1,1],[2,2,2],[3,3,3]])

print("\n Matrix (N)  ==>  \n", N)


print ("Sum of all elements in a Matrix  ==>")
print (np.sum(N))
N = np.array([[1,1,1],[2,2,2],[3,3,3]])

print("\n Matrix (N)  ==>  \n", N)

print ("Column-Wise summation ==> ")
print (np.sum(N,axis=0))
N = np.array([[1,1,1],[2,2,2],[3,3,3]])

print("\n Matrix (N)  ==>  \n", N)

print ("Row-Wise summation  ==>")
print (np.sum(N,axis=1))
M1 = np.array([[1,2,3] , [4,5,6]]) 
M1
M2 = np.array([[10,10,10],[10,10,10]])
M2
np.kron(M1,M2)
A = np.array([[1,2,3] ,[4,5,6]])
v = np.array([10,20,30])
print ("Matrix Vector Multiplication  ==> \n" , A*v)
A = np.array([[1,2,3] ,[4,5,6]])
v = np.array([10,20,30])

print ("Matrix Vector Multiplication  ==> \n" , A@v)
M1 = np.array([[1,2],[4,5]])
M1
#Matrix to the power 3

M1@M1@M1
#Matrix to the power 3

np.linalg.matrix_power(M1,3)
# Create Tensor

T1 = np.array([
  [[1,2,3],    [4,5,6],    [7,8,9]],
  [[10,20,30], [40,50,60], [70,80,90]],
  [[100,200,300], [400,500,600], [700,800,900]],
  ])

T1
T2 = np.array([
  [[0,0,0] , [0,0,0] , [0,0,0]],
  [[1,1,1] , [1,1,1] , [1,1,1]],
  [[2,2,2] , [2,2,2] , [2,2,2]]
    
])

T2
A = T1+T2
A
np.add(T1,T2)
S = T1-T2
S
np.subtract(T1,T2)
P = T1*T2
P
np.multiply(T1,T2)
D = T1/T2
D
np.divide(T1,T2)
T1
T2
np.tensordot(T1,T2)
A = np.array([[1,10,3] , [4,5,6] , [7,8,9]])
A
B = np.random.random((3,1))
B
# Ist Method
X = np.dot(np.linalg.inv(A) , B)
X
# 2nd Method
X = np.matmul(np.linalg.inv(A) , B)
X
# 3rd Method
X = np.linalg.inv(A)@B
X
# 4th Method
X = np.linalg.solve(A,B)
X