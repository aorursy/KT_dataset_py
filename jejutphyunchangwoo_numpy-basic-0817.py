import numpy as np

import numpy.linalg as lin

import time
a = np.array(2.0)

print(a)
print(type(a))
a.shape
b = np.array([2.0])

print(b)
b.shape

b.shape[0]
c = np.array([2,3,4])

print(c)

print(c.shape)

print("length ", c.shape[0])
m = np.array([[1,2], [3,4], [5,6]])

print(m)

print(m.shape)

print("행 ", m.shape[0])

print("행 ", m.shape[1])
m = np.array([[1,2,3]])

print(m)

print(m.shape)
m = np.array([[1],[2],[3]])

print(m)

print(m.shape)
a=[1, 2.0]

b=np.array([1,2,3])

c=np.array([1,2,3.0])



print(type(a[0]))

print(type(a[1]))

print(b.dtype)

print(c.dtype)

print(type(c[0]))

print(type(c[2]))
a = [[1,2], [3,4,5]]

print(a)

b = np.array([[1,2], [3,4,5]])

print(b)
a = np.array([[1,2,3], [4,5,6], [7,8,9]])

print(a)

print("a[0,0]==>", a[0,0])

print("a[0]==>", a[0])

print("a[:,1]==>", a[:,1])

print("a[1, 1:3]==>", a[1,1:3])
i = [0,2]

b = a[i, :]

print(b)
c = np.array([1,2,3,4,5,6,7,8,9])

print(c[0::2])
print(a[1:-1])
print(a[1:])
print(a)

a[1,2] = 7

a[:,0] = [0,9,1]

print(a)
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]] )

print(a)



a[1:-1, 1:-1] = 0

print(a)
a = np.array([[1,2,3], [4,5,6]])

b = np.array([[1,2,3], [4,5,6]]).T

c = np.array([[1,2,3], [4,5,6]]).transpose()

print(a.shape, b.shape, c.shape)
a = np.array([[1,2,3,4]])

b = np.array([[1,2,3,4]]).T

print(a)

print(b)

print(a.shape)

print(b.shape)
a = np.array([1,2,3,4])

b = np.array([1,2,3,4]).T

print(a.shape, b.shape)
a = np.array([[1,2], [3,4], [5,6], [7,8], [9,10], [11,12] ])

b = a.reshape(3,4)

print(a.shape)

print(a)

print(b.shape)

print(b)
a = np.array([1,2,3,4,5,6])

b = a.reshape(2,3)

print(a)

print(a.shape)

print(b)

print(b.shape)
a = np.array([[1],[2],[3],[4] ])

b = np.array([[1,2,3,4] ]).T

c = np.array([1,2,3,4]).reshape(4,1)

d = np.array([1,2,3,4]).T

print(a)

print(b)

print(c)

print(d)
a = np.array([[1,2,3],[4,5,6],[7,8,9]] )

np.save("a.npy", a)

b = np.load("a.npy")

print(b)
a = np.array([1,3,5,7,9])

b = np.array([3,5,6,7,9])

c = a+b

print(c)

print(type(c))

print(c.dtype)
a = np.array([1,3,5,7,9])

b = np.array([3,5,6,7,9.0])

c = a+b

print(c)

print(a.dtype)

print(c.dtype)
a = np.array([1,2,3,4,5])

b = np.array([1,2,3,4,5])

print( np.sum(a*b) )
p = np.array([2,2])

q = np.array([3,7])

(np.sum( (p-q) ** 2) ) ** (0.5)
a = np.array([1,2,3,3,2,5]).reshape(2,3)

b = np.array([[-1,3,5], [1,4,2]])

print(a)

print(b)

print(a+b)
a = np.array([1,2,3,3,2,5]).reshape(2,3)

b = np.array([[-1,-1,-1], [0,0,0]])

print(a)

print(b)

print(a*b)
a = np.array([[1,2],[3,4],[5,6]])

b = a ** 2

c = a * a

print(b)

print(c)
x = np.array([[1,2],[4,5],[6,7] ])

y = np.array([[1,2],[4,5]])

print(x)

print(y)

print(np.matmul(x,y))
x = np.array([[1,2,3], [4,5,6]])



print(np.matmul(x, x.T))

print(np.matmul(x, np.transpose(x) ))
a = np.random.normal(0, 0.1, [500,500])

b = np.random.normal(0, 0.1, [500,500])



start_time = time.time()

c = np.matmul(a, b)

print(time.time() - start_time)
def matmul_py(A, B):

    m = len(A)

    n = len(A[0] )

    p = len(B[0] )

    C = [[0]*p for i in range(m)]

    

    for i in range(0,m):

        for j in range(0,p):

            for k in range(0, n):

                C[i][j] += A[i][k] * B[k][j]

    return C



A = a.tolist()

B = b.tolist()

start_time = time.time()

matmul_py(A,B)

print(time.time() - start_time)
a = np.array([[2,2,0],[-2,1,1],[3,0,1]])

inv = lin.inv(a)

print(inv.shape)

print(inv.dtype)

print(inv) #inv() 함수: 정방행렬에 대해서만 적용 가능
A = np.array([[1,2], [2,4], [3,6.4]])

pinv = np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T)

print(A)

print(pinv)
def pinv(A):

    return np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T)



A = np.array([[1,2], [2,4], [3,6.4]])

print(pinv(A))
a = np.array([1,2,3,4,5,6])

b = a+4

print(a)

print(b)
a = np.array([1,2,3,4,5,6])

b = a*2

print(a)

print(b)
a = np.array([1,2,3,4,5,6])

b = a*2 + 3

print(a)

print(b)
a = np.array([78, 77, 76, 75])

b = (a-32) / 1.8

print(a)

print(b)
a = np.array([26, 25, 24, 23])

b = a * 1.8 + 32

print(a)

print(b)
a = np.arange(31)

b = a * 1.8 + 32

print(a)

print(b)
#정수 반올림

a = np.array([121.5732, 458.4221])

b = ( (a+5) / 10).astype(int) * 10

print(a)

print(b)
#실수 반올림

a = np.array([31.5732, 8.4221])

b = ( (a+0.05) * 10).astype(int) / 10

print(a)

print(b)
x = np.array([1,2,3,4,5])

w = 4.5

b = 7.2

print(w*x + b)
x = np.arange(-3, 3, 1)

print(x)

print(2 * x + 0.1)
a = np.array([1,2,3,4,5,6])

a2 = np.array([4 for i in range(6)])

print(a)

print(a2)

print(a + a2)
a = np.array([1,2,3,4,5,6]).reshape(2,3)

b = a+4

print(a)

print(b)
a = np.array([1,2,3,4,5,6]).reshape(2,3)

b = a * -1

print(a)

print(b)
a = np.array([[1,1,1], [1,1,1], [1,1,1] ])

b = np.array([0,1,2])

print(a+b)
a = np.array([ [1,2,3], [4,5,6] ])

b = np.array([1,2,3])

print(a+b)
x = np.array([ [1,2], [3,4], [5,6], [7,8] ]) 

w = np.array([ [0.1, 0.2, 0.3], [0.2, 0.3, 1] ]) 

b = np.array([1,2,3])

print(np.matmul(x,w) + b)
a = np.array([ [0.1, 2, 34], [0.9, 7, 12], [0.2, 1, 6], [0.5, 1.5, 27] ])

n = a / np.max(a, axis=0)

print(a)

print(n)
#1

w = 20.3

b = 351.2

x = [13, 15, 20]

y = np.array(x) * w + b

print(y)
y = 840

x = (y - b) / w

print(y)

print(x)
w = 20.3

b = 351.2



x = np.array([4,9,10,14,4,7,12,22,1,3,8,11,5,6,10,11,16,13,13,10])

y = np.array([390,580,650,730,410,530,600,790,350,400,590,640,450,520,690,690,770,700,730,640])



error = (y - (x*w+b))**2

result = np.sqrt(np.mean(error) )

print(result)
x = np.array([[4,9,10,14,4,7,12,22,1,3,8,11,5,6,10,11,16,13,13,10]]).T

y = np.array([[390,580,650,730,410,530,600,790,350,400,590,640,450,520,690,690,770,700,730,640]]).T



print(x.shape)

print(y.shape)
A = np.append(x, np.ones((x.shape[0], 1)), axis = 1)

print(A)

print(A.shape)
def pinv(A):

    return np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T)



W = np.matmul( pinv(A), y )

w, b = W

print(w, b)
error2 = (y - (x*w+b))**2

result2 = np.sqrt(np.mean(error2) )



print(result)

print(result2)
Interest_rate = np.array([[2.75, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.25, 2.25, 2.25,2,2,2,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75]]).T

Unemployment_rate = np.array([[5.3, 5.3, 5.3, 5.3, 5.4, 5.6, 5.5, 5.5, 5.5, 5.6, 5.7, 5.9, 6, 5.9, 5.8, 6.1, 6.2, 6.1, 6.1, 6.1, 5.9, 6.2, 6.2, 6.1]]).T

Stock_Index_Price = [1464, 1394, 1357, 1293, 1256, 1254, 1234, 1195, 1159, 1167, 1130, 1075, 1047, 965, 943, 958, 971, 949, 884, 866, 876, 822, 704, 719]



A = np.append(Interest_rate, Unemployment_rate, axis=1)

A = np.append(A, np.ones((Interest_rate.size, 1)), axis=1)

print(A.shape)



W = np.matmul(pinv(A), Stock_Index_Price)

print(W)
x_data = np.array([[5.1, 6.2, 1], [5.1, 6.1, 1], [5.1, 6.2, 1], [5.2, 6.2, 1] ])

result = np.matmul(x_data, W)

print(result)
print( result[1] - result[0])

print( result[3] - result[2])