import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from numpy.linalg import norm

import numpy.linalg as la
N = 10

A = np.array([[N+2,1,1],

             [1,N+4,1],

             [1,1,N+6]],dtype = "float")

b = np.array([N+4,N+6,N+8],dtype = "float").reshape(-1,1)

b
def create_matrix(A,b,e = 0):

    if e <= 0:

        e = norm(A)

    u = 2/(norm(A)+e)

    B = np.eye(len(A)) - u*A

    c = u*b

    return(B,c)



def simple_iteration_method(A,b,e = 1e-5):

    B,c = create_matrix(A,b)

    x = c.copy()

    x_old = x + 1

    coef = norm(B)/(1-norm(B))

    print(norm(B))

    while coef*norm(x - x_old) > e:

        x_old = x.copy()

        x = B.dot(x) + c

    return x



def simple_iteration_method2(A,b,e = 1e-5):

    B,c = create_matrix(A.T.dot(A),A.T.dot(b))

    x = c.copy()

    x_old = x + 1000

    coef = norm(B)/(1-norm(B))

    print(coef)

    k=0

    while norm(x - x_old) > e:

        x_old = x.copy()

        k+=1

        x = B.dot(x) + c

    print("Количество итерация:",k)

    return x
simple_iteration_method(A,b,1e-10)

simple_iteration_method2(A,b,1e-10)
def trapezoid_matrix(A,b):

        e = 1.0e-30

        M = np.concatenate((A,b),axis=1)

        i, j = 0, 0

        while i != M.shape[0]  and j != M.shape[1]:

            while abs(M[i, j]) <= e:

                print(i,j,"Тут")

                for k in range(i+1, M.shape[0]):

                    if abs(M[k, j]) >= e:

                        M[i, :], M[k, :] = M[k,:].copy(), M[i, :].copy()

                        break

                j = (j+1) if abs(M[i, j]) <= e else j

                if j ==  M.shape[1]:

                    return matrix(M)

            for k in range(i+1,M.shape[0]):

                M[k,:] = M[k, :] - M[k, j]/M[i, j]*M[i, :]



            i+=1

            j+=1

        return M

    

def Gauss_method(A,b):

    M = trapezoid_matrix(A,b)

    answers = np.zeros(M.shape[1]-1)

    i = len(M) - 1

    j = M.shape[1] - 2

    while i != -1 and j!=-1:

            answers[j] = (M[i, -1] - sum(answers * M[i, :-1])) / M[i, j]

            i -= 1

            j -= 1

    return answers.reshape(-1,1)
Gauss_method(A,b)
def create_matrix1(i,j):

    ans = np.zeros((len(i),len(j)))

    ans += (i==j)*1

    ans += (i<j)*(-1)

    return ans

def create_matrix2(i,j):

    ans = np.ones((len(i),len(j)))

    ans -= 2*(i<j)

    return ans
n = 322

e = 0

A = np.fromfunction(create_matrix1,(n,n), dtype = float) - e*N*np.fromfunction(create_matrix2,(n,n), dtype = float)

b = np.array([-1 for i in range(n-1)]+[1]).reshape(-1,1)

norm(A)
Gauss_method(A,b)
simple_iteration_method(A,b,1e-10)

simple_iteration_method2(A,b,1e-10)
la.solve(A,b)