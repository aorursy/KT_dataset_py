import numpy as np
def Cal_LU(D,g):

    A=np.array((D),dtype=float)

    f=np.array((g),dtype=float)

    n = f.size

    for i in range(0,n-1):     # Loop through the columns of the matrix

        for j in range(i+1,n):     # Loop through rows below diagonal for each column

            if A[i,i] == 0:

                print("Error: Zero on diagonal!")

                print("Need algorithm with pivoting")

                break

            m = A[j,i]/A[i,i]

            A[j,:] = A[j,:] - m*A[i,:]

            f[j] = f[j] - m*f[i]

    return A,f



def Back_Subs(A,f):

    n = f.size

    x = np.zeros(n)             # Initialize the solution vector, x, to zero

    x[n-1] = f[n-1]/A[n-1,n-1]    # Solve for last entry first

    for i in range(n-2,-1,-1):      # Loop from the end to the beginning

        sum_ = 0

        for j in range(i+1,n):        # For known x values, sum and move to rhs

            sum_ = sum_ + A[i,j]*x[j]

        x[i] = (f[i] - sum_)/A[i,i]

    return x
A = np.array([[2,3,5],[3,4,1],[6,7,2]])

f = np.array([23,14,26])

B,g = Cal_LU(A,f)

x= Back_Subs(B,g)

print(x)
y = np.linalg.solve(A,f)

print(y)
A = np.array([[0.9,0.3,0.1], [0.1,0.5,0.2],[0,0.2,0.7]])

f = np.array([30.0,25.0,10.0])

B,g = Cal_LU(A,f)

x = Back_Subs(B,g)

print(x)
x = np.linalg.solve(A,f)

print(x)
def Cal_LU_pivot(D,g):

    A=np.array((D),dtype=float)

    f=np.array((g),dtype=float)

    n = len(f)

    for i in range(0,n-1):     # Loop through the columns of the matrix

        

        if np.abs(A[i,i])==0:

            for k in range(i+1,n):

                if np.abs(A[k,i])>np.abs(A[i,i]):

                    A[[i,k]]=A[[k,i]]             # Swaps ith and kth rows to each other

                    f[[i,k]]=f[[k,i]]

                    break

                    

        for j in range(i+1,n):     # Loop through rows below diagonal for each column

            m = A[j,i]/A[i,i]

            A[j,:] = A[j,:] - m*A[i,:]

            f[j] = f[j] - m*f[i]

    return A,f
A = np.array([[0,3,5],[3,0,1],[6,7,2]])

f = np.array([23,14,26])

B,g = Cal_LU_pivot(A,f)

x= Back_Subs(B,g)

print(x)
x = np.linalg.solve(A,f)

print(x)