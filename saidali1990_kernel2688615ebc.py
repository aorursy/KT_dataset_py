def Solution(A):

    n=len(A)

    x=False

    sum2 = 0

    for i in range(1,n-3):

        sum1=(sum(A[0:i]))

        j=i+1

        while sum2<sum1:

            sum2=sum(A[-(j):n])

            j=j+1

        if(sum1==sum2):

            if(sum1==sum(A[i+1:-(j)])):

                x= True

                print(sum1)

                print(sum2)

                print(sum(A[i+1:-(j)]))

                break

    return x               
A =[1,3,4,2,2,2,1,1,2]
Solution(A)
D =[]

K =[]

Z =[]

for i in range(10000):

    D.append(1)

    D.append(2)

print(sum(D))
Solution(D)
C =[1,1,1,1,1,1,1]
Solution(C)