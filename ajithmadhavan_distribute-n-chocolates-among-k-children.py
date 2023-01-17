k=4

n=52

A=[0]*k

Sum=0

counter=False

li=[]

print(f"Total number of Students is {k} and the count of choclates to be distributed is {n}\n")

def func(k,chocsLeft,stopPoint):

    count=1

    print(f"Still left with {chocsLeft} choclates. Last student got {stopPoint} choclates")

    while chocsLeft>0:

        l2=[0]*k

        SumAll=0

        for i in range(k):

            #print(i)

            if chocsLeft-SumAll<=stopPoint+1:

                l2[i]=chocsLeft-SumAll

                print("Distributed all choclates")

                break

            else:

                l2[i]=stopPoint+1

                SumAll=SumAll+l2[i]

                stopPoint=l2[i]

                print(k,chocsLeft-SumAll,stopPoint)

                if ((i==k-1) and (chocsLeft-SumAll>0)):

                    func(k,chocsLeft-SumAll,stopPoint)

        li.append(l2)

        #print(li)

        chocsLeft=0

    

        

for i in range(k):

        if n-Sum<i+1:

            A[i]=n-Sum

            print("Student number=",i+1)

            print("Total choclates distributed=",Sum)

            print("Total choclates left=",n-Sum)

            print("list=",A)

            print('\n')

            print("Distributed all choclates")

            break

        else:

            A[i]=i+1

            Sum=Sum+A[i]

            print("Student number=",i+1)

            print("Total choclates distributed=",Sum)

            print("Total choclates left=",n-Sum)

            print("list=",A)

            print('\n')

            if ((i==k-1) and (n-Sum>0)):

                func(k,n-Sum,A[i])

print(li)

print(A)

print('\n')
k=4

n=16

Sum=0

chocLeft=n

lastElem=0

Final=[]

def distChoc(k,Sum,lastElem,chocLeft,st):

    li=[0]*k

    for i in range(k):

            li[i]=lastElem+1

            Sum=Sum+li[i]

            if Sum>=n:

                li[i]=chocLeft

                print(f"The students will get the chocolates in {li} order")

                Final.append(li)

                break

            lastElem=li[i]

            chocLeft=n-Sum

            if (chocLeft>0 and i==k-1):

                print(f"The students will get the chocolates in {li} order")

                print(f"Still left with {chocLeft} more choclates")

                Final.append(li)

                distChoc(k,Sum,lastElem,chocLeft,Final)

print(f"We have {k} students and {n} chocolates\n")

distChoc(k,Sum,lastElem,chocLeft,Final)



FinalList=[sum(x) for x in zip(*Final)]

print("\nFinal stats for each students is ", FinalList)