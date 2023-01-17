l1=[]
def recur_fibo(n):  

   if n <= 1:  

       return n  

   else:  

       return(recur_fibo(n-1) + recur_fibo(n-2))  

# take input from the user  

nterms = int(input("How many terms? "))  

# check if the number of terms is valid  

if nterms <= 0:  

   print("Plese enter a positive integer")  

else:  

   print("Fibonacci sequence:")  

   for i in range(nterms):

        l1.append(recur_fibo(i))

        print(l1[i])

  
l1
l2=[3,5,3,4,8,1,2]
def checkfib(l1):

    not_flag=0

    i=1

    while i < (len(l1)-2):

            if(l1[-i]-l1[-i-1]!=l1[-i-2]):

                not_flag=1

            i=i+1

    return not_flag
if checkfib(l1):

    print("not a fibonnaci")

else:

    print("fibonnaci")
if checkfib(l2):

    print("not a fibonnaci")

else:

    print("fibonnaci")