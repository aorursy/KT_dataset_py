# Function for nth Fibonacci number 

x=int(input("enter the number:")) 

def Fibonacci(n): 

    if n<0: 

        print("Incorrect input") 

    # First Fibonacci number is 0 

    elif n==0: 

        return 0

    # Second Fibonacci number is 1 

    elif n==1: 

        return 1

    else: 

        return Fibonacci(n-1)+Fibonacci(n-2) 

  

# Driver Program 

  

print(Fibonacci(x)) 

  

#mihir
# Program to display the Fibonacci sequence up to n-th term

nterms = int(input("How many terms? "))

# first two terms

n1, n2 = 0, 1

count = 0

# check if the number of terms is valid

if nterms <= 0:

   print("Please enter a positive integer")

elif nterms == 1:

   print("Fibonacci sequence upto",nterms,":")

   print(n1)

else:

   print("Fibonacci sequence:")

   while count < nterms:

       print(n1)

       nth = n1 + n2

       # update values

       n1 = n2

       n2 = nth

       count += 1
# Python3 program to implement Goldbach's  

# conjecture 

import math 

MAX = 10000; 

  

# Array to store all prime less  

# than and equal to 10^6 

primes = []; 

  

# Utility function for Sieve of Sundaram 

def sieveSundaram(): 

      

    # In general Sieve of Sundaram, produces  

    # primes smaller than (2*x + 2) for a  

    # number given number x. Since we want 

    # primes smaller than MAX, we reduce  

    # MAX to half. This array is used to  

    # separate numbers of the form i + j + 2*i*j  

    # from others where 1 <= i <= j 

    marked = [False] * (int(MAX / 2) + 100); 

  

    # Main logic of Sundaram. Mark all  

    # numbers which do not generate prime 

    # number by doing 2*i+1 

    for i in range(1, int((math.sqrt(MAX) - 1) / 2) + 1): 

        for j in range((i * (i + 1)) << 1,  

                        int(MAX / 2) + 1, 2 * i + 1): 

            marked[j] = True; 

  

    # Since 2 is a prime number 

    primes.append(2); 

  

    # Print other primes. Remaining primes  

    # are of the form 2*i + 1 such that  

    # marked[i] is false. 

    for i in range(1, int(MAX / 2) + 1): 

        if (marked[i] == False): 

            primes.append(2 * i + 1); 

  

# Function to perform Goldbach's conjecture 

def findPrimes(n): 

      

    # Return if number is not even  

    # or less than 3 

    if (n <= 2 or n % 2 != 0): 

        print("Invalid Input"); 

        return; 

  

    # Check only upto half of number 

    i = 0; 

    while (primes[i] <= n // 2): 

          

        # find difference by subtracting  

        # current prime from n 

        diff = n - primes[i]; 

  

        # Search if the difference is also 

        # a prime number 

        if diff in primes: 

              

            # Express as a sum of primes 

            print(primes[i], "+", diff, "=", n); 

            return; 

        i += 1; 

  

# Driver code 

  

# Finding all prime numbers before limit 

sieveSundaram(); 

  

# Express number as a sum of two primes 

findPrimes(4); 

findPrimes(38); 

findPrimes(100); 