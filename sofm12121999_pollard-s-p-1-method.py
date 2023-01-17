# Python code for Pollard p-1  
# factorization Method 
   
# importing "math" for  
# calculating GCD 
import math 
   
# importing "sympy" for  
# checking prime 
import sympy 




dem = 0
      
# function to generate  
# prime factors 
def pollard(n): 
    global dem
    # defining base 
    a = 2
   
    # defining exponent 
    i = 2
   
    # iterate till a prime factor is obtained 
    while(True): 
   
        # recomputing a as required 
        a = (a**i) % n 
   
        # finding gcd of a-1 and n 
        # using math function 
        d = math.gcd((a-1), n) 
        # check if factor obtained
        dem = dem +1
        if (d > 1): 
            
            #return the factor 
           
            return d 
            break
   
        # else increase exponent by one  
        # for next round 
        i += 1
  
# Driver code 
n = 71977 
   
# temporarily storing n 
num = n 
   
# list for storing prime factors 
ans = [] 
# iterated till all prime factors 
# are obtained 
while(True): 
   
    # function call 
    d = pollard(num) 
   
    # add obtained factor to list 
    ans.append(d) 
   
    # reduce n 
    r = int(num/d) 
       
    # check for prime using sympy 
    if(sympy.isprime(r)): 
   
        # both prime factors obtained 
        ans.append(r) 
   
        break
   
    # reduced n is not prime, so repeat 
    else: 
   
        num = r 
  
# print the result 
print("Prime factors of", n, "are", *ans) 
print("Số bước để thực hiện thuật toán là",dem)