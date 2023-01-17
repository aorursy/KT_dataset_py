### creating an empty list 
lst = [] 
  
# number of elemetns as input 
n = int(input("Enter number of elements : ")) 
  
# iterating till the range 
for i in range(0, n): 
    ele = int(input()) 
  
    lst.append(ele) # adding the element 
      
print(lst) 
import math  
  
# print the square root of  0  
print(math.sqrt(0))  
  
# print the square root of 4 
print(math.sqrt(4))  
  
# print the square root of 3.5 
print(math.sqrt(3.5))  

## creating an empty list 
lst = [] 
sqrrt=[] 
  
# number of elemetns as input 
n = int(input("Enter number of elements : ")) 
  
# iterating till the range 
for i in range(0, n): 
    ele = int(input()) 
    sqt = math.sqrt(ele)
    lst.append(ele) # adding the element 
    sqrrt.append(sqt)
print(lst) 
print(sqrrt)