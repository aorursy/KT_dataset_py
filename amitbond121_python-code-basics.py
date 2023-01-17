#program to print passport status

def passport_check(passport_no):
    if(len(passport_no)==8):
        if(passport_no[0]>='A' or passport_no[0]<='Z'):
            status="VALID PASSPORT"
        else:
            status="xxx INVALID PASSPORT xxx"
            
    else:
        status="xxx INVALID PASSPORT xxx"
    return status

passport_status=passport_check('BKDFG845')
    
print (passport_status)


#program to print sqaure of a number

def find_sqaure(n):
    result=n*n
    return ("Square of this number is", result)

find_sqaure(11)  

#sum of first n numbers
'''Rules of global Keyword
The basic rules for global keyword in Python are:

When we create a variable inside a function, it’s local by default.
When we define a variable outside of a function, it’s global by default. You don’t have to use global keyword.
We use global keyword to read and write a global variable inside a function.
Use of global keyword outside a function has no effect'''

sum1=0

def sum_num(n):
    
    for i in range(1,n+1,1):
        global sum1 
        
        sum1 += i
         
    return (sum1)

sum_of_first=sum_num(10)
print(sum_of_first)
    
