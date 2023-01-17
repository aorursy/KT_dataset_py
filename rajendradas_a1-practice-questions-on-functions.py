def inum_mul():
    inum = int(input('Enter a number:'))
    for icount in range(1,11,1):
        print(inum,"x",icount,"=",inum*icount)

# Calling the defined function
inum_mul()
import sympy as s

def twinprime_program(num_start, num_end):
    for tpp in range(num_start, num_end):
        tpp2 = tpp + 2
        if (s.isprime(tpp) and s.isprime(tpp2)):
            print("Twin Primes are: ",tpp, "and", tpp2)
                  
twinprime_program(2,1000)
import math as m

def prime_fac(num):
    while num % 2 == 0:
        print(2)
        num = num / 2
    
    for pfi in range(3,int(m.sqrt(num))+1,2):
        while num % pfi == 0:
            print(pfi)
            num = num / pfi
            
    if num > 2:
        print(int(num))
        
prime_fac(260)
import math as m

def perm_comb_formula(n,r):
    pvalue = int(m.factorial(n) / m.factorial(n-r))
    cvalue = int(pvalue/m.factorial(r))
    print("The permutations of {} objects taken {} at a time is: {}".format(n,r,pvalue))
    print("The combinations of {} objects taken {} at a time is: {}".format(n,r,cvalue))
    
perm_comb_formula(17,3)
out_lst = []

def decimal2binary(num):
    quo, rem = divmod(num,2)
    out_lst.append(rem)
    while quo != 0:
        quo, rem = divmod(quo,2)
        out_lst.append(rem)
    out_lst.reverse()
    print(*out_lst,sep='')

decimal2binary(13)
def cubesum():
    inum = input("Enter a number: ")
    inum_lst = list(inum)
    sum_num = 0
    for jnum in range(len(inum_lst)):
        sum_num = sum_num + int(inum_lst[jnum])**3
    return int(inum), sum_num

def isArmstrong(n, s):
    if (n == s):
        print("Yes - {} is an Armstrong Number".format(n))
    else:
        print("False - {} is not an Armstrong Number".format(n))
        
def PrintArmstrong():
    n,s = cubesum()
    isArmstrong(n, s)
    
PrintArmstrong()
    
def prodDigits(pdnum):
    tmpdig=pdnum
    while 1:
        prddig=1
        while tmpdig!=0:
            remdig=tmpdig%10
            prddig=prddig*remdig
            tmpdig=int(tmpdig/10)
        
        if prddig<10:
            print("Product of digits of the number {} is {}".format(inum,prddig))
            break
        
        tmpdig=prddig

inum=int(input("Enter a number: "))

prodDigits(inum)
def prodDigits(pdnum):
    tmpdig=pdnum
    cntdig=0
    while 1:
        prddig=1
        cntdig = cntdig + 1
        while tmpdig!=0:
            remdig=tmpdig%10
            prddig=prddig*remdig
            tmpdig=int(tmpdig/10)
        
        if prddig<10:
            return cntdig, prddig
            #print("Product of digits of the number {} is {}".format(inum,prddig))
            break
        
        tmpdig=prddig

def MDR(mnum):
    mnum1, mnum2 = prodDigits(mnum)
    print("Multiplicative Digital Root of the number {} is: {}".format(mnum,mnum2))

def MPersistence(mnum):
    mp1, mp2 = prodDigits(mnum)
    print("Multiplicative Presistance of the number {} is: {}".format(mnum,mp1))

#Ask user to input a number:
mnum=int(input("Enter a number: "))

#calling the Multiplicative Digital Root function:
MDR(mnum)

#calling the Multiplicative Presistance function:
MPersistence(mnum)
def sumPdivisors(inums):
    spd_lst = []
    for i in range(1,inums):
        if inums%i == 0:
            spd_lst.append(i)
    print('Proper Divisors of number {} are:'.format(inums))
    print(*spd_lst,sep=',')

inums = int(input('Enter a number: '))
sumPdivisors(inums)
def perfectNumber(pnum_range):
    spn_lst = []
    sum_lst = []
    div_lst = []
    final_lst = []
    
    for i in range(pnum_range[0],pnum_range[1]+1):
        for j in range(1,i+1):
            if i%j == 0 and i != j:
                div_lst.append(j)
        spn_lst.append(i)    
        sum_lst.append(sum(div_lst))
        div_lst = []
    
    for x in range(len(spn_lst)):
        if spn_lst[x] == sum_lst[x]:
            final_lst.append(spn_lst[x])
    
    if len(final_lst) == 0:
        print("NO numbers in the range of {} to {} are perfect !".format(pnum_range[0],pnum_range[1]))
    else:
        print("Perfect Numbers in the range of {} to {} are - ".format(pnum_range[0],pnum_range[1]),final_lst)

pnum_range = list(map(int,input("Enter the number range with a space in between like 1 10 ->   ").strip().split()))[:2]
perfectNumber(pnum_range)
def amicableNumber(anum_range):
    spn_lst = []
    sum_lst = []
    div_lst = []
    final_lst = []
    
    for i in range(anum_range[0],anum_range[1]+1):
        for j in range(1,i+1):
            if i%j == 0 and i != j:
                div_lst.append(j)
        spn_lst.append(i)    
        sum_lst.append(sum(div_lst))
        div_lst = []
    
    for x in range(len(spn_lst)):
        for x1 in range(x):
            if spn_lst[x] == sum_lst[x1] and spn_lst[x1] == sum_lst[x]:
                final_lst.append([spn_lst[x],spn_lst[x1]])
    
    if len(final_lst) == 0:
        print("NO numbers in the range of {} to {} are amicable !".format(anum_range[0],anum_range[1]))
    else:
        print("Amicable Numbers in the range of {} to {} are - ".format(anum_range[0],anum_range[1]),final_lst)

anum_range = list(map(int,input("Enter the number range with a space in between like 1 10 ->   ").strip().split()))[:2]
amicableNumber(anum_range)
import numpy as np

def filterOddNumbers(fon_range):
    odd_lst = list(filter(lambda xnum : (xnum%2 != 0),np.arange(fon_range[0],fon_range[1]+1,1)))
    print("Odd Number(s) in the range {} to {} are - ".format(fon_range[0],fon_range[1]),odd_lst)

fon_range = list(map(int,input("Enter the number range with a space in between like 1 10 -> ").strip().split()))[:2]
filterOddNumbers(fon_range)
import numpy as np

def mapCubeElements(mce_range):
    mce_lst = list(map(lambda mce_num : (mce_num**3),np.arange(mce_range[0],mce_range[1]+1,1)))
    print("Cubes of element(s) in the range {} to {} are - ".format(mce_range[0],mce_range[1]),mce_lst)

mce_range = list(map(int,input("Enter the number range with a space in between like 1 10 -> ").strip().split()))[:2]
mapCubeElements(mce_range)
import numpy as np

def CubeEvenNumbers(cen_range):
    even_lst = list(filter(lambda ev_num : (ev_num%2 == 0),np.arange(cen_range[0],cen_range[1]+1,1)))
    cen_lst = list(map(lambda mce_num : (mce_num**3),even_lst))
    print("Cubes of even element(s) in the range {} to {} are - ".format(cen_range[0],cen_range[1]),cen_lst)

cen_range = list(map(int,input("Enter the number range with a space in between like 1 10 -> ").strip().split()))[:2]
CubeEvenNumbers(cen_range)