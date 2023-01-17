#Exercise 5

InSet=set(map(float,input('Please enter the numbers splitted by commas",": ').split(',')))
OutSet=set()

for A in InSet:
    if A%2==0:
        OutSet.add(A)
        
if OutSet:
    print('The largest even number is ',int(max(OutSet)),'.',sep='')
else:
    print('None of the numbers entered is an even.')
#Exercise 6

Input=int(input("Please enter a positive integer: "))

RangeMin=2
RangeMax=6+1
pwr=RangeMin
Combination=0


if Input>1:
    while pwr in range (RangeMin,RangeMax):
        root=round(Input**(1/pwr),10)
        if root%1==0:
            print(Input," equals ",int(root)," to the power of ",pwr,".",sep="")
            Combination+=1
            print("Answer combination: ",Combination)
        pwr+=1
    if Combination==0:
        print(Input," is not equal to any integer to the power from ",RangeMin," to ",RangeMax-1,".",sep="")

if Input==1:
    print(Input," equals any power of 1.",sep="")

if Input<1:
    print(Input," is not a positive integer.",sep="")
#Exercise 7

# Find the square root of an integer by bisection method
Input = float(input())

if Input==0:
    print('0 is cube root of 0')

x = abs(Input)
epsilon = 0.01
numGuesses = 0
low = 0.0
high = max (1.0, x)
ans = (high + low)/2.0
while abs(ans**3 - x) >= epsilon:
     print ('low =', low, 'high = ', high, 'ans =', ans)
     numGuesses += 1 
     if ans**3 < x:
          low = ans
     else:
          high = ans
     ans = (high + low)/2.0     
print ('numGuesses =', numGuesses)

if Input>0:
    print(ans, 'is close to cube root of',Input)
else:
    print(-ans, 'is close to cube root of',Input)