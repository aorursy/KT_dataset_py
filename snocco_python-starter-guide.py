import this
num0 = 5

num1 = 2

num2 = 4.0



print("The num0's type is: ", type(num0))

print("The num1's type is: ", type(num1))

print("The num2's type is: ", type(num2))
add = num0+num1

print('%d + %d = %d' % (num0, num1, add))
sub = num0-num1

print('%d - %d = %d' % (num0, num1, sub))
mul = num0*num1

print('%d * %d = %d' % (num0, num1, mul))
div = num0/num1

print('%d / %d = %f' % (num0, num1, div))
exp = num0**num1

print('%d ** %d = %d' % (num0,num1, exp))
iDiv = num0//num1

print('%d // %d = %f' % (num0, num1, iDiv))
module = num0%num1

print('%d mod %d = %f' % (num0, num1, module))
bignumber0 = 15948594859485948594589458497382732837289372893728937298372893728937289372893729837289462398

bignumber1 = 4636473647364736473467346374637463746374637463746374634736486578367238647384637846374863748657757557555757557



print(bignumber1*bignumber0)
import math 

print('These are the functions of the math module ...\n\n', dir(math))
import math



num1 = 65

num2 = 10



sqrt = math.sqrt(num1) #module's name like qualificator, dot & resource's name

esp = math.exp(num2)

loge = math.log(num1)

logd = math.log10(num1)



print("Sqrt : ", sqrt)

print("Exp  : ", esp)

print("Log e: ", loge)

print("Log10: ", logd)
from math import pi, sin, cos, sqrt #Another type of import



sqr = sqrt(num1)

sinpi = sin(pi) #Use only resource's name

cospi = cos(pi)

print("Sqrt  :", sqr)

print("Sin Pi:", sinpi)

print("Cos Pi:", cospi)
from math import * #All module's resources withouth using the qualificator
import random

print('These are the functions of the random module ...\n\n', dir(random))
random.seed(7) #for reproducibility
random.random() #random number
random.choice([1,2,3,4])
s1 = "I'm learning to use Python. Python is quite simple."

s2 = "Python"

s3 = "Spam"

c0 = s1[0]



test0 = s2 in s1

test1 = s3 in s1



print(c0)

print(s1.count("e"))

print(s1.isalpha())

print(s1.isdigit())

print(s1.lower())

print(s1.upper())

print(s1.capitalize())

print(test0)

print(test1)
s0 = "Hi, my name is Stefano "

s1 = "and i'm 26 years old."

print(s0+s1)
s1 = "Brian"

try:

    s1[0] = "C"

except TypeError:

    print("Strings are immutable!")
l = [] # l = list()



l1 = ["test0","test1"]

l2 = [3, 5, 6]



l3=l1+l2



print('List1: ',l1)

print('List2: ',l2)

print('List3: ',l3)

print('Len List3: ', len(l3))
s = "Monty Python's Life of Brian"

strList = list(s) 

print(strList)
v = 0

n = 10

l=[v] * n 



print(l)

print(v in l) 

print(v not in l) 
s = "name/surname/age"

sep = "/"

l=s.split(sep)

s=sep.join(l)



print(s)

print(sep)

print(l)

print(s)
M = [[1,2,3],

     [4,5,6],

     [7,8,9]]

print(M)

print(M[1])

print(M[2][2])
s = []

for x in range(10):

    if x%2 == 0:

        s.append(x**2)

print(s)
s = [x**2 for x in range(10) if x%2 == 0] #Only one row!

print(s)
d = {"Stefano":"Nocco", 

     "Kaggle":"Team",

     "Brian":"Nazareth"}

d
d.keys()
d.values()
d.items()
"Stefano" in d.keys()
a = ["a", "b", "c", "d", "e"]

b = [0,1,2,3,4]



mydict = dict(zip(a,b))

print(mydict)

print('-'*30)



mydict["f"] = {"ste":"nocco"} #new item

print(mydict)

print('-'*30)



v0 = mydict.get("f")

v1 = v0.get("ste")

print(v0)

print('-'*30)

print(v1)
d = {"id0":["Stefano", "Nocco", "Data Scientist"],

      "id1":["Brian", "Nazareth", "Data Engineer"]}
d.keys()
d.values()
a = ["a", "b", "c"]

b = [0,1,2]



d = {k:v for k,v in zip(a,b)} 

print(d)
t = (1,2,3,4,5,6)
type(t)
t[0]
t.index(4)
t.count(4)
t1 = (7,8,9)

t2 = t+t1

print(t2)
cond = True



if cond:

    print('Run the nested code')
spam = 5



if spam == 5:

    print('Hi Brian!')
eggs = 7

bacon = 5



if eggs < bacon:

    print('Release Brian!')

else:

    print('Crucify Brian!')
eggs = 7

bacon = 5



if eggs < bacon: print('Release Brian!')

else: print('Crucify Brian!')
counter = 0

if counter <= 10:

    print(counter)

    counter = counter + 1
counter = 0

while counter <= 10:

    print(counter)

    counter = counter + 1
#cond = True



#while cond:     

#    print('infinite loop !!!')
counter = 0

while True:

    print(counter)

    counter += 1

    if counter > 10:

        print ("Bye bye!")

        break
counter = 0

while counter < 10:

    print(counter)

    counter += 1     

    if counter == 3:

        print ('Skipped!')

        continue

   
counter = 10

print('Increasing ...')

for number in range(counter):

    print(number)
print('Decreasing ...')

for numero in range(10,0,-1):

    print(numero)
eggs = ['spam', 'ham', 'brian', 'stefano']

for egg in eggs:

    print(egg)
chars = 'I love Python'

for char in chars:

    print(char)