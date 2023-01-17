#1.1
def func1(number):
    newNumber = number + 1
    return newNumber

a = 5
print('a =',a)
print(func1(a))

#1.2
def func1(number):
    newNumber = number + 1
    return newNumber

a = 5
print('a =',a)
print(func1(a))
print('a =',a)
#1.3
def func1A(a):
    a = a + 1
    return a

a = 5
print('a =',a)
print(func1A(a))
print('a =',a)

#1.4
def func2(number1,number2):
    if number1>number2:
        return number1
    else:
        return number2

brian = 6
print(func2(3,5))
print(func2(5,3))
print(func2(-3,-5))
print(func2(3,brian))

#1.5
def func2A(number1,number2):
    if number1>number2:
        return number1
        print('First is larger')
    else:
        return number2
        print('Second is larger')

brian = 6
print(func2A(3,5))
print(func2A(5,3))
print(func2A(-3,-5))
print(func2A(3,brian))
#1.6
def func3(number):
    a = 6
    number = 7
    return 8

a = 5
print(a)
print(func3(a))
print(a)
print(number)

#1.7
def func4(a):
    print('in func4 a =',a)
    b = 100 + a
    d = 2 * a
    print('in func4 b =',b)
    print('in func4 d =',d)
    print('in func4 c =',c)
    return b + 10

a = 10
b = 15
c = 25
c = func4(b)
print('_______')
print('a =',a)
print('b =',b)
print('c =',c)
#1.8
def func5a(name):
    print('func5b(name)')
    return 'brian'

def func5b(name):
    print(func5a(name))
    return 'fred'

print(func5a('sue'))
print(func5b('betty'))

#1.9
def mysum(list,start,end):
    index = start
    answer = list[index]
    while index<end:
        index += 1
        answer += list[index]
    return answer

list1 = [1, 3, 5, 7, 9]
sum1 = mysum(list1,1,3)
sum2 = mysum(list1,0,4)
print(sum1)
print(sum2)
list2 = ["are ","you ","brian ","smith "]
sum3 = mysum(list2,0,2)
sum4 = mysum(list2,2,3)
print(sum3)
print(sum4)

#2.1
def func6(message):
    print(message)
    if len(message)>1:
        #Remove last character
        message = message[:-1]
        func6(message)

func6('Hello World')

#2.2
def func7(message):
    if len(message)>1:
        #Remove last character
        message = message[:-1]
        func7(message)
        print(message)

func7('Hello World')
# 3.1
def func8(mynum,mylist):
    mynum = 1000
    mylist[0] = 1000

a = 4
b = [1, 2, 3, 4, 5]
func8(a,b)
print('a=',a)
print('b=',b)

#3.2
def func9(mylist,a,b):
    temp = mylist[a]
    mylist[a] = mylist[b]
    mylist[b] = temp
    
list1 = [1000, 7, 23.6, 431]
print(list1)
func9(list1,0,1)
print(list1)
func9(list1,0,3)
print(list1)