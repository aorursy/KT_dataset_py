# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
name="Coimer"

surname="Reis"

print("my name is {} {}".format(name,surname))

#Also you can simply change the order with using their indexes.

#default is like that:

# print("my name is {0} {1}".format(name,surname))

print("my name is {1} {0}".format(name,surname))

#Additionally, you can change the indexes

print("my name is {s} {n}".format(n=name,s= surname))
#Lets look another example;

operation=100/7

print(operation)

#We are gonna write operation in a string with format() method

print("the result is {op:7.3}".format(op=operation))

#"   14.3"= 3space+14+.+3 =totally 7 gap for our number

#I mean there are seven space for our number

#op:7.3 then what the is 3 in there? It mean show me 3 digit only

#So Python rounds the number after the comma. Lets do another example

wow=100/3

print(wow)

print("the result is {w:4.3}".format(w=wow)) #33.3

wow=100/3

print(wow)

print("the result is {w:4.7}".format(w=wow)) #33.33333 

wow=100/460

print(wow)

print("the result is {w:10.3}".format(w=wow))
#More basic method is f string feature

print(f"My name is {name} {surname}")

#You add only "f" in front of your string.
#You can change uppercase to lowercase or can do the opposite

text=" How are you doing? "

print(text.lower()) # how are you doing?

print(text.upper()) # HOW ARE YOU DOING?

#First char of all the words are capitalized

print(text.title()) # How Are You Doing?

#Leading and trailing characters removed

#If you wanna remove blanks only left side= lstrip()

#                           only right side= rstrip()

print(text.strip())

#We can remove a word or something like that with strip

print(text.lstrip(" Ho"))

print(text.rstrip("g? "))

print(text.strip(" H.ing? "))





#List of strings after breaking the given string by the specified separator

#Seperator can be letter or simply "."

print(text.split())

text2="Welcome. Are you completely new to programming. If not then we presume you will be looking for information about why and how to get started with Python. Fortunately an experienced programmer in any programming language (whatever it may be) can pick up Python very quickly. It's also easy for beginners to use and learn, so jump in!"

array=text2.split(".")

print(array[1])

#Or you can split every word

word=text.split(" ")

print(word[3]) #you

#If you re looking for a word in text ,you can easyly find its starting index

hey="What the fuck are you doing?"

hey.find("fuck") #9

#If you wanna start counting from the right

#print(hey.rfind("fuck")) 

hey.startswith("Wha") #True

hey.endswith("k") #False

hey.replace("fuck","f**k") #'What the f**k are you doing?'



cha="İnsanın çalıştıkça çalışası geliyor."

cha=cha.replace("ç","c")

cha=cha.replace("ş","s")

cha=cha.replace("ı","i")

cha

#or..

cha=cha.replace("ç","c").replace("ş","s").replace("ı","i").replace("ö","o")

cha



text=text.strip()

print(text.replace(" ",""))

print(text.replace(" ","",2))

print(text.replace(" ","",1))

net="https://www.w3schools.com/python/"

print(net.isalpha()) #there are digits in it

print("hi".isalpha()) #true

print("342".isdigit()) #true
apple="Your time is limited, so don't waste it living someone else's life."

print(apple.count("i"))

#You can search in a specific area btwn 0 ,15

print(apple.count("i",0,15))

#We find 3 "i" until 15.index
#Takes my string into a container and centers it

#For example i assume my container has 50 spaces

txt="Python for everyone"

print(txt.center(50))



#We can  fill in the blanks with chars

print(txt.center(50,"-"))

#Justify expression on right or left side

print(txt.ljust(50,"-"))

print(txt.rjust(50,"-"))
myList=[1,2,3]

print(myList)

#List containers that can store many kind of values.

secondList=[1,"there",False,0.99]

print(secondList)

#We can easily combine lists with "+"

list1=["one","two","three"]

list2=[4,5,6]

x=list1+list2

print(x)

len(x)

print(x[2])

#You can add lists without disturbing the list structure

user1=["john",34]

user2=["isabel",23]

users=[user1,user2]

print(users)

print(users[0][1])
myLst=["bmw","mercedes","opel","volvo","mazda"]

len(myLst)#5

print(myLst[0])#bmw

print(myLst[-1]) #mazda

#### lists can be changed ####

myLst[-1]=myLst[0]

myLst[0]="mazda"

myLst #['mazda', 'mercedes', 'opel', 'volvo', 'bmw']



result="opel" in myLst

print(result)



myLst[-2:]=["toyota","renault"]

myLst



result=myLst+["audi","nissan"]

result #['mazda', 'mercedes', 'opel', 'toyota', 'renault', 'audi', 'nissan']



del myLst[-1]

myLst



print(myLst[::-1])

numbers=[1,10,234,2,342,2,2,2,56,2]

letters=["a","h","d","k"];



## min() max()

print(min(numbers)) #1

print(min(letters)) #a

print(max(letters)) #k

#Lists are changeable

numbers[3]=5

print(numbers) #[1, 10, 234, 5, 342, 2, 2, 2, 56, 2]



## append()

#We can add new members with append() method

numbers.append(49)

print(numbers) #[1, 10, 234, 5, 342, 2, 2, 2, 56, 2, 49]



## insert()

#Or you can choice where do you wanna add new member with insert method

#It says add 9999 where? before the 3.index

numbers.insert(3,9999)

print(numbers)#[1, 10, 234, 9999, 5, 342, 2, 2, 2, 56, 2, 49]

numbers.insert(-1,100000)

print(numbers) #[1, 10, 234, 9999, 5, 342, 2, 2, 2, 56, 2, 100000, 49]



##pop()

numbers.pop()

print(numbers) #[1, 10, 234, 9999, 5, 342, 2, 2, 2, 56, 2, 100000]

#You can decide which memeber of list will be popped

numbers.pop(0) #[10, 234, 9999, 5, 342, 2, 2, 2, 56, 2, 100000]

print(numbers)

numbers.pop(3) #[10, 234, 9999, 342, 2, 2, 2, 56, 2, 100000]

print(numbers)

numbers.pop(-1) #numbers.pop() same

print(numbers)



##remove()

numbers.remove(9999)

print(numbers) #[10, 234, 342, 2, 2, 2, 56, 2]



##sort()

numbers.sort()

print(numbers) #[2, 2, 2, 2, 10, 56, 234, 342]

letters.sort() #alphabetic

print(letters) #['a', 'd', 'h', 'k']



##reverse()

numbers.reverse()

print(numbers) #[342, 234, 56, 10, 2, 2, 2, 2]



##count()

print(numbers.count(2)) #4



##clear()

numbers.clear()

print(numbers) #[]

mytuple=1,"two",3

print(type(mytuple))



atuple=(1,2,"dsgf")

print(type(atuple))



mylist=[1,2,3]

print(type(mylist))



print(len(mytuple)) #3

print(tuple[1]) #two



# !!!!tuple object doesnt support item assignment

#  mytuple[1]="hi"

#  print(mytuple)



#You can change the whole tuple but you cant change one of the member of tuple

mytuple=(3,45,67,88)

print(mytuple)



#addition

wow=("ella","daniel","edd")+mytuple

print(wow) #('ella', 'daniel', 'edd', 3, 45, 67, 88)

#It works key-value logic

city=["london","DC","mumbai"]

code=[44,1,91]

print(code[city.index("DC")]) #1



#Lets learn how can we define dictionaries

#dict={"key":"value",..}

code={"london":44,"DC":1,"mumbai":91}

print(code["mumbai"]) #91



#We can simply add new key&values

code["madrid"]=34

print(code) #{'london': 44, 'DC': 1, 'mumbai': 91, 'madrid': 34}



#We can change the values

code["DC"]="+1"

print(code)



#an Example

users={

    "stephan":{

        "age":23,

        "email":"stephan56@gmail.com",

        "city":"London",

        "phone":439712353,

        "roles":["user"]

    },

    "aissa":{

        "age":43,

        "email":"aissa_324@gmail.com",

        "city":"Delhi",

        "phone":329473975,

        "roles":["admin","user"]

    }

    

}



print(users["stephan"])

#{'age': 23, 'email': 'stephan56@gmail.com', 'city': 'London', 'phone': 439712353}

print(users["stephan"]["city"]) #London



print(users["aissa"]["roles"][0]) #admin



fruits={"orange","banana","cherry"}

#print(fruits[0]) can not be indexed



for a in fruits:

    print(a)



fruits.add("apple")

fruits #{'apple', 'banana', 'cherry', 'orange'}



fruits.update(["mango","grape"])

fruits #{'apple', 'banana', 'cherry', 'grape', 'mango', 'orange'}



#If you trying to add same value inside of set,it wont allow that

fruits.update(["apple"])

fruits #{'apple', 'banana', 'cherry', 'grape', 'mango', 'orange'}

#Same set



mylst=[1,2,3,4,3,3,1]

print(set(mylst)) #{1, 2, 3, 4}

#repetitive elements are removed from the list



fruits.remove("banana") #or discard()

fruits #{'apple', 'cherry', 'grape', 'mango', 'orange'}
'''

x=4

y="sdf"

z=324.3

'''

x,y,z=4,"sdf",324.3

print(x,y,z)   # 4 sdf 324.3



x,y=y,x

print(x,y,z)   # sdf 4 324.3



y+=5 #y=y+5 #9

y**=2 #y=y**2 #9*9

y #81



values=4,3,2

print(type(values))

x,y,z=values

print(x,y,z)

#But their lenght will be match each other

''' #TRY IT

values=4,3,2,7

print(type(values))

x,y,z=values

print(x,y,z) #ERROR

'''

#TRY IT

values=4,3,2,7,45,34,12

print(type(values))

x,y,*z=values

print(x,y,z) #z will have a list z=[2, 7, 45, 34, 12]

print(z[2]) #45



q,*w,e=2,34,23,56,78,89,1

print(w) #q=2 w=[34, 23, 56, 78, 89] e=1



a,s,d,f=1,1,2,3

print(a==s)  #True

print(a==d)  #False

print(a!=s)  #False

print(a>=d)  #False

print(a<d)  #True

print(True+False+56) #57
# WHICH ONE IS GREATER?

a=int(input("First one: "))

b=int(input("Second one: "))



if(a==b):

    print(f"{a} equal to {b}")

elif(a<b):

    print(f"{b} is greater than {a}")

else:

    print(f"{a} is greater than {b}")
x=7

result=5<x<10

print(result) #true



#and

result=x>5 and x<10

print(result) #true



#or

user1="elsa"

admin="tresa"

a,b=45,56



print((admin=="tresa") or (user1=="asd")) #True

print(a<34 or b>12)#True

print((a%2==0) or b<10) #False



#not

print(not(a==23)) #True a is not equal 23 so result is false

#but if you write infront of the station "not" it will reverse 



#### Identity Operator : is

x=y=[1,2,3]

z=[4,5,6]

w=[1,2,3]



print(x==y) #True

print(x is y) #True

print("*"*10)



print(x==z) #False

print(x is z) #False

print("*"*10)



print(x==w) #True

print(x is w) #False ?WHY? It looks ,are they sharing the same memory location

####  Membership Operator : in

x=["fears","big","and","small","are","universial","fact"]

print("are" in x) #True



print("small" not in x) #False
a = 200

b = 33

if b > a:

    print("b is greater than a")

elif a == b:

    print("a and b are equal")

else:

    print("a is greater than b")
# We ll add
def sayHi(name):

    print("Hi " +name)

    

sayHi("Stephan")
def sayHi(name):

    return ("Hi " +name)

    

msg=sayHi("ellise")

print(msg)
def add(num1,num2):

    return num1+num2

total=add(2,3)

print(total)
year=2020

def HowOld(birth):

    return year-birth



#print(HowOld(1986))



#Retirement Calculator



def RetCalc(birth,name):

    '''

    DOCSTRING: Number of years required for retirement.

    INPUT: Birth year,name

    OUTPUT: year

    '''

    age=HowOld(birth)

    ret=65-age

    return ret



RetCalc(1987,"Allice")

print(help(RetCalc))
#PASS BY VALUE

def ChangeName(n):

    n="Ada"

name="Lovelace"



ChangeName(name)

print(name)



#PASS BY REFERENCE

#updating info at address

def ChangeCity(n):

    n[0]="Istanbul"

    n[1]="Buhara"

cities=["London","Birmingham"]



ChangeCity(cities)

cities



def add(a,s,d=0):

    return sum((a,s,d))

#You can add 3 component or 2

add(1,2)



#we have a better solution about this. We add *params in function

#(you can change the name params, like *numbers) try it

#and you can add lots of number as you want

def add2(*params):

    return sum((params))



print(add2(1,2,3,3,5))

print(add2(2,3))



'''

or

************************

def add2(*numbers):

    sum=0

    for n in params:

        sum=sum+n

    return sum

************************  

'''

#We can use it in dictionary



def displayUser(**p):

    for key,value in p.items():

        print(f"{key} is {value}")

    

displayUser(name="qasa")

displayUser(name="wasa",age=34)

displayUser(name="hehoy",age=45,mail="hehoy98@gmail.com")
def myFunc(a,b,*args,**kwargs):

    print(a)

    print(b)

    print(args)

    print(kwargs)

    

myFunc(10,20,30,40,50,60,70,key1="value1",key2="value2")
class Animal(object):

    name="dog"

    age=2

    #if you write def in class it is a method



    def getAge(self):

        return self.age

    

a1=Animal()

a1_age=a1.getAge()

print(a1_age) #2
class Animal(object):

    

    #if you write def in class it is a method

    #select ve f9 

    def __init__(self,name,age):

        self.name=name

        self.age=age

        

    def getAge(self):

        return self.age

    def getName(self):

        print (self.name)

    

a1=Animal("dog",5)

a1_age=a1.getAge()

print(a1_age) #5

a2=Animal("cat",3)

a2.getName() #cat
class Calc(object):

    "calculator"

    

    

    #init method

    def __init__(self,a,b):

        "initialize values"

        self.value1=a

        self.value2=b

        #attribute

        

    def add(self):

        "addition a+b=result"

        return self.value1+self.value2



    def islem(self):

        "multiplication a*b=result->return result"

        return self.value1*self.value2

    def div(self):

        return self.value1/self.value2





v1=int(input("first value"))

v2=int(input("second value"))

c1=Calc(v1,v2)

print(c1.add())

print(c1.islem()) 

print(c1.div())
#herhangi bir nesnenin motodlarını verilerini ve değişkenlerini diğer nes

#diğer nesnelerden saklayarak ve bunlara erişimi sınırlandırarak yalnış kullanımdan koruma konsepti



class BankAccount(object):

    def __init__(self,name,money,address):

        self.name=name #global

        self.money=money

        self.address=address

        

p1=BankAccount("messi",10000,"barcelona")

p2=BankAccount("neymar",5000,"paris")



p1.money+=p2.money

#We can axess the bank accounts

#so we can also change the amount of money

#it is wrong logic so encapsulation is coming here

class BankAccount(object):

    def __init__(self,name,money,address):

        self.name=name #global

        self.__money=money #private

        self.address=address

        

    def getMoney(self):

        return self.__money

    def setMoney(self,amount):

        self.__money=amount

        

    def __increase(self):

        self.__money=self.__money+500

        

p1=BankAccount("messi",10000,"barcelona")

p2=BankAccount("neymar",5000,"paris")



#    p1.__money

#We couldnt axess the money variable

#AttributeError: 'BankAccount' object has no attribute '__money'



#soooo; we cant observe or change this private value,





print("get method: ",p1.getMoney()) #10000



p1.__increase()

print("after raise: ",p1.getMoney())



#!!!AttributeError: 'BankAccount' object has no attribute '__increase'

#you cannot reach this method becux of encapsulation

#you can use __increaseonly in the class
#yine bir class yazarken bazı seyleri onceki bşr klass tan inheritance ediyoruz

##boylece herseyi bastan yazmak zorunda kalmayız





#parent

class Animal:

    def __init__(self):

        print("animal is created")

    def toString(self):

        print("animal")

    def walk(self):

        print("animal wallk")



class Monkey(Animal):

    def __init__(self):

        super().__init__()

        #use init of parent class

        print("monkey is created")

    def toString(self):

        print("monkey")

    def climb(self):

        print("monkey can climb")

        

m1=Monkey()

#animal is created

#monkey is created



m1.toString() #monkey

m1.walk() #animal wallk
class Website(object):

    def __init__(self,name):

        self.name=name

    def loginInfo(self):

        print("Name: "+self.name)

class WebA(Website):

    def __init__(self,name,ids):

        Website.__init__(self,name)

        #or u can initialize that method with using its name

        #Website.__init__(self)

        self.ids=ids

        

    def loginInfo(self):

        print("Name: "+self.name+"  Id: "+self.ids)

        

class WebB(Website):

    def __init__(self,name,email):

        Website.__init__(self,name)

        self.email=email

    def loginInfo(self):

        print("Name: "+self.name+"  Id: "+self.email)

        

        

p1=Website("ali")

p1.loginInfo()

p2=WebA("ece","9757")

p2.loginInfo()



p3=WebB("akri","akri@lik.com")

p3.loginInfo()
#soyut class lar

#super class:parent,sub class:child

#super classlar, sub class lar için şablon görevi görür ve kullanılacak metodları tutarlar



#Burada super class lar instantiate edilemez

# yani a=Animal() diyemezsin



class Animal: #super class

    pass

class Bird(Animal):

    pass



a=Animal #olmaz! Animal class ından obje üretilemez
from abc import ABC ,abstractmethod

class Animal(ABC): 

    @abstractmethod

    def walk(self): pass



    @abstractmethod

    def run(self): pass



class Bird(Animal):

    #child class da implement etmekzorundasın bu walk ve run ı

    def __init__(self):

        print("bird")

    def walk(self): 

        print("WALKK")



   

    def run(self): 

        print("RUNN")

#a=Animal()

#TypeError: Can't instantiate abstract class Animal with abstract methods run, walk





b1=Bird()

#TypeError: Can't instantiate abstract class Bird with abstract methods run, walk

# If you dont write walk or run method

b1.run()







#1#Animalla ilgil obje yaratamam

#2#Super clasta kullandıgım metodları tekrar yazmak zorundayım
class Animal:

    def call(self):

        print("animal")

        

class Monkey(Animal):

    def call(self):

        print("monkey")

        

a1=Animal()

a1.call()



m1=Monkey()

m1.call()



#animal

#monkey monkey calls overriding method
#Çok biçimlilik

#Super class tan sub class a inheritance yoluyla aktarılan

#ama subclass ta farklı bir şekilde kullanılan metodlar varsa biz buna polymorphism diyoruz

class Employee:

    def raisee(self):

        raise_rate=0.1

        return 100+100*raise_rate

    

class CompEng(Employee):

    def raisee(self):

        raise_rate=0.2

        return 100+100*raise_rate

class EEE(Employee):

    def raisee(self):

        raise_rate=0.3

        return 100+100*raise_rate

e1=Employee()

e1.raisee()

ce=CompEng()

ce.raisee()

eee=EEE()

eee.raisee()
#abstract base class abc

from abc import ABC ,abstractmethod

class Shapes(ABC): #Abstract class

    """

    Shape=super class/abstract class

    """

    @abstractmethod

    def area(self,a,b):

        pass

    @abstractmethod

    def perimeter(self):

        pass        

    #overriding and polymorphism

    def toString(self):

        pass        

    

class Circle(Shapes):

    "circle class"

    PI=3.14

    #constant variable(buyuk harfle yazdık)

    def __init__(self,r):

        self.__r=r

    

    def area(self):

        areas=self.PI*self.__r*self.__r

        return areas

    def perimeter(self):

        per=2*self.PI*self.__r

        return per

    def toString(self):

        return "Circle"

        

class Square(Shapes):

    "sub class"

    def __init__(self,edge):

        self.__edge=edge #encapsulation #private attribute

    def area(self):

        result=self.__edge**2

        return result

    def perimeter(self):

        result=self.__edge*4

        return result

    def toString(self):

        return "Square"





c=Circle(4)

a=c.toString()



print(a," Area: ", c.area())

print(a," Perimeter ",c.perimeter())





print("*****************************")

s=Square(5)

b=s.toString()



print(b," Area: ", s.area())

print(b," Perimeter ",s.perimeter())


