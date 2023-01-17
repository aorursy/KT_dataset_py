#0.for basic function



def f():

    x=2

    y=3

    return x+y

print(f())
#1.If the function has a value of x, the function uses this value.



x=7

def f():

    x=2

    y=3

    return x+y

print(x)

print(f()) 
#2.If the function does not have a value of x, the function uses the value x out.



x=7

def f():

    y=3

    return x//y,x/y

print(f())
#3.nested function (first we perform the function inside, then the function of the outside function)



def f():

    def g():

        x=2

        y=3

        return x+y

    return g()*2

print(f())
#4.one, two, or all of them in the function



def f(x,y=4,z=3):

    return x+y+z

print("1.",f(5))             #The value of x is replaced by 5 (y=4, z=3)

print("2.",f(5,2))           #Instead of the value of x, 5 is replaced with 2 instead of y (z=3).

print("3.",f(5,3,7))         #Instead of the value of x, 5, instead of the value of y, 3 is replaced by 7 instead of z.

#5.args()



def f(*args):

    for i in args:

        print(i)

f(1,3,5,7)

f(7)



#2.way

each=1

while(each<8):

    print("2.",each)

    each=each+2
#6. kwargs



def f(**kwargs):

    for a,b in kwargs.items():           #dictionary for; dictionary.items also used.

        print(a,":",b)

f(country="Norway",city=["Harstad","Stavanger","Bergen"],population=[24820,117157,271949])



#2.way:

dictionary={"country":"Norway","city":["Harstad","Stavanger","Bergen"],"population":[24820,117157,271949]}

for a,b in dictionary.items():

    print("2.",a,":",b)

    

#3.way:

ma_1=["country","city","population"]

ma_2=["Morway",["Harstad","Stavanger","Bergen"],[24820,117157,271949]]

zippi=zip(ma_1,ma_2)

print("3.",list(zippi))



#4.way

ma_1=["country","city","population"]

ma_2=["Morway",["Harstad","Stavanger","Bergen"],[24820,117157,271949]]

print("4.",ma_1[0],":",ma_2[0],ma_1[1],":",ma_2[1],ma_1[2],":",ma_2[2])
#7.lambda function



sqrt=lambda x:x**0.5                   #sqrt function: example: sqrt(64)=8, sqrt(x**2)=x

print("1.1",sqrt(9))



#2.way

print("1.2",sqrt(9))



#3.way

def f(x):

    return x**0.5

print("1.3",f(9))







example=lambda x,z,y:y//z+x

print("2.1",example(2,7,3))



#2.way

def f(x,z,y):

    return y//z+x

print("2.2",f(2,7,3))
#8.map function (Improved lambda function)



list_1=[1,3,5,7,9]

map_function=map(lambda q:q**3,list_1)   #Perform operations from right to left.ex:for 3=27 (3**3)

print(list(map_function))



#2.way

each=1

while(each<10):

    each**3

    print("2.",each)

    each=each+2
#9.zip



list_2=["name","surname","age","sex"]    #indexes    index1="name", index2="surname", index3="age", index4="sex"

list_3=["martin","aniston",22,"male"]    #values     value1="martin", value2="aniston", value3=22, value4="male"

zip_1=zip(list_2,list_3)                 #combine the first index of list_3 with the first value of list_4... 

print(list(zip_1))   

#10.list comprehension



list_4=[2,4,6,8]

list_5=[each*7//3/2 for each in list_4]   #Take the value on the left side of each value in the list_4.

print(list_5)   
