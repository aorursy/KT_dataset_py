v_name = "muhammet ali"

v_surname = "kaya"

v_fullname = v_name + v_surname

print(v_fullname)
v_message = "helo world my name is"

v_name = "muhammet ali"

v_surname = "kaya"

v_fullname = v_message + " " + v_name + " " + v_surname

print(v_fullname)
print(v_fullname)

print()

v_var1 = "6"

v_var2 = "1"

v_varsum = v_var1 + v_var2

print(v_varsum)
v_fullname = "muhammet ali"

v_chr1 = v_fullname[3]

v_chr2 = v_fullname[2]

v_chr3 = v_fullname[4]

v_chr4 = v_fullname[6]

v_chr5 = v_fullname[7]

v_chrfull= v_chr1 + v_chr2 + v_chr3 + v_chr4 + v_chr5

print(v_chrfull)
# integer

v_num1 = 29

v_num2 = 32

v_numfull1 = v_num1 + v_num2

print(v_numfull1)
# float

v_num3 = 30.5

v_num4 = 30.5

v_numfull2 = v_num3 + v_num4

v_numfull = v_numfull1 + v_numfull2

print(v_numfull)

#lenght

_lenfull = len(v_fullname)

print("v_fullname :" , v_fullname , " and lenght is :" , _lenfull)
v_titleF = v_fullname.title()

print("v_fullname :", v_fullname ,  " and title is : " , v_titleF)

#Upper :

_upperF = v_fullname.upper()

print(_upperF)



#lower

v_lowerF = v_fullname.lower()

print("v_fullname : " , v_fullname , " Upper : " , v_upperF , " Lower : " , v_lowerF)
_2ch = v_fullname[10]

print(_2ch)
_num1 = 100

_num2 = 500

_num3 = 25

_num4 = 30

_sum1 = _num1 + _num2 

print("collection" , _sum1)
_sum2 = _sum1 - _num2 + _num1 + _sum1 

print(_sum2)
_sum3 = _num3 + 55

_sum4 = _num4 + 70

_sum5 = _sum1 + _sum2 + _sum3 + _sum4

print(_sum5)
_sum6 = _sum5 + 70

print(_sum6)
_fl1 = 25.5

_fl2 = 15.5

_s3 = v_fl1 + v_fl2



print(v_s3)
def f_SayHello():

    print("Hi. I love you")

    

def f_SayHello2():

    print("Hi. I am coming from istanbul")

    print("Good")

    

def f_sayHello3():

    print("hello world my name is muhammet  ali")

    print("welcome")
f_SayHello2()
f_sayHello3()
# Default Functions :

def f_getSchoolInfo(v_Name,v_StudentCount,v_City = "ISTANBUL"):

    print("Name : " , v_Name , " St Count : " , v_StudentCount 

          , " City : " , v_City)

f_getSchoolInfo("AAIHL" , 661)

f_getSchoolInfo("istanbul Fen" , 661 , "istanbul")
# Flexible Functions :



def f_Flex1(v_Name , *v_messages):

    print("Hi " , v_Name , " your first message is : " , v_messages[2])

f_Flex1("ali" , "Selamün aleyküm" , "Naber" , "İyisindir İnşallah")
def f_alan(kenar1,kenar2):

    print(kenar1*kenar2)

    

    

f_alan(8,9)

def f_alan2(kenar3,kenar4):

    print(kenar3*kenar4)



f_alan2(1,5)
def f_alan3 (kenar5,kenar6):

    print(kenar5*kenar6)

f_alan3(3,4)
def f_alan4 (kenar7,kenar8):

    print:(kenar7*kenar8)

f_alan4(10,5)