print('my name is ahmet DURAN')
v_message = "my name is ahmet DURAN"



print("hi")

print(v_message)
v_name = "ahmet"



v_surname = "DURAN"

 



v_fullname = v_name + v_surname

print(v_fullname)
v_fullname = v_name + " " + v_surname





print(v_fullname)
v_num1 = "700"

v_num2 = "719"

v_numSum1 = v_num1 + v_num2

print(v_numSum1)
v_lenFull = len(v_fullname)

print("v_fullname : " ,v_fullname, " and lenght is : " ,v_lenFull)
v_titleF = v_fullname.title()

print("v_fullname :", v_fullname ,  " and title is : " , v_titleF)

v_upperF = v_fullname.upper()



#lower

v_lowerF = v_fullname.lower()

print("v_fullname : " , v_fullname , " Upper : " , v_upperF , " Lower : " , v_lowerF)
v_7ch = v_fullname[4]

print(v_7ch)
v_num1 = 800

v_num2 = 900

v_sum1 = v_num1 + v_num2



print(v_sum1 , " and  type : " , type(v_sum1))
v_num1 = v_num1 + 95

v_num2 = v_num2 - 55.5

v_sum1 = v_num1 + v_num2



print(v_num1)
print("v_sum1 : ",v_sum1 , " type : " , type(v_sum1))
v_f8 = 15.5

v_f5 = 7.4

v_s3 = v_f8 + v_f5



print(v_s3 , type(v_s3))
def f_SayHi():

    print("league of legends f_SayHi")

    

def f_SayHi2():

    print("Hi. I am coming from f_SayHi")

    print("Good")

    

f_SayHi()
f_SayHi2()
def f_sayMessage(v_Message1):

    print(v_Message1 , " my friends 'f_sayMessage'")

    

def f_getFullName(v_FirstName , v_Surname , v_Age):

    print("come " , v_FirstName , " " , v_Surname , " your age : " , v_Age)
f_sayMessage("bugün nasılsın ?")

f_getFullName("AHMET" , "DURAN" , 15)
def f_Calc1(f_Num1 , f_Num2 , f_Num3):

    v_Sonuc = f_Num1 + f_Num2 + f_Num3

    print("Sonuç = " ," " , v_Sonuc)
f_Calc1(930 , 620 , 310)
# return function

def f_Calc2(v_Num1 , v_Num2 , v_Num3 , v_Num4):

    v_Out = v_Num1+v_Num2+v_Num3+v_Num4*9

    print("Hello from f_Calc2")

    return v_Out
v_gelen =  f_Calc2(1,2,3,4)

print("Score is : " , v_gelen)
# Default Functions :

def f_getSchoolInfo(v_Name,v_StudentCount,v_City = "RUSİA"):

    print("Name : " , v_Name , " St Count : " , v_StudentCount 

          , " City : " , v_City)
f_getSchoolInfo("AAIHL" , 620)

f_getSchoolInfo("SİVAS ANADOLU" , 310 , "SİVAS")
# Flexible Functions :



def f_Flex1(v_Name , *v_messages):

    print("Hello " , v_Name , " your my friends Abdüssamed : " , v_messages[1])
f_Flex1("AHMET" , "DURAN" , "MERHABALAR" , "İSMİNİZ NEDİR ?")
# Lambda Function :



v_result1 = lambda x : x*8

print("Result is : " , v_result1(4))
def f_alan(kenar1,kenar2, kenar3, kenar4):

    print(kenar1*kenar2+kenar3/kenar4)
f_alan(20,40,60,5)
A_list1 = [5,10,15,20,25,30]

print(A_list1)

print("Type of 'A_list1' is : " , type(A_list1))
A_list1_4 = A_list1[5]

print(A_list1_4)

print("Type of 'A_list1_4' is : " , type(A_list1_4))
A_list2 = ["AHMET","MEHMET","DİLARA","CANSU","ABDÜS","SAMED","BAKİ"]

print(A_list2)

print("Type of 'A_list2' is : " , type(A_list1))