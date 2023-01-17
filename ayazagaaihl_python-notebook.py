print("Hello World")
v_message = "hello world"



print("Hi")
print(v_message)
v_name = "gazi"

v_surname = "erdogan"



v_fullname = v_name + v_surname

print(v_fullname)
v_fullname = v_name + " " + v_surname



print(v_fullname)
v_num1 = "100"

v_num2 = "200"

v_numSum1 = v_num1 + v_num2

print(v_numSum1)
#length

v_lenFull = len(v_fullname)

print("v_fullname : " ,v_fullname, " and lenght is : " ,v_lenFull)
v_titleF = v_fullname.title()

print("v_fullname :", v_fullname ,  " and title is : " , v_titleF)
#upper :

v_upperF = v_fullname.upper()



#lower

v_lowerF = v_fullname.lower()

print("v_fullname : " , v_fullname , " Upper : " , v_upperF , " Lower : " , v_lowerF)

v_2ch = v_fullname[11]

print(v_2ch)
v_num1 = 100

v_num2 = 200

v_sum1 = v_num1 + v_num2



print(v_sum1 , " and  type : " , type(v_sum1))
#it will get error

#v_sum2 = v_num1 + v_name

#print(v_sum2)
v_num1 = v_num1 + 50

v_num2 = v_num2 - 25.5

v_sum1 = v_num1 + v_num2



print(v_num1)
print("v_sum1 : ",v_sum1 , " type : " , type(v_sum1))
v_fl1 = 25.5

v_fl2 = 15.5

v_s3 = v_fl1 + v_fl2



print(v_s3 , type(v_s3))
def f_SayHello():

    print("Hi. I am coming from f_SayHello")

    

def f_SayHello2():

    print("Hi. I am coming from f_SayHello2")

    print("Good")

    

f_SayHello()
f_SayHello2()
def f_sayMessage(v_Message1):

    print(v_Message1 , " came from 'f_sayMessage'")

    

def f_getFullName(v_FirstName , v_Surname , v_Age):

    print("Welcome " , v_FirstName , " " , v_Surname , " your age : " , v_Age)

    
f_sayMessage("How are you ?")
f_getFullName("Gazi" , "ERDOĞAN" , 36)
def f_Calc1(f_Num1 , f_Num2 , f_Num3):

    v_Sonuc = f_Num1 + f_Num2 + f_Num3

    print("Sonuç = " ," " , v_Sonuc)

    
f_Calc1(100 , 250 , 50)
# return function

def f_Calc2(v_Num1 , v_Num2 , v_Num3):

    v_Out = v_Num1+v_Num2+v_Num3*2

    print("Hi from f_Calc2")

    return v_Out

    
v_gelen =  f_Calc2(1,2,3)

print("Score is : " , v_gelen)
# Default Functions :

def f_getSchoolInfo(v_Name,v_StudentCount,v_City = "ISTANBUL"):

    print("Name : " , v_Name , " St Count : " , v_StudentCount 

          , " City : " , v_City)
f_getSchoolInfo("AAIHL" , 521)

f_getSchoolInfo("Ankara Fen" , 521 , "ANKARA")
# Flexible Functions :



def f_Flex1(v_Name , *v_messages):

    print("Hi " , v_Name , " your first message is : " , v_messages[2])
f_Flex1("Gazi" , "Selam" , "Naber" , "İyisindir İnşallah")
# Lambda Function :



v_result1 = lambda x : x*3

print("Result is : " , v_result1(6))
def f_alan(kenar1,kenar2):

    print(kenar1*kenar2)

    

    
f_alan(3,5)