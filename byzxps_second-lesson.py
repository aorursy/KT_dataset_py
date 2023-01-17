def b_SayGm():

    print("Hi good morning")

    

def b_SayGn():

    print("Hi good night")

    print("Nice")

    

b_SayGm()
b_SayGn()
def b_sayMessage(v_Message1):

    print(v_Message1 , " Ara bni")

    

def b_getFullName(v_FirstName , v_Surname , v_Age):

    print("Welcome " , v_FirstName , " " , v_Surname , " your age : " , v_Age)
b_sayMessage("Napyon ?")
b_getFullName("Beyza" , "Baştürk" ,14 )

def Square(b_Num1 , b_Num2 , b_Num3 , b_Num4 ):

    v_Sonuc = b_Num1 + b_Num2 + b_Num3 + b_Num4    



    print("Sonuç = " ," " , v_Sonuc)
Square(50,50,50,50)
# return function

def Square_2(b_Num1 , b_Num2 , b_Num3):

    v_Out = b_Num1/2 + b_Num2-2 + b_Num3*2

    print("Cogratulations")

    return v_Out
b_yaz =  Square_2(4,6,8)

print("Score is : " , b_yaz)
# Default Functions :

def b_GamerInfo(b_Name,b_age,b_Username = "New Player"):

    print("Name : " , b_Name , " Age : " , b_age

          , " User Name: " , b_Username)
b_GamerInfo("Alex", 21 )

b_GamerInfo("CrazyGamer_3418648" , 15 ,  "Merhabalar ..")
# Flexible Functions :



def b_GelenKutusu(v_Name , *v_messages):

    print("Look! " , v_Name , " U have New message From ur mother  : " , v_messages[2])
b_GelenKutusu("Abd-i Dulah-i MüQ", "Yarın Büyük Bir Deprem Olacak!" , "Bugün okulda birnin başı koptu la" , "Ekmek Alsana")
# Lambda Function :



b_result1 = lambda x : x/2

print("Result is : " , b_result1(1500))
def b_alan(kenar1,kenar2):

    print(kenar1*kenar2)

    
b_alan(50,20)