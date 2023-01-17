print("hello notebook")
v_message="hello world"



print("Hola")
print(v_message)
v_name="rabianur"

v_surname="softa"



v_fullname=v_name + v_surname

print(v_fullname)
v_fullname=v_name + " " + v_surname

print(v_fullname)
v_num1="200"

v_num2="300"

v_numSum1=v_num1+v_num2

print(v_numSum1)
#lenth

v_lenFull=len(v_fullname)

print("v_fullname :",v_fullname ,"and lenth is :" , v_lenFull)
v_titleF=v_fullname.title()

print("v_fullname:",v_fullname, "and tittle is ",v_titleF)

#upper :

v_upperF = v_fullname.upper()



#lowe : 

v_lowerF = v_fullname.lower()

print("v_fullname: " , v_fullname , "Upper : " ,v_upperF , "Lower :" , v_lowerF ,)
v_2ch =v_fullname[7]

print(v_2ch)
v_num1=200

v_num2=300

v_sum1=v_num1 + v_num2



print(v_sum1 , "and type is :" ,type(v_sum1))
v_num1=v_num1 + 100

v_num2=v_num2 - 40.2

v_sum1=v_num1 + v_num2



print(v_num1)
print("v_sum1 : " , v_sum1 , "type :" , type(v_sum1))
v_fl1 =18.2

v_fl2 =82.2

v_p3  =v_fl1 + v_fl2



print(v_p3 , type(v_p3))
def f_sayHOLA():

    print("Hola,Everything's good here")

    

def f_sayHOLA2():

    print("Hey!Everyting's good here too")

    print("legandary")

    



f_sayHOLA()
f_sayHOLA2()
def f_sayMESSAGE(v_Message):

    print(v_Message  , "Im coming from infinity")

    

def f_GetFULLname(v_firstname , v_surname , v_team , v_age ):

    print("WELCOME" , v_firstname  , " " , v_surname ,  "your favorite team is :" , v_team , "and your age is : " ,  v_age)

    

    
f_sayMESSAGE("ARE YOU OKAY ")

f_GetFULLname("RABİANUR" , "SOFTA" , "BESİKTAS" , "16")
def f_SQUARE(v_num1 , v_num2):

    v_RESULT = v_num1 + v_num2

    print("RESULT = " , " " , v_RESULT)

    
f_SQUARE(5 , 5 )
#return function

def f_SQUARE2(v_num1 , v_num2):

    v_OUT = v_num1+v_num2*2

    print("HOLA! come from f_SQUARE2 ")

    

    return v_OUT
v_come = f_SQUARE2(4 , 4 )

print("puan is : " , v_come)
#Default Functıons : 

def f_TEAMINFO(v_name , v_country , v_yearoffoundatıon , v_city ="ISTANBUL"):

    print("name : " , v_name ,  "country : " , v_country , "year of fd :" , v_yearoffoundatıon , "city : " , v_city)

    

f_TEAMINFO("BESİKTAS" , "TURKEY" , 1903 )

f_TEAMINFO("BAYERN MUNIH"  , "GERMANY" , 1900 , "MUNIH")
#Flexıble Functıons :

def f_flex1(v_team , *v_slogan):

    print("HI" , v_team , "your slogan is :" , v_slogan[3])

    
f_flex1("LIVERPOOL" , "YOU'LL" , "NEVER" , "WALK" , "ALONE")
#lambda functıon :



v_RESULT = lambda x : x*7

print("RESULT IS : " , v_RESULT(98))
def f_aritmeticmean(r1,r2,rn , n):

    print(r1 + r2 + rn / n)
f_aritmeticmean(4 , 8 , 12 ,3)
v_numbers = [100,200,300,400,500]

print(v_numbers)

print("Type of v_numbers is : " ,type(v_numbers) )
v_choose = v_numbers[4]

print(v_choose)

print("Type of v_choose is : ", type (v_choose))
v_city = ["berlin","madrid","roma","barcelona","toronto","oslo","moskow"]

print(v_city)

print("Type of v_city is : ", type(v_city))

v_choose2 =v_city[5]

print(v_choose2)

print("Type of v_choose2 is : " , type(v_choose2))
v_citychoose =v_city[-3]

print(v_citychoose)
v_mycity = v_city[1:4]

print(v_mycity)
#len

p_len_v_city_1 =len(v_mycity)

print("Size of v_mycity is ", p_len_v_city_1)

print(v_mycity)



#Append

v_mycity.append("SAO PAULO")

print(v_mycity)



v_mycity.append("PARIS")

print(v_mycity)
#reverse

v_city.reverse()

print(v_city)
#Sort

v_city.sort()

print(v_city)
v_mycity.sort()

print(v_mycity)
#Remove



#First add barcelona and after this add roma then Remove barcelona and roma 

v_city.append("barcelona")

v_city.append("roma")

print(v_city)

v_city.remove("barcelona")

v_city.remove("roma")

print(v_city)
country_populatıon = {"Belgıum" : "11 miilion" , "Germany" : "82 million" , "Colombia" : "49 million " , "Canada" : "37 million"}



print(country_populatıon)

print(type(country_populatıon))
p_count =country_populatıon["Germany"]

print(p_count)

print(type(p_count))
#Keys and Values

p_keys = country_populatıon.keys()

p_values = country_populatıon.values()



print(p_keys)

print(type(p_keys))



print(p_values)

print(type(p_values))
print(len(p_keys))



print(len(p_values))



p_num1=25

p_num2=37



if p_num1 > p_num2 :

    print(p_num1 , "it is grater then " , p_num2)

elif p_num1 < p_num2 : 

    print(p_num1 , "it is smaller than" , p_num2)

    

else : 

    print("this two variables are equal")

    
def p_Comparison1(p_num1 , p_num2):

    if p_num1 > p_num2:

        print(p_num1 , "is grater then " , p_num2)

    elif p_num1 < p_num2 :

        print(p_num1 ,  "is smaller then" , p_num2)

    else :

        print("These" , p_num1 , "variables are equal")

        

p_Comparison1(200,300)

p_Comparison1(100,171)

p_Comparison1(111,111)
#usıng 'IN' with LIST



def p_usıng(p_search,p_searchlist):

    if p_search in p_searchlist:

        print("you are lucky" , p_search , "is in list")

    else:

        print(p_search , "is not in list.PLEASE TRY AGAIN")

        

p_list =list(country_populatıon.keys())       

print(p_list)

print(type(p_list))





p_usıng("Germany" , p_list)

p_usıng("turkey"  , p_list)
for x in range(0,8):

    print("NUMBER IS" , x)
p_yourmessage = "NUMBERS ARE ABOVE"

print(p_yourmessage)
for p_chrs in p_yourmessage:

    print(p_chrs)

    print("-------")
for p_chrs in p_yourmessage.split():

    print(p_chrs)

print(v_numbers)

p_sum_v_numbers = sum(v_numbers)

print("TOTAL IS V_NUMBERS IS: " , p_sum_v_numbers)



print()

v_cum_v_numbers=0

v_loopindex=0

for v_current in v_numbers : 

    v_cum_v_numbers = v_cum_v_numbers + v_current

    print(v_loopindex, "nd value is : " , v_current)

    print("Cumulative is : " , v_cum_v_numbers)

    v_loopindex = v_loopindex + 1 

    print("------")

x = 0

while(x < 6):

    print("SCORE IS : " , x)

    x = x+1
print(v_numbers)

print()



y = 1 

k =len(v_numbers)



while(y<k):

    print(v_numbers[2])

    y =y+1

#Lets find minimum and maximum number in list



p_list = [5,7,9,-100,-32,-85,23,-95]



v_min = 0

v_max = 0



v_index = 0

v_len = len(p_list)



while(v_index < v_len):

    v_current = p_list[v_index]

    

    if v_current > v_max:

        v_max = v_current

        

    if v_current < v_min:

        v_min =v_current

        

    v_index = v_index+1

        

print("MAXIMUM NUMBER IS : " ,v_max)

print("MINIMUM NUMBER IS : " ,v_min)

#Import library to use 

import numpy as np
p_array= [1,2,3,4,5,6,7,8,9,10,11,12,13,14]

p_array_np = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14])

print("p_array : " , p_array)

print("type of p_array : " , type(p_array))
print("p_array_np : " , p_array_np)

print("type of p_array_np : " , type(p_array_np))
#shape 

p_shape = p_array_np.shape

print("p_shape : " , p_shape , "and type is : " , type(p_shape))
#Reshape 

p_array1_np = p_array_np.reshape(7,2)

print(p_array1_np)
p_shape1 = p_array1_np.shape

print("p_shape1 : " , p_shape1 ,  "and type is : " , type(p_shape1) )
#Dımensıon

p_dimen = p_array_np.ndim

print("p_dimen : ", p_dimen , "and type is : " , type(p_dimen))
#Dtype.name

p_dtype = p_array_np.dtype.name

print("p_dtype : " , p_dtype , "and type is : " , type(p_dtype))
#Size 

p_size = p_array_np.size

print("p_size : " , p_size , "and type is : " , type(p_size))
#lets crate 2*7 array

p_array2_np = np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])

print(p_array2_np)

print("--------------")

print("Shape is : " , p_array2_np.shape)
#Zeros

p_array3_np = np.zeros((7,2))

print(p_array3_np)
#Uptade an item on this array

p_array3_np[0,1]=18

print(p_array3_np)
#Ones

p_array4_np = np.ones((2,7))

print(p_array4_np)
#Empty

p_array5_np=np.empty((2,5))

print(p_array5_np)
#Arange

p_array6_np=np.arange(20,85,30)

print(p_array6_np)

print(p_array6_np.shape)
#Linspace

p_array7_np = np.linspace(20,30,40)

p_array8_np = np.linspace(50,60,70)



print(p_array7_np)

print(p_array7_np.shape)

print("--------")

print(p_array8_np)

print(p_array8_np.shape)
#Sum , Subtract , Square

p_np = np.array([3,4,5])

p_np1 = np.array([6,8,10])



print(p_np + p_np1)

print(p_np - p_np1)

print(p_np1 - p_np)

print(p_np ** 4)
#Sinus

print(np.sin(p_np))

#True / False

p_np_TF = p_np < 7

print(p_np_TF)

print(p_np_TF.dtype.name)
#Element wise Product

p_np = np.array([3,4,5])

p_np1 = np.array([6,8,10])

print(p_np * p_np1)
#Transpose

p_np3 = np.array([[4,7,9],[9,2,5]])

p_np3Transpose = p_np3.T

print(p_np3)

print(p_np3.shape)

print()

print(p_np3Transpose)

print(p_np3Transpose.shape)
#Matrix Multiplication

p_np4 = p_np3.dot(p_np3Transpose)

print(p_np4)
#Exponential -->We will use on Statistics Lesson

p_np5Exp = np.exp(p_np3)



print(p_np3)

print(p_np5Exp)

#Random

p_np6 = np.random.random((12,12)) # -->>ıt will get between 0 and 1 random numbers

print(p_np6)
#Sum , #Max , Min

p_np6Sum = p_np6.sum

print("Sum of array : " , p_np6Sum)

print("Max of array : " , p_np6.max())

print("Min of array : " , p_np6.min())
#Sum with Column or Row

print("Sum of Columns : ")

print(p_np6.sum(axis=0)) # Sum of Columns

print()

print("Sum of Row : " )

print(p_np6.sum(axis=1)) #Sum of Row
#Square , Sqrt

print(np.sqrt(p_np6))

print()

print(np.square(p_np6))
#Add

p_np7 = np.array([4,5,6,7])

p_np8 = np.array([20,40,60,80])



print(np.add(p_np7,p_np8))
p_np9 = np.array([10,11,12,14,15])



print("First item is : " , p_np9[0])

print("Third item is : " , p_np9[2])
#Get top  rows : 

print(p_np9[0:3])
#Reserve

p_np9_Rev =p_np9[::-1]

print(p_np9_Rev)
p_np10 = np.array([[4,5,6,7,8] ,[9,10,11,12,13]])

print(p_np10)

print()

print(p_np10[0,4]) #-->>Get a row



print()

p_np10 [0,4] = 45 #-->Uptade a row

print(p_np10)
#Get all rows but 3rd columns

print(p_np10[:,2])
#Get 2nd rows but 2,3th columns

print(p_np10[1,1:4])
#Get last rows all columns

print (p_np10[-1,:])
#Get all rows but las columns

print(p_np10[:,-1])
#Flatten

p_np11=np.array([[3,4,5], [6,7,8],[9,10,11],[12,13,14]])

p_np12=p_np11.ravel()



print(p_np11)

print("Shape of p_np11 is : " , p_np11.shape)

print()

print(p_np12)

print("Shape of p_np12 is : " , p_np12.shape)

print()
#Reshape

p_np13 = p_np12.reshape(12,1)

print(p_np13)

print("Shape of p_np13 is : "  , p_np13.shape)
p_np14 = p_np13.T

print(p_np14)

print("Shape of p_np14 is : " , p_np14.shape)
p_np15 = np.array([[3,4],[5,6],[7,8]])



print(p_np15)

print()

print(p_np15.reshape(3,2))



print()

print(p_np15)
#Resize

p_np15.resize((3,2))

print(p_np15)
p_np16 = np.array([[1,2],[3,4]])

p_np17 = np.array([[6,7],[8,9]])



print(p_np16)

print(p_np17)

print()
#Vertical Stack

p_np18 = np.vstack((p_np16,p_np17))

p_np19 =np.vstack((p_np17,p_np16))



print(p_np18)

print()

print(p_np19)
#Horizantal Stack

p_np20 = np.hstack((p_np18,p_np19))

p_np21 = np.hstack((p_np19,p_np18))



print(p_np20)

print()

print(p_np21)
p_list1 = [4,5,6,7]

p_np26 = np.array(p_list1)



print(p_list1)

print("Type of list : " , type(p_list1))

print()

print(p_np26)

print("Type of p_np26 is : " , type(p_np26))
p_list2 = list(p_np26)

print(p_list2)

print("Type of p_list2 is : " , type(p_list2))
p_list3 =p_list2

p_list4 =p_list2



print(p_list2)

print()

print(p_list3)

print()

print(p_list4)
p_list2[0]= 1923

print(p_list2)

print()

print(p_list3)

print()

print(p_list4)
p_list4 = p_list2.copy

p_list6 = p_list2.copy



print(p_list4)

print()

print(p_list6)
p_list2[0] = 1919

print(p_list2)

print()

print(p_list4)

print()

print(p_list6)