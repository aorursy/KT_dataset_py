print("Hi Kaggle")
v_message = "hello world"



print("Hi")
print(v_message)
v_name = "melike"

v_surname = "karaman"



v_fullname = v_name + v_surname

print(v_fullname)
v_fullname = v_name + " " + v_surname

print(v_fullname)
v_num1 ="800"

v_num2 ="600"

v_numSum1 = v_num1 + v_num2

print(v_numSum1)
#length



v_lenFull = len(v_fullname)

print("v_fullname:" ,v_fullname, "and lenght is : " ,v_lenFull)

v_titleF = v_fullname.title()

print ("v_fullname :", v_fullname , " and title is : " , v_titleF)
#upper:

v_upperF = v_fullname.upper()



#lower

v_lowerF = v_fullname.lower()

print("v_fullname:" , v_fullname , " Upper :" ,v_upperF , " Lower : " ,v_lowerF)
v_2ch = v_fullname [8]

print(v_2ch)
v_num1 = 800

v_num2 = 600

v_sum1 = v_num1 + v_num2



print(v_sum1, " and type : " , type(v_sum1))
v_num1 = v_num1 + 88

v_num2 = v_num2 - 25.5

v_sum1 = v_num1 + v_num2 



print(v_num1)
print("v_sum1 : " ,v_sum1 , " type : " , type(v_sum1))
v_fl1 = 48.5

v_fl2 = 22.5

v_s3 = v_fl1 + v_fl2



print(v_s3 , type(v_s3))
def f_SayHi():

    print("Hi.I am coming from f_SayHi")

    

def f_SayHi2():

    print("Hi.I am coming from f_SayHi2")

    print("Nice")

    

    

f_SayHi()
f_SayHi2()
def f_saymessage(v_message1):

    print(v_message1,"came from neptune")

    

    

def f_getfullname(v_firstname,v_surname,v_age):

    print("Welcome", v_firstname, "", v_surname, "your age is:",v_age)
f_saymessage("How do you feel ?")
f_getfullname("Melike" ,"KARAMAN" ,16)
def f_Calc1(f_num1,f_num2,f_num3):

    v_result = f_num1*f_num2 - f_num3

    print("Result =", "", v_result)

    
f_Calc1(80,60,180)
#return function

def f_Calc2(v_num1,v_num2,v_num3):

    v_out = v_num1 + v_num2 + v_num3*8

    print("Hi from f_Calc2")

    return v_out

v_came=f_Calc2(8,3,9)

print("Score is :",v_came)
#Default Functions:

def f_populationInfo(v_city,v_population,v_country = "TURKEY"):

    print("City : " , v_city , "Population :", v_population, "Country :",v_country)
f_populationInfo("Trabzon","764.714")

f_populationInfo("St.Petersburg","6.000.000","RUSSIA")
#Flexible Functions :



def f_flex1 (v_name, *v_colors):

    print("Hi",v_name ,"your favorite color is :",v_colors[4])
f_flex1("Melike","Blue","Black","Yellow","Brown","Purple","Green")
# Lambda Function :



v_result = lambda x : x*8

print("Result is :",v_result(86))
def f_total(v_edge1,v_edge2,v_edge3,v_edge4):

    v_perimeter=(v_edge1+v_edge2+v_edge3+v_edge4)

    v_area=(v_edge1*v_edge3)

    print(v_perimeter+v_area)
f_total(12,12,4,4)
v_list1 = [3,1.8,4,1,5.2,9]

print(v_list1)

print("Type of 'v_list1' is : " , type(v_list1))
v_number1 = v_list1 [4]

print(v_number1)

print("Type of 'v_number1' is : " , type (v_number1))
v_planet1 = ["Mercury","Venus","Earth","Mars","Jupiter","Saturn","Uranus","Neptune","Pluto"]

print(v_planet1)

print("Type of 'v_planet1' is : " , type(v_planet1))
v_planet1_2 = v_planet1[7]

print(v_planet1_2)

print("Type of 'v_planet1_2' is : " , type(v_planet1_2))
v_planet1_c1 = v_planet1 [-9]

print(v_planet1_c1)
v_planet1_c2 = v_planet1 [1:7]

print(v_planet1_c2)
#Len

v_len_planet1 = len(v_planet1_c2)

print("Size of 'v_planet1_c2' is : " , v_len_planet1)

print(v_planet1_c2)

#Append

v_planet1_c2.append ("Moon")

print(v_planet1_c2)



v_planet1_c2.append ("Mars")

print(v_planet1_c2)
#Reverse

v_planet1_c2.reverse()

print(v_planet1_c2)

#Sort

v_planet1_c2.sort()

print(v_planet1_c2)
#Remove



#First add 'Uranus' then remove 'Uranus'



v_planet1_c2.append ("Uranus")

print(v_planet1_c2)
v_planet1_c2.remove("Uranus")

print(v_planet1_c2)
v_capital = {"Ireland":"Dublin" , "Mongolia" : "Ulaanbaatar" , "Norway": "Oslo" , "Sweden":"Stockholm" , "Switzerland":"Bern"}



print(v_capital)

print(type(v_capital))
v_Ireland = v_capital["Ireland"]

print(v_Ireland)

print(type(v_Ireland))
#Keys & Values



v_keys = v_capital.keys()

v_values = v_capital.values()





print(v_keys)

print(type(v_keys))



print()

print(v_values)

print(type(v_values))
v_num1 = 88

v_num2 = 66



if v_num1 > v_num2:

    print(v_num1 , " is greater then " , v_num2)

    

elif v_num1 < v_num2:

    print(v_num1 , " is smaller then " , v_num2)

    

else :

    print("This 2 variables are equal")
# < , <= , > , >= , == , <>





def v_Comparison1(v_num1 , v_num2):

    

    

    if v_num1 > v_num2:

        print(v_num1 , " is greater then " , v_num2)

        

    elif v_num1 < v_num2:

        print(v_num1 , " is smaller then " , v_num2)

        

    else :

        print("These " , v_num1 , " variables are equal")

        

v_Comparison1(16,16)

v_Comparison1(25,36)

v_Comparison1(64,49)
# using 'IN' with LIST





def d_search (v_search, v_searchList):

    

    if v_search in v_searchList :

        print("Yup! ",v_search , " is in list :)")

        

    else :

        print( v_search , " is not in list :( ")



v_list = list(v_capital.keys())

print(v_list)

print(type(v_list))



d_search("Sweden" , v_list)

d_search("Vietnam" , v_list)
for a in range(0,8):

    print(  a , " days has passed since that day")
v_fav_country =  "IRELAND IS MY FAVORITE COUNTRY"

print(v_fav_country)
for v_down in v_fav_country:

    print(v_down)

    print("------")
for v_down in v_fav_country.split():

    print(v_down)
v_list2 = [8,6,10,1,16,18]

print(v_list2)
print(v_list2)

v_sum_list2 = sum(v_list2)

print("Sum of v_list2 is : " , v_sum_list2)



print()

v_cum_list2 = 0

v_loopindex = 0



for v_current in v_list2:

    

    v_cum_list2 = v_cum_list2 + v_current

    

    print(v_loopindex , " nd value is : " , v_current)

    print("Cumulative is : " , v_cum_list2)

    

    v_loopindex = v_loopindex + 1

    print("------")
m = 0

while(m < 10):

    

    print("You have traveled", m ,"kilometer"  )

    

    m = m+2
print(v_list2)

print()



m= 0

x = len(v_list2)



while(m<x):

    print(v_list2[m])

    m=m+1
#Let's find minimum and maximum number in list



v_list3 = [19,-28,7,-10,4,62,-18,300,-267]



v_min = 0

v_max = 0



v_index = 0

v_len = len(v_list3)



while (v_index < v_len):

    v_current = v_list3[v_index]

    

    if v_current > v_max:

        v_max = v_current

    

    if v_current < v_min:

        v_min = v_current

    

    v_index = v_index+1



print ("Max. number is : " , v_max)

print ("Min. number is : " , v_min)
#Import library to use



import numpy as np
v_array1 = [9,8,7,6,5,4,8,7,6,5,4,3,7,6,5,4,3,2,6,5,4,3,2,1,5,4,3,2,1,0]



v_array1_np = np.array([9,8,7,6,5,4,8,7,6,5,4,3,7,6,5,4,3,2,6,5,4,3,2,1,5,4,3,2,1,0])
print("v_array1 : " , v_array1)

print("Type of v_array1 : " , type(v_array1))
print("v_array1_np : " , v_array1_np)

print("Type of v_array1_np : " , type(v_array1_np))
# shape

v_shape1 = v_array1_np.shape

print("v_shape1 : " , v_shape1 , " and type is : " , type(v_shape1))
# Reshape

v_array2_np = v_array1_np.reshape(5,6)

print(v_array2_np)
v_shape2 = v_array2_np.shape

print("v_shape2 : " , v_shape2 , " and type is : " , type(v_shape2))
# Dimension

v_dimen = v_array2_np.ndim

print("v_dimen : " , v_dimen , " type is : " , type(v_dimen))
# Dtype.name

v_dtype = v_array2_np.dtype.name

print("v_dtype : " , v_dtype , " and type is : " , type(v_dtype))
# Size

v_size = v_array2_np.size

print("v_size : " , v_size , " and type : " , type(v_size))
# Let's create 5x6 array

v_array3_np = np.array([[9,8,7,6,5,4],[8,7,6,5,4,3],[7,6,5,4,3,2],[6,5,4,3,2,1],[5,4,3,2,1,0]])

print(v_array3_np)

print("---------------")

print("Shape is : " , v_array3_np.shape)
# Let's do it with string







v_array_str = ['Sanma şâhım ','herkesi sen', 'sâdıkâne', 'yâr olur','Herkesi sen','dost mu sandın' ,'belki ol','ağyâr olur','Sâdıkâne', 'belki ol' ,'bu âlemde', 'dildâr olur','Yâr olur', 'ağyâr olur', 'dildâr olur', 'serdâr olur']



v_array_str1 = np.array (['Sanma şâhım ','herkesi sen', 'sâdıkâne', 'yâr olur','Herkesi sen','dost mu sandın' ,'belki ol','ağyâr olur','Sâdıkâne', 'belki ol' ,'bu âlemde', 'dildâr olur','Yâr olur', 'ağyâr olur', 'dildâr olur', 'serdâr olur'])
print("v_array_str : " , v_array_str)

print("Type of v_array_str : " , type(v_array_str))
print("v_array_str1 : " , v_array_str1)

print("Type of v_array_str1: " , type(v_array_str1))
v_shape1_str = v_array_str1.shape

print("v_shape1_str : " , v_shape1_str , " and type is : " , type(v_shape1_str))
v_array2_str = v_array_str1.reshape(4,4)

print(v_array2_str)

print("                                                    " ,"-Yavuz Sultan Selim")
# Zeros

v_array4_np = np.zeros((5,7))

print(v_array4_np)



type(v_array4_np)
# Update an item on this array 

v_array4_np[2,3] = 8

print(v_array4_np)
# Ones



v_array5_np = np.ones((8,6))

print(v_array5_np)
# Empty

v_array6_np = np.empty((5,4))

print(v_array6_np)
# Arrange

v_array7_np = np.arange(24,96,8)

print(v_array7_np)

print(v_array7_np.shape)
# Linspace



v_array8_np = np.linspace(20,400,5)

v_array9_np = np.linspace(7,84,30)



print(v_array8_np)

print(v_array8_np.shape)

print("-----------------------")

print(v_array9_np)

print(v_array9_np.shape)
# Sum , Subtract , Square

v_np = np.array([8,6,4])

v_np1 = np.array([5,7,9])



print(v_np - v_np1)

print(v_np + v_np1)

print(v_np1 - v_np)

print(v_np ** 2)
# Sinus



print(np.sin(v_np))
# True / False

v_np_tf = v_np < 6

print(v_np_tf)

print(v_np_tf.dtype.name)
# Element wise Prodcut



v_np = np.array([8,6,4])

v_np1 = np.array([5,7,9])



print(v_np * v_np1)
# Transpose





v_np1= np.array([[6,7,5],[8,2,3]])

v_np1transpose = v_np1.T

print(v_np1)

print(v_np1.shape)

print()

print(v_np1transpose)

print(v_np1transpose.shape)
# Matrix Multiplication



v_np2 = v_np1.dot(v_np1transpose)

print(v_np2)
# Exponential --> We will use on Statistics Lesson



v_np1Exp = np.exp(v_np1)



print(v_np1)

print(v_np1Exp)
# Random 



v_np3 = np.random.random((8,4)) # --> It will get between 0 and 1 random numbers

print(v_np3)
#Sum , Max ,Min



v_np3Sum = v_np3.sum()



print("Sum of array : ", v_np3Sum)  #--> Remember ! If you get sum of array we can use that :  sum(array1)



print("Max of array : ", v_np3.max()) #--> Remember ! If you get max of array we can use that :  max(array1)



print("Min of array : ", v_np3.min()) #--> Remember ! If you get min of array we can use that :  min(array1)
# Sum with Column or Row

print("Sum of columns :")

print(v_np3.sum(axis=0)) # --> Sum of Columns

print()

print("Sum of rows :")

print(v_np3.sum(axis=1)) # -->Sum of Rows
# Square , Sqrt



print(np.sqrt(v_np3))

print()

print(np.square(v_np3))
# Add



v_np4 = np.array([7,4,9,6,5])

v_np5 = np.array([56,84,73,19,46])



print(np.add(v_np4,v_np5))
v_np6 = np.array([54,85,7,4,14,6,35,58,9,44,75,915])



print("First item is : " , v_np6[0])

print("Eighth item is : " , v_np6[7])
# Get top 7 rows :



print(v_np6[0:7])
# Reverse



v_np6_Rev = v_np6[::-1]

print(v_np6_Rev)
v_np7 = np.array([[8,6,4,7,2],[17,21,12,13,18]])

print(v_np7)

print()

print(v_np7[0,3]) #--> Get a row



print()

v_np7[0,3] = 9 #--> Update a row

print(v_np7)
# Get all rows but 3rd columns :



print(v_np7[:,2])
#Get 2nd row but 2,3,4th columns



print(v_np7[1,1:4])
# Get last row all columns



print(v_np7[-1,:])
# Get last columns but all rows



print(v_np7[:,-1])
#Flatten

v_np8 = np.array([[24,23,22],[21,20,19],[18,17,16],[15,14,13]])

v_np9 = v_np8.ravel()



print(v_np8)

print("Shape of v_np8 is : " ,v_np8.shape)

print()

print(v_np9)

print("Shape of v_np9 is : " ,v_np9.shape)

print()
# Reshape



v_np10 = v_np9.reshape(3,4)

print(v_np10)

print("Shape of v_np10 is : " ,v_np10.shape)
v_np11 = v_np10.T

print(v_np11)

print("Shape of v_np11 is : " ,v_np11.shape)
v_np12 = np.array([[4,5],[2,8],[6,8]])



print(v_np12)

print()

print(v_np12.reshape(2,3))



print()

print(v_np12) #--> It has not changed !
# Resize

v_np12.resize((2,3))

print(v_np12) # --> Now it changed !  Resize can change its shape
v_np13 = np.array([[5,4],[3,2]])

v_np14 = np.array([[9,8],[7,6]])



print(v_np13)

print()

print(v_np14)
# Vertical Stack

v_np15 = np.vstack((v_np13,v_np14))

v_np16 = np.vstack((v_np14,v_np13))



print(v_np15)

print()

print(v_np16)
# Horizontal Stack

v_np17 = np.hstack((v_np13,v_np14))

v_np18 = np.hstack((v_np14,v_np13))



print(v_np17)

print()

print(v_np18)
v_list = [9,8,7,6]

v_np19 = np.array(v_list)



print(v_list)

print("Type of list : " , type(v_list))

print()

print(v_np19)

print("Type of v_np19 : " , type(v_np19))
v_list1 = list(v_np19)

print(v_list1)

print("Type of list1 : " , type(v_list1))
v_list2 = v_list1

v_list3 = v_list1



print(v_list1)

print()

print(v_list1)

print()

print(v_list3)
v_list1[2] = 86



print(v_list1)

print()

print(v_list2) # --> Same address with list1

print()

print(v_list3) # --> Same address with list1
v_list4 = v_list1.copy()

v_list5 = v_list1.copy()



print(v_list4)

print()

print(v_list5)
v_list1[0] = 45



print(v_list1)

print()

print(v_list4) # --> Not same address with list1

print()

print(v_list5) # --> Not same address with list1