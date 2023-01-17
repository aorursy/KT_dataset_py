print("Hello World")
v_message = "Hİ FRİENDS!"

print("Hi")
print(v_message)
v_name = "ilayda"

v_surname = "tepeyurt"

v_fullname = v_name + v_surname

print(v_fullname)
v_fullname = v_name + " " + v_surname

print(v_fullname)
v_num1 = "100"

v_num2 = "200"

v_numsum1 = v_num1 + v_num2

print(v_numsum1)
#lenght

v_lenfull = len(v_fullname)

print("v_fullname : " , v_fullname , "and lengt is : " , v_lenfull)
v_titleF = v_fullname.title()

print("v_fullname : " , v_fullname , "and title is : " , v_titleF)
#upper

v_upperF = v_fullname.upper()

print("v_fullname : " , v_fullname , "and upper is : " , v_upperF)



#lower

v_lowerF = v_fullname.lower()

print("v_fullname : " , v_fullname , "and lower is : " , v_lowerF)
v_2ch = v_fullname[7]

print(v_2ch)
v_num1 = 100

v_num2 = 200

v_sum1 = v_num1 + v_num2

print(v_num1 , "and type is:" , type(v_sum1))

v_num1 = v_num1 + 50

v_num2 = v_num2 - 25.5

v_sum1 = v_num1 + v_num2



print(v_num1)
print("v_sum1 :" , v_sum1 , "and type is: " , type(v_sum1))
v_fl1 = 25.5

v_fl2 = 15.5

v_s3 = v_fl1 + v_fl2

print(v_s3, type(v_s3))
def f_SayHello():

    print("hi. I am from f_SayHello")

    

def f_SayHello2():

        print("hi. I am from f_SayHello2")

        print("Good")

        

f_SayHello()
f_SayHello2()

def f_saymessage(v_message1):

    print(v_message1 , "came from 'f_saymessage'")

    

def f_getfullname(v_firstname , v_surname , v_age):

    print("welcome" , v_firstname ," " , v_surname ,"your age :" , v_age)

    
f_saymessage("what about you?")
f_getfullname("İLAYDA" , "TEPEYURT" , 16)

def f_calc1(f_num1 , f_num2 , f_num3):

    v_sonuc = f_num1 + f_num2 + f_num3

    print("sonuc = " , v_sonuc)

    
f_calc1(80 , 50 , 50)
#return fonction

def f_calc2(v_num1 , v_num2 , v_num3):

    v_out = v_num1 + v_num2 + v_num3*3

    print("hi from f_calc2")

    return v_out

v_gelen = f_calc2(1,2,4)

print("score is :" , v_gelen)
#default functions :

def f_getschoolınfo(v_name , v_studentcount , v_city = "istanbul"):

    print("name : " , v_name , "st count : " , v_studentcount ,"city : " , v_city)

    

f_getschoolınfo("AAIHL" , "380")

f_getschoolınfo("AAIHL" , "380" , "GİRESUN")

f_getschoolınfo("AAİHL" , " " , "ANKARA")


#flexible functions :

def f_flex1(v_name , *v_messages):

    print("hi" , v_name , "your first message : " , v_messages[2])

    
f_flex1("ilayda" , "hello" , "what's up" , "what about you")

#lambda function

v_result = lambda x : x*3

print("result is : " , v_result(6))
def f_alan(v_kenar1,v_kenar2):

    print(v_kenar1*v_kenar2)
f_alan(2,2)
l_list=[1,2,3,4,5,6,7]

print(l_list)

print("the type of l_list : " , type(l_list))

l_list2=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

print(l_list2)

print("type of'l_list2':" ,type(l_list))
i_list2_4=l_list2[4]

print(i_list2_4)

print("type of 'i_list2_4'is:",type(i_list2_4))
i_list_3=l_list2[-4]

print(i_list_3)
i_list_2=l_list2[0:5]

print(i_list_2)

#Len

r_len_i_list_2=len(i_list_2)

print("size of'i_list_2' is:", r_len_i_list_2)

print(i_list_2)
#Append

i_list_2.append("Saturday")

print(i_list_2)



i_list_2.append("Sunday")

print(i_list_2)
#Reverse

i_list_2.reverse()

print(i_list_2)
#Sort

i_list_2.sort()

print(i_list_2)
#Remove

#firs add 'saturday' then remove 'saturday'

i_list_2.append("Saturday")

print(i_list_2)
i_list_2.remove("Saturday")

print(i_list_2)
i_dict = {"Home":"Ev" , "Color":"Renk" , "Table":"Masa" , "Mirror":"Ayna"}



print(i_dict)

print(type(i_dict))
v_color = i_dict["Color"]

print(v_color)

print(type(v_color))
#keys and values

v_keys = i_dict.keys()

v_values = i_dict.values()





print(v_keys , "and type is :",type(v_keys))

print(v_values , "and type is :", type(v_values))
v_var1 = 10

v_var2 = 30

if v_var1 > v_var2:

    print(v_var1 , "is greater then" , v_var2)

    

elif v_var1 < v_var2:

    print(v_var1 , "is smiller then" , v_var2)

    

else :

    print("this 2 variables are equel")

    

    
#< , > , <= , >= , == , <>

def f_Comparison1(v_comp1 , v_comp2):

    if v_comp1 > v_comp2:

        print(v_comp1 , "is greater then" , v_comp2)

    

    elif v_comp1 < v_comp2:

        print(v_comp1 , "is smiller then" , v_comp2)

    

    else :

        print("this" , v_comp1 , "variables are equel")

        

f_Comparison1(22,35)

f_Comparison1(22,15)

f_Comparison1(13,13)



    
#using "İN" with list



def f_includeornot(v_search , v_searchList) :

    if v_search in v_searchList :

        print("good news !" , v_search , "is in list.")

    else : 

        print(v_search , "is not in list. Sorry!")

        

l_list = list(i_dict.keys())

print(l_list)

print(type(l_list))



f_includeornot("Home",l_list)

f_includeornot("Pencil",l_list)
for a in range(0,5):

    print(a,"THİS İS MY PYTHON")
fav_song="DUSK TİLL DAWN"

print(fav_song)
for v_font in fav_song:

    print(v_font)

    print("------")
for v_font in fav_song.split():

    print(v_font)
i_list3=[1,2,3,4,5,6]

print(i_list3)
print(i_list3)

i_sum_list3= sum(i_list3)

print("sum of i_list3 is : " , i_sum_list3)



print()

i_cumlist3=0

i_loopindex=0



for v_current in i_list3:

    i_cumlist3=i_cumlist3 + v_current

    print(i_loopindex , "nd value is : " , v_current)

    print("cumulative is : " , i_cumlist3)

    

    i_loopindex= i_loopindex + 1

    print("------")
i = 0

while(i < 3) :

    print("YOU WİLL WRİTE" ,i, "SENTENCES")

    

    i=i+2



print(i_list3)

print()



i=0

x=len(i_list3)



while(i<x):

    print(i_list3[i])

    i=i+1
#let's find minimum and maximum number in list



i_list4=[11,-35,700,60,-55]



v_min=0

v_max=0



v_index=0

v_len=len(i_list4)



while(v_index<v_len):

    v_current=i_list4[v_index]

    

    if v_current>v_max:

        v_max=v_current

        

    if v_current<v_min:

        v_min=v_current

        

        

    v_index=v_index+1

    

    

print("max. number is : " , v_max)

print("min. number is : " , v_min)

        
#import library to use

import numpy as np
v_array1=[1,2,3,4,5,6,7,8,9,10]

v_array2_np=np.array([1,2,3,4,5,6,7,8,9,10])
print("v_array1:", v_array1)

print("the type is : ", type(v_array1))
print("v_array2_np:", v_array2_np)

print("the type is:",type(v_array2_np))
#shape

r_shape1=v_array2_np.shape

print("r_shape is:" , r_shape1,"and type is :" , type(r_shape1))
#reshape

v_reshape1=v_array2_np.reshape(2,5)

print(v_reshape1)
v_reshape2=v_reshape1.shape

print("v_reshape2:", v_reshape2, "and type is:", type(v_reshape2))
#dimension

r_dimen1=v_array2_np.ndim

print("r_dimen1:",r_dimen1, type(r_dimen1) )
#dtype.name

r_dtype1=v_array2_np.dtype.name

print("r_dtype1:",r_dtype1,"the type of:", type(r_dtype1))
#size

r_size1=v_array2_np.size

print("r_size1:",r_size1,"the type of:",type(r_size1))
#let's create 3x4 array

v_array3_np=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

print(v_array3_np)

print("--------")

print("shape is:",v_array3_np.shape)
#zeros

v_array4_np=np.zeros((3,6))

print(v_array4_np)
#Update an item on this array

v_array4_np[0,5]=21

print(v_array4_np)
v_array5_np=np.ones((4,4))

print(v_array5_np)
#empty

v_array6_np=np.empty((2,4))

print(v_array6_np)
#arrange

v_array7_np=np.arange(10,40,10)

print(v_array7_np)

print(v_array7_np.shape)
#linspace

v_array8_np=np.linspace(10,40,5)

v_array9_np=np.linspace(4,32,4)



print(v_array8_np)

print(v_array8_np.shape)

print("the type is:",type(v_array8_np))

print("--------")

print(v_array9_np)

print(v_array9_np.shape)

print("the type is:",type(v_array9_np))
#SUM,SUBTRACT,SQUARE

v_np1=np.array([2,4,6])

v_np2=np.array([3,6,9])



print(v_np1 + v_np2)

print(v_np1 - v_np2)

print(v_np2 - v_np1)

print(v_np1 ** v_np2)
#sinus

print(np.sin(v_np2))
#true / false

v_np2_TF=v_np2<7

print(v_np2_TF)

print(v_np2.dtype.name)
#element wise product

v_np1=np.array([1,2,3])

v_np2=np.array([2,4,6])

print(v_np1 * v_np2)

print("the type is:",type(v_np1 * v_np2))
#Transpose

v_np4=np.array([[1,2,3,4,5],[6,7,8,9,10]])

v_np4Transpose=v_np4.T

print(v_np4)

print(v_np4.shape)

print()

print(v_np4Transpose)

print(v_np4Transpose.shape)
# Matrix Multiplication

v_np5 = v_np4.dot(v_np4Transpose)

print(v_np5)
# Exponential --> We will use on Statistics Lesson

v_np4Exp= np.exp(v_np4)

print(v_np4)

print(v_np4Exp)
# Random 

v_np7 = np.random.random((6,6)) # --> It will get between 0 and 1 random numbers

print(v_np7)
#sum , max , min

v_np7sum=v_np7.sum()

print("Sum of array : ", v_np7sum)  #--> Remember ! If you get sum of array we can use that :  sum(array1)

print("Max of array : ", v_np7.max()) #--> Remember ! If you get max of array we can use that :  max(array1)

print("Min of array : ", v_np7.min()) #--> Remember ! If you get min of array we can use that :  min(array1)

print("the type of v_np7sum:", type(v_np7sum))
# Sum with Column or Row

print("Sum of Columns :")

print(v_np7.sum(axis=0)) # --> Sum of Columns

print()

print("Sum of Rows :")

print(v_np7.sum(axis=1)) #Sum of Rows
# Square , Sqrt

print(v_np7)

print()

print(np.sqrt(v_np7))

print()

print(np.square(v_np7))
# Add

v_np9 = np.array([2,4,6,8,10])

v_np10 = np.array([10,30,40,60,70])



print(np.add(v_np9,v_np10))
v_np10 = np.array([2,4,6,8])



print("First item is : " , v_np10[0])

print("Third item is : " , v_np10[2])
# Get top 3 rows :

print(v_np10[0:3])
# Reverse

v_np10_Rev = v_np10[::-1]

print(v_np10_Rev)
v_np11 = np.array([[1,2,3,4],[11,12,13,14]])

print(v_np11)

print()

print(v_np11[1,3]) #--> Get a row



print()

v_np11[1,3] = 314 #--> Update a row

print(v_np11)
# Get all rows but 3rd columns :

print(v_np11[:,2])
#Get 2nd row but 2,3,4th columns

print(v_np11[1,1:4])

# Get last row all columns

print(v_np11[-1,:])
# Get last columns but all rows

print(v_np11[:,-1])
#Flatten

v_np12 = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])

v_np13 = v_np12.ravel()



print(v_np12)

print("Shape of v_np12 is : " ,v_np12.shape)

print()

print(v_np13)

print("Shape of v_np13 is : " ,v_np13.shape)

print()
# Reshape

v_np15 = v_np13.reshape(3,4)

print(v_np15)

print("Shape of v_np15 is : " ,v_np15.shape)
v_np16 = np.array([[1,2],[3,4],[5,6]])



print(v_np16)

print()

print(v_np16.reshape(2,3))



print()

print(v_np16) #--> It has not changed !!
# Resize

v_np16.resize((2,3))

print(v_np16) # --> Now it changed !  Resize can change its shape
v_np17 = np.array([[1,2],[3,4]])

v_np18 = np.array([[5,6],[7,8]])



print(v_np17)

print()

print(v_np18)
# Vertical Stack

v_np19 = np.vstack((v_np17,v_np18))

v_np20 = np.vstack((v_np18,v_np17))



print(v_np19)

print()

print(v_np20)

# Horizontal Stack

v_np21 = np.hstack((v_np17,v_np18))

v_np22 = np.hstack((v_np18,v_np17))



print(v_np21)

print()

print(v_np22)
v_list1 = [1,2,3,4]

v_np23 = np.array(v_list1)



print(v_list1)

print("Type of list : " , type(v_list1))

print()

print(v_np23)

print("Type of v_np23 : " , type(v_np23))
v_list2 = list(v_np23)

print(v_list2)

print("Type of list2 : " , type(v_list2))
v_list3 = v_list2

v_list4 = v_list2



print(v_list2)

print()

print(v_list3)

print()

print(v_list4)
v_list2[0] = 30



print(v_list2)

print()

print(v_list3) # --> Same address with list2

print()

print(v_list4) # --> Same address with list2
v_list5 = v_list2.copy()

v_list6 = v_list2.copy()



print(v_list5)

print()

print(v_list6)
v_list2[0] = 88



print(v_list2)

print()

print(v_list5) # --> Not same address with list2

print()

print(v_list6) # --> Not same address with list2
# Import Library

import pandas as pd
# Let's create Data Frame from Dictionary

v_dict1 = { "LESSON" : ["MATH","HISTORY.","CHEMİSTRY","BIYOLOGY"],

            "POİNT":["85","90","80","86"],

            "HOUR":["6","2","2","2"]}



v_dataFrame1 = pd.DataFrame(v_dict1)



print(v_dataFrame1)

print()

print("Type of v_dataFrame1 is : " , type(v_dataFrame1))
# get top 5 rows

v_head1 = v_dataFrame1.head()

print(v_head1)

print()

print("Type of v_head1 is :" ,type(v_head1))
# get top 100 rows

print(v_dataFrame1.head(100))
# get last 5 rows

v_tail1 = v_dataFrame1.tail()

print(v_tail1)

print()

print("Type of v_tail1 is :" ,type(v_tail1))
# Columns

v_columns1 = v_dataFrame1.columns

print(v_columns1)

print()

print("Type of v_columns is : " , type(v_columns1))
v_info1 = v_dataFrame1.info()

print(v_info1)

print()

print("Type of v_info1 is : " , type(v_info1))
v_dtypes1 = v_dataFrame1.dtypes

print(v_dtypes1)

print()

print("Type of v_dtypes1 is : " , type(v_dtypes1))
v_descr1 = v_dataFrame1.describe()

print(v_descr1)

print()

print("Type of v_descr1 is : " , type(v_descr1))
v_lesson1 = v_dataFrame1["LESSON"]

print(v_lesson1)

print()

print("Type of v_lesson1 is : " , type(v_lesson1))
#add new columns

v_topic1 = ["NUMBERS","WAR","ATOMS","CELL"]

v_dataFrame1["TOPIC"] = v_topic1

print(v_dataFrame1.head())
# Get all rows ,  1 column



e_city=e_dataframe.loc[:,"City"]

print(e_city)

print()

print("type of e_city is:",type(e_city))