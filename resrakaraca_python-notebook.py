print("Hello World")
v_message="hello world"

print("Hi")
print(v_message)
v_name="rüveyda"

v_surname="karaca"

v_fullname=v_name+v_surname



print(v_fullname)
v_fullname=v_name+" "+v_surname



print(v_fullname)
#length

v_lenFull=len(v_fullname)



print("v_fullname :" ,v_fullname,"and length is :",v_lenFull )

            
v_titleF=v_fullname.title()



print("v_fullname:",v_fullname,"and title is:",v_titleF)
#upper:

v_upperF=v_fullname.upper()

#lower

v_lowerF=v_fullname.lower()



print("v_fullname:",v_fullname,"Upper:",v_upperF,"Lower:",v_lowerF)

    
v_2ch=v_fullname[5]



print(v_2ch)
v_num1="100"

v_num2="200"

v_numSum1=v_num1+v_num2



print(v_numSum1)
v_num1=100

v_num2=200

v_sum1=v_num1+v_num2



print(v_sum1,"and type:",type(v_sum1))
v_num1=v_num1+50

v_num2=v_num2-25.5

v_sum1=v_num1+v_num2



print(v_num1)
print("v_sum1:",v_sum1,"type:",type(v_sum1))
v_fl1=25.5

v_fl2=15.5

v_s3=v_fl1+v_fl2



print(v_s3,type(v_s3))
def f_SayHello():

    print("Hi.My pretty robot f_SayHello")

def f_SayHello2():

    print("Hi.My pretty robot f_SayHello2")

    print("Yes,it's me")

f_SayHello()
f_SayHello2()
def f_sayMessage(v_Message1):

    print(v_Message1,"I'm NextGen 'f_sayMessage'")

def f_getfullname(v_Name , v_Surname , v_Age):

    print("Welcome " , v_Name , " " , v_Surname , " your age : " , v_Age)
f_sayMessage("How are you ?")
f_getfullname("RÜVEYDA","KARACA",14)
def f_Calc1(f_Num1 , f_Num2 , f_Num3):

    v_Result = f_Num1 + f_Num2 + f_Num3

    print("Result = " ," " , v_Result)
f_Calc1(30 , 40 , 50)
# return function

def f_Calc2(v_Num1 , v_Num2 , v_Num3):

    v_Out = v_Num1+v_Num2+v_Num3*2

    print("Hi from f_Calc2")

    return v_Out
v_gelen = f_Calc2 (2,4,3)

print("Score is :",v_gelen )

        
# Default Functions :

def f_getSchoolInfo(v_Name,v_StudentCount,v_City = "ISTANBUL"):

    print("Name : " , v_Name , " St Count : " , v_StudentCount , " City : " , v_City)
f_getSchoolInfo("AAIHL" , 333)

f_getSchoolInfo("KAL" , 333 , "KOCAELİ")
# Flexible Functions :



def f_Flex1(v_Name , *v_messages):

    print("Hi " , v_Name , " your first message is : " , v_messages[2])
f_Flex1("NextGen" , "hi" , "how are you" ,"deleting all memories" )
# Lambda Function :



v_result1 = lambda x : x*4

print("Result is : " , v_result1(33))
def f_alan(kenar1,kenar2):

    print(kenar1*kenar2)
f_alan(5,8)
l_list1 = [1,5,3,4,9,2,6,0]

print(l_list1)

print("Type of 'l_list1' is : " , type(l_list1))
v_list1_4 = l_list1[7]

print(v_list1_4)

print("Type of 'v_list1_4' is : " , type(v_list1_4))
l_list2 = ["Slytherin","Gryffindor","Hufflepuff","Ravenclaw"]

print(l_list2)

print("Type of 'l_list2' is : " , type(l_list1))

v_list2_4 = l_list2[0]

print(v_list2_4)

print("Type of 'v_list2_4' is : " , type(v_list2_4))
v_list2_x3 = l_list2[-3]

print(v_list2_x3)
l_list2_2 = l_list2[0:2]

print(l_list2_2)
#Len

v_len_l_list2_2 = len(l_list2_2)

print("Size of 'l_list2_2' is : ",v_len_l_list2_2)

print(l_list2_2)
#Append

l_list2_2.append("Slytherin")

print(l_list2_2)



l_list2_2.append("Gryffindor")

print(l_list2_2)
#Reverse

l_list2_2.reverse()

print(l_list2_2)
#Sort

l_list2_2.sort()

print(l_list2_2)

#Remove



#First add 'Huflepuff' then Remove 'Huflepuff'

l_list2_2.append("Huflepuff")

print(l_list2_2)
l_list2_2.remove("Slytherin")

print(l_list2_2)

                  

d_dict1 = {"Gryffindor":"Lion" , "Slytherin" : "Snake" , "Hufflepuff": "Badger","Ravenclaw":"Eagle"}



print(d_dict1)

print(type(d_dict1))
v_school = d_dict1["Slytherin"]

print(v_school)

print(type(v_school))

          

        
v_keys = d_dict1.keys()

v_values = d_dict1.values()





print(v_keys)

print(type(v_keys))



print()

print(v_values)

print(type(v_values))
v_var1 = 23.3

v_var2 = 98.6



if v_var1 > v_var2:

    print(v_var1 , " is greater then " , v_var2)

elif v_var1 < v_var2:

    print(v_var1 , " is smaller then " , v_var2)

else :

    print("This 2 variables are equal")

def f_Comparison1(v_Comp1 , v_Comp2):

    if v_Comp1 > v_Comp2:

        print(v_Comp1 , " is greater then " , v_Comp2)

    elif v_Comp1 < v_Comp2:

        print(v_Comp1 , " is smaller then " , v_Comp2)

    else :

        print("These " , v_Comp1 , " variables are equal")

        

f_Comparison1(33,64)

f_Comparison1(28,22)

f_Comparison1(333,333)
def f_IncludeOrNot(v_search, v_searchList):

    if v_search in v_searchList :

        print("surprise ! ",v_search , " is in list.")

    else :

        print(v_search , " is not in list. Sorry :(")



l_list = list(d_dict1.keys())

print(l_list)

print(type(l_list))



f_IncludeOrNot("REK" , l_list)

f_IncludeOrNot(":)" , l_list)

for a in range(1904):

    print("BJK " , a)

v_turkMessage = "BİR ÖLÜR BİN DİRİLİRİZ"

print(v_turkMessage)
for v_chrs in v_turkMessage:

    print(v_chrs)

    print("------")
for v_chrs in v_turkMessage.split():

    print(v_chrs)
v_list2 = [3,0,160,1903,2443,188]

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
i = 0

while(i < 9):

    print("REK" , i)

    i = i+1
m = 0

while(m < 10):

    

    print("first game", m ,"level"  )

    

    m = m+2
print(v_list2)

print()



m= 0

x = len(v_list2)



while(m<x):

    print(v_list2[m])

    m=m+1
#Let's find minimum and maximum number in list



v_list3 = [100,445,7,-10,00,297,-18,390,-267]



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
# Import library to use

import numpy as np
v_array1 = [2,4,6,8,10,12,14,16,18,20]

v_array2_np = np.array([2,4,6,8,10,12,14,16,18,20])
print("v_array1 : " , v_array1)

print("Type of v_array1 : " , type(v_array1))
print("v_array2_np : " , v_array2_np)

print("Type of v_array2_np : " , type(v_array2_np))
# shape

v_shape1 = v_array2_np.shape

print("v_shape1 : " , v_shape1 , " and type is : " , type(v_shape1))
# Reshape

v_array3_np = v_array2_np.reshape(2,5)

print(v_array3_np)
v_shape2 = v_array3_np.shape

print("v_shape2 : " , v_shape2 , " and type is : " , type(v_shape2))
# Dimension

v_dimen1 = v_array3_np.ndim

print("v_dimen1 : " , v_dimen1 , " type is : " , type(v_dimen1))
# Dtype.name

v_dtype1 = v_array3_np.dtype.name

print("v_dtype1 : " , v_dtype1 , " and type is : " , type(v_dtype1))
# Size

v_size1 = v_array3_np.size

print("v_size1 : " , v_size1 , " and type : " , type(v_size1))
# Let's create 3x4 array

v_array4_np = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

print(v_array4_np)

print(">>>>>>>><<<<<<<<<")

print("Shape is : " , v_array4_np.shape)
# Zeros

v_array5_np = np.zeros((6,2))

print(v_array5_np)
# Update an item on this array 

v_array4_np[2,3] = 8

print(v_array4_np)
# Ones

v_array6_np = np.ones((5,8))

print(v_array6_np)
# Empty

v_array6_np = np.empty((5,4))

print(v_array6_np)
# Arrange

v_array8_np = np.arange(12,36,5)

print(v_array8_np)

print(v_array8_np.shape)
# Linspace

v_array9_np = np.linspace(14,30,5)

v_array10_np = np.linspace(14,30,20)



print(v_array9_np)

print(v_array9_np.shape)

print(">>>>><<<<<")

print(v_array10_np)

print(v_array10_np.shape)
# Sum , Subtract , Square

v_np1 = np.array([6,7,8])

v_np2 = np.array([8,7,6])



print(v_np1 + v_np2) #--> We can not addition of 2 arrays

print(v_np1 - v_np2)

print(v_np2 - v_np1)

print(v_np1 ** 2)
# Sinus

print(np.sin(v_np2))
# True / False

v_np2_TF = v_np2 < 6

print(v_np2_TF)

print(v_np2_TF.dtype.name)
# Element wise Prodcut

v_np1 = np.array([3,6,9])

v_np2 = np.array([2,4,6])

print(v_np1 * v_np2)
# Transpose

v_np5 = np.array([[1,3,5],[4,6,8]])

v_np5Transpose = v_np5.T

print(v_np5)

print(v_np5.shape)

print()

print(v_np5Transpose)

print(v_np5Transpose.shape)

v_np6 = v_np5.dot(v_np5Transpose)

print(v_np6)
v_np5Exp = np.exp(v_np5)



print(v_np5)

print(v_np5Exp)
v_np8 = np.random.random((6,6)) # --> It will get between 0 and 1 random numbers

print(v_np8)
v_np8Sum = v_np8.sum()

print("Sum of array : ", v_np8Sum)  #--> Remember ! If you get sum of array we can use that :  sum(array1)

print("Max of array : ", v_np8.max()) #--> Remember ! If you get max of array we can use that :  max(array1)

print("Min of array : ", v_np8.min()) #--> Remember ! If you get min of array we can use that :  min(array1)
print(np.sqrt(v_np8))

print()

print(np.square(v_np8))
v_np10 = np.array([2,4,6,8,10])

v_np11 = np.array([10,20,30,40,50])



print(np.add(v_np10,v_np11))
v_np12 = np.array([1,3,5,7,9])



print("First item is : " , v_np12[1])

print("Third item is : " , v_np12[3])
print(v_np12[0:7])
v_np12_Rev = v_np12[::-3]

print(v_np12_Rev)
v_np13 = np.array([[3,4,5,6,7],[11,12,13,14,15]])

print(v_np13)

print()

print(v_np13[1,3]) #--> Get a row



print()

v_np13[1,3] = 314 #--> Update a row

print(v_np13)
print(v_np13[:,1])
print(v_np13[1,1:4])
print(v_np13[-2,:])
print(v_np13[:,-2])
v_np14 = np.array([[3,6,9],[2,4,6],[4,8,12],[5,10,15]])

v_np15 = v_np14.ravel()



print(v_np14)

print("Shape of v_np14 is : " ,v_np14.shape)

print()

print(v_np15)

print("Shape of v_np15 is : " ,v_np15.shape)

print()
v_np16 = v_np15.reshape(3,4)

print(v_np16)

print("Shape of v_np16 is : " ,v_np16.shape)
v_np17 = v_np16.T

print(v_np17)

print("Shape of v_np17 is : " ,v_np17.shape)
v_np20 = np.array([[8,0],[0,8],[8,0]])



print(v_np20)

print()

print(v_np20.reshape(2,3))



print()

print(v_np20) #--> It has not changed !!
# Resize

v_np20.resize((2,3))

print(v_np20) # --> Now it changed !  Resize can change its shape
v_np21 = np.array([[1,2],[3,4]])

v_np22 = np.array([[5,6],[7,8]])



print(v_np21)

print()

print(v_np22)
# Vertical Stack

v_np23 = np.vstack((v_np21,v_np22))

v_np24 = np.vstack((v_np22,v_np21))



print(v_np23)

print()

print(v_np24)
# Horizontal Stack

v_np25 = np.hstack((v_np21,v_np22))

v_np26 = np.hstack((v_np22,v_np21))



print(v_np25)

print()

print(v_np26)
v_list1 = [0,00,000]

v_np30 = np.array(v_list1)



print(v_list1)

print("Type of list : " , type(v_list1))

print()

print(v_np30)

print("Type of v_np30 : " , type(v_np30))
v_list2 = list(v_np30)

print(v_list2)

print("Type of list2 : " , type(v_list2))

v_list3 = v_list2

v_list4 = v_list2



print(v_list2)

print()

print(v_list3)

print()

print(v_list4)
v_list2[0] = 00



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
v_list2[1] =00



print(v_list2)

print()

print(v_list5) # --> Not same address with list2

print()

print(v_list6) # --> Not same address with list2
# Import Library

import pandas as pd
# Let's create Data Frame from Dictionary



v_dict1 = { "COUNTRY" : ["TURKEY","RUSSIA","U.S.A","NEW ZELAND","PALESTINIAN"],

            "CAPITAL":["ISTANBUL","MOSCOVA","WASHINGNTON","WELLINGTON","KUDUS"],

            "POPULATION":[13.03,15.08,21.06,19.01,42]}



v_dataFrame1 = pd.DataFrame(v_dict1)



print(v_dataFrame1)

print()

print("Type of v_dataFrame1 is : " , type(v_dataFrame1))
#get top 5 rows



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
v_country1 = v_dataFrame1["COUNTRY"]



print(v_country1)



print()



print("Type of v_country1 is : " , type(v_country1))
v_currenyList1 = ["TRY","RUB","EUR","NZD","ILS"]



v_dataFrame1["CURRENCY"] = v_currenyList1



print(v_dataFrame1.head())
v_AllCapital = v_dataFrame1.loc[:,"CAPITAL"]

print(v_AllCapital)

print()

print("Type of v_AllCapital is : " , type(v_AllCapital))
v_top3Currency = v_dataFrame1.loc[0:2,"CURRENCY"]

print(v_top3Currency)
v_CityCountry = v_dataFrame1.loc[:,["CAPITAL","COUNTRY","Helbet caanığm"]] #---> Helbet caanığmnot defined !!!

print(v_CityCountry)
v_Reverse1 = v_dataFrame1.loc[::-2,:]

print(v_Reverse1)
print(v_dataFrame1.loc[:,:"POPULATION"])

print()

print(v_dataFrame1.loc[:,"POPULATION":])
print(v_dataFrame1.iloc[:,3])
v_filter1 = v_dataFrame1.POPULATION > 4

print(v_filter1)
v_filter2 = v_dataFrame1["POPULATION"] < 20

print(v_filter2)
print(v_dataFrame1[v_filter1 & v_filter2])
print(v_dataFrame1[v_dataFrame1["CURRENCY"] == "ILS"])
v_meanPop =v_dataFrame1["POPULATION"].mean()

print(v_meanPop)



v_meanPopNP = np.mean(v_dataFrame1["POPULATION"])

print(v_meanPopNP)
for a in v_dataFrame1["POPULATION"]:

    print(a)
v_dataFrame1["POP LEVEL"] = ["Low" if v_meanPop > a else "HIGH" for a in v_dataFrame1["POPULATION"]]

print(v_dataFrame1)
print(v_dataFrame1.columns)



v_dataFrame1.columns = [a.lower() for a in v_dataFrame1.columns]



print(v_dataFrame1.columns)

v_dataFrame1.columns = [a.split()[0]+"_"+a.split()[1] if (len(a.split())>2) else a for a in v_dataFrame1.columns]

print(v_dataFrame1.columns)
v_data1 = v_dataFrame1.head()

v_data2 = v_dataFrame1.tail()



print(v_data1)

print()

print(v_data2)

v_dataConcat1 = pd.concat([v_data1,v_data2],axis=0) # axis = 0 --> VERTICAL CONCAT

v_dataConcat2 = pd.concat([v_data2,v_data1],axis=0) # axis = 0 --> VERTICAL CONCAT



print(v_dataConcat1)

print()

print(v_dataConcat2)
v_CAPITAL = v_dataFrame1["capital"]

v_POPULATION = v_dataFrame1["population"]



v_dataConcat3 = pd.concat([v_CAPITAL,v_POPULATION],axis=1) #axis = 1 --> HORIZONTAL CONCAT

v_dataConcat4 = pd.concat([v_POPULATION,v_CAPITAL],axis=1) #axis = 1 --> HORIZONTAL CONCAT

print(v_dataConcat3)

print()

print(v_dataConcat4)

v_dataFrame1["test1"] = [a*2 for a in v_dataFrame1["population"]]

print(v_dataFrame1)
def f_multiply(v_population):

    return v_population*3



v_dataFrame1["test2"] = v_dataFrame1["population"].apply(f_multiply)

print(v_dataFrame1)