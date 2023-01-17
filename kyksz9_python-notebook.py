print('HELLO WORLD')
r_message = 'hello world'



print('hi')
print(r_message)
r_name = 'rumeysa'

r_surname = 'kaykısız'

 

r_fullname = r_name + r_surname 

print(r_fullname)
r_fullname = r_name +' '+ r_surname 

print(r_fullname)
r_num1 = '100'

r_num2 = '200'

r_numSum1 = r_num1 + r_num2

print(r_numSum1)
#LENGHT

r_lenFull = len(r_fullname)

print('r_fullname :',r_fullname,'and lenght is :',r_lenFull)
r_titleF = r_fullname.title()

print('r_fullname:',r_fullname,'and title is :',r_titleF)
#UPPER:

r_upperF = r_fullname.upper()



#LOWER

r_lowerF = r_fullname.lower()

print('r_fullname:',r_fullname,'upper :',r_upperF,'lower :',r_lowerF)

r_2ch = r_fullname[11]

print(r_2ch)
r_num1 = 100

r_num2 = 200

r_sum1 = r_num1 + r_num2  



print(r_sum1 , "and type :" ,type (r_sum1))
r_num1 = r_num1 + 60

r_num2 = r_num2 - 25.5

r_sum1 = r_num1 + r_num2



print(r_num1)
print("r_sum1 : ",r_sum1,'type :',type(r_sum1))
r_fl1 = 25.5

r_fl2 = 15.5

r_s3 = r_fl1 + r_fl2



print(r_s3,type(r_s3))
def f_SayHello():

    print('Hi,I am coming from f_SayHello')

    

def f_SayHello2():

    print('Hi,I am coming from f_SayHello2')

    print('Good')

    

f_SayHello()    

        
f_SayHello2()
def f_sayMessage(r_Message1) :

    print (r_Message1 ,"came from 'f_sayMessage'")

    

def f_getFullname (r_FirstName,r_Surname,r_Age):

    print ("Welcome",r_FirstName," ",r_Surname,"your age:",r_Age)
f_sayMessage('How are you?')
f_getFullname("Rumeysa","KAYKISIZ",16)
def f_Calc1(f_Num1,f_Num2,f_Num3):

    r_Sonuc = f_Num1 + f_Num2 + f_Num3

    print('sonuç=',' ',r_Sonuc)
f_Calc1(100,250,50)
# return functıon

def f_Calc2(r_Num1 , r_Num2,r_Num3):

    r_Out = r_Num1 + r_Num2 + r_Num3 * 2

    print('Hi from f_Calc2')

    return r_Out
r_gelen = f_Calc2(1,2,3)

print('Score is :',r_gelen)
#Default Functions :

def f_getSchoolInfo(r_Name , r_StudentCount , r_City = "ISTANBUL"):

    print("Name :", r_Name , "St Count:",r_StudentCount ,"City:" , r_City)
f_getSchoolInfo("AAIHL",521)

f_getSchoolInfo("Ankara Fen",521,"ANKARA")
#Flexible Function:

def f_Flex1(r_Name , *r_messages):

    print("Hi",r_Name , "your first message is:",r_messages[2])
f_Flex1("Rumeysa", "Selam", "Naber","İyisindir İnşAllah")
#Lambda Function:



r_result1 = lambda x:x*3

print("Result is :", r_result1(6))
def f_alan(kenar1 ,kenar2):

    print(kenar1 * kenar2)
f_alan(6,8)
l_list1=[1,5,3,6,8,9]

print(l_list1)

print("type of'l_list1'is:",type(l_list1))
l_list2=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

print(l_list2)

print("type of'l_list2'is:",type(l_list1))
r_list2_4= l_list2[3]

print(r_list2_4)

print("Type of'r_list2_4'is:",type(r_list2_4))
r_list2_x3 = l_list2[-3]

print (r_list2_x3)
l_list2_2=l_list2[0:3]

print(l_list2_2)

#Len



r_len_l_list2_2=len(l_list2_2)

print("Size of 'l_list2_2'is:",r_len_l_list2_2)

print(l_list2_2)
#Append

l_list2_2.append("Saturday")

print(l_list2_2)



l_list2_2.append("Tuesday")

print(l_list2_2)
#Reverse

l_list2_2.reverse()

print(l_list2_2)
#Sort

l_list2_2.sort()

print(l_list2_2)
#Remove



#First add 'Saturday' then Remove 'Saturday'

l_list2_2.append("Saturday")

print(l_list2_2)
l_list2_2.remove("Saturday")

print(l_list2_2)
d_dict1={"Şinasi":"Şair Evlenmesi","Mehmet Rauf":"Eylül","Namık Kemal":"İntibah"}



print(d_dict1)

print(type(d_dict1))
r_Şinasi=d_dict1["Şinasi"]

print(r_Şinasi)

print(type(r_Şinasi))




r_keys=d_dict1.keys()

r_values=d_dict1.values()



print(r_keys)

print(r_values)

print(type(r_values))
r_var1=15

r_var2=20



if r_var1 > r_var2:

    print(r_var1,"is greater then",r_var2)

elif r_var1 < r_var2:

    print(r_var1,"is smaller then",r_var2)

else:

    print("This 2 variables are equal")

    
def f_comparison1(r_comp1,r_comp2):

    if r_comp1>r_comp2:

        print(r_comp1,"is greater then",r_comp2)

    elif r_comp1 < r_comp2:

        print(r_comp1,"is smaller then",r_comp2)

    else:

        print("These",r_comp1,"variablesare equal")

        

f_comparison1(25,42)

f_comparison1(85,62)

f_comparison1(8,8)








def f_IncludeOrNot(r_search,r_searchlist):

    if r_search in r_searchlist:

        print("Good news!",r_search ,"is in list.")

    else :

        print(r_search ,"is not in list.Sorry :(")

        

l_list = list (d_dict1. keys()) 

print(l_list)

print(type(l_list))



f_IncludeOrNot ("Şinasi",l_list)

f_IncludeOrNot ("Mevlana",l_list)



  
for a in range(10,20):

    print("Hi" , a)

   
r_happyMessage = "GOOD MORNING"

print(r_happyMessage)



for  r_chrs in r_happyMessage :

    print(r_chrs)

    print("-------")
for r_chrs in r_happyMessage.split():

    print(r_chrs)
print(l_list1)

r_sum_list1 = sum(l_list1)

print("Sum of l_list1 is : " , r_sum_list1)



print()

r_cum_list1 = 0

r_loopindex = 0

for r_current in l_list1:

    r_cum_list1 = r_cum_list1 + r_current

    print(r_loopindex , " nd value is : " , r_current)

    print("Cumulative is : " , r_cum_list1)

    r_loopindex = r_loopindex + 1

    print("------")

i=0

while(i<4):

    print("Hi", i)

    i=i+1
print(l_list1)

print()



i=0

k=len(l_list1)



while(i<k):

    print(l_list1[i])

    i=i+1

#Let's find minimum and maximum number in list



l_list2=[0,5,-6,-698,-45,56,-56,-9,7]



r_min=0

r_max=0



r_index=0

r_len=len(l_list2)



while(r_index<r_len):

    r_current=l_list2[r_index]

    

    if r_current > r_max:

        r_max=r_current

        

    if r_current<r_min:

        r_min=r_current

        

    r_index=r_index+1

    

print("Maximum number is",r_max)

print("Minimum number is",r_min)

        

   
# Import library to use

import numpy as np
r_array1=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

r_array2_np=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
print("r_array1:",r_array1)

print("Type of r_array1:",type(r_array1))
print("r_array2_np :",r_array2_np)

print("Type of r_array2_np:",type(r_array2_np))
#Shape

r_shape1=r_array2_np.shape

print("r_shape1:",r_shape1,"and type is:",type(r_shape1))
#Reshape

r_array3_np=r_array2_np.reshape(3,5)

print(r_array3_np)

r_shape2=r_array3_np.shape

print("r_shape2:",r_shape2,"and type is:",type(r_shape2))
#Dimension

r_dimen1=r_array3_np.ndim

print("r_dimen1:",r_dimen1," type is:",type(r_dimen1))
#Dtype.name

r_dtype1=r_array3_np.dtype.name

print("r_dtype1:",r_dtype1,"and type is:",type(r_dtype1))
#Size

r_size1=r_array3_np.size

print("r_size1:",r_size1,"type is:",type(r_size1))
# Let's create 3x4 array

r_array4_np= np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

print(r_array4_np)

print("---------------")

print("Shape is:",r_array4_np.shape)
#Zeros

r_array5_np=np.zeros((3,5))

print(r_array5_np)
# Update an item on this array 

r_array5_np[0,2]=21

print(r_array5_np)
#Ones

r_array6_np=np.ones((3,3))

print(r_array6_np)
#Empty

r_array7_np=np.empty((2,3))

print(r_array7_np)
#Arrange

v_array8_np = np.arange(10,45,5)

print(v_array8_np)

print(v_array8_np.shape)
# Linspace

v_array9_np = np.linspace(10,30,5)

v_array10_np = np.linspace(10,30,20)



print(v_array9_np)

print(v_array9_np.shape)

print("-----------------------")

print(v_array10_np)

print(v_array10_np.shape)
# Sum , Subtract , Square

v_np1 = np.array([1,2,3])

v_np2 = np.array([7,8,9])



print(v_np1 + v_np2)

print(v_np1 - v_np2)

print(v_np2 - v_np1)

print(v_np1 ** 2)
# Sinus

print(np.sin(v_np2))
# True / False

v_np2_TF = v_np2 < 8

print(v_np2_TF)

print(v_np2_TF.dtype.name)
# Element wise Prodcut

v_np1 = np.array([1,2,3])

v_np2 = np.array([7,8,9])

print(v_np1 * v_np2)
# Transpose



r_np5=np.array([[2,4,8],[3,6,1]])

r_np5Transpose=r_np5 .T

print(r_np5)

print(r_np5.shape)

print()

print(r_np5Transpose)

print(r_np5Transpose.shape)
# Matrix Multiplication



r_np6=r_np5.dot(r_np5Transpose)

print(r_np6)
# Exponential --> We will use on Statistics Lesson

r_np5Exp=np.exp(r_np5)



print(r_np5)

print(r_np5Exp)
# Random

r_np8=np.random.random((6,6)) # --> It will get between 0 and 1 random numbers

print(r_np8)
#Sum , Max ,Min

r_np8Sum = r_np8.sum()

print("Sum of array : ", r_np8Sum)  #--> Remember ! If you get sum of array we can use that :  sum(array1)

print("Max of array : ", r_np8.max()) #--> Remember ! If you get max of array we can use that :  max(array1)

print("Min of array : ", r_np8.min()) #--> Remember ! If you get min of array we can use that :  min(array1)
# Sum with Column or Row

print("Sum of Columns :")

print(r_np8.sum(axis=0)) # --> Sum of Columns

print()

print("Sum of Rows :")

print(r_np8.sum(axis=1)) #Sum of Rows
# Square , Sqrt

print(np.sqrt(r_np8))

print()

print(np.square(r_np8))
# Add

v_np10 = np.array([1,2,3,4,5])

v_np11 = np.array([10,20,30,40,50])



print(np.add(v_np10,v_np11))
v_np12 = np.array([9,8,7,6,5,4,3,2,1])



print("First item is : " , v_np12[0])

print("Third item is : " , v_np12[2])
# Get top 4 rows :



print(v_np12[0:4])
# Reverse



v_np12_Rev = v_np12[::-1]

print(v_np12_Rev)
v_np13 = np.array([[1,2,3,4,5],[11,12,13,14,15]])

print(v_np13)

print()

print(v_np13[1,3]) #--> Get a row



print()

v_np13[1,3] = 314 #--> Update a row

print(v_np13)
# Get all rows but 3rd columns :



print(v_np13[:,2])
#Get 2nd row but 2,3,4th columns



print(v_np13[1,1:4])
# Get last row all columns



print(v_np13[-1,:])
# Get last columns but all rows

print(v_np13[:,-1])
#Flatten



v_np14 = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])

v_np15 = v_np14.ravel()



print(v_np14)

print("Shape of v_np14 is : " ,v_np14.shape)

print()

print(v_np15)

print("Shape of v_np15 is : " ,v_np15.shape)

print()
# Reshape



v_np16 = v_np15.reshape(3,4)

print(v_np16)

print("Shape of v_np16 is : " ,v_np16.shape)
v_np17 = v_np16.T

print(v_np17)

print("Shape of v_np17 is : " ,v_np17.shape)
v_np20 = np.array([[1,2],[3,4],[5,6]])



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
v_list1 = [21,5,28,4]

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
v_list2[0] = 12



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
v_list2[0] = 13



print(v_list2)

print()

print(v_list5) # --> Not same address with list2

print()

print(v_list6) # --> Not same address with list2
# Import Library

import pandas as pd
r_dict1 = { "COUNTRY" : ["TURKEY","U.K.","GERMANY","FRANCE","U.S.A","AZERBAIJAN","IRAN"],

            "CAPITAL":["ISTANBUL","LONDON","BERLIN","PARIS","NEW YORK","BAKU","TAHRAN"],

            "POPULATION":[15.07,8.13,3.57,2.12,8.62,4.3,8.69]}



r_dataFrame1 = pd.DataFrame(r_dict1)



print(r_dataFrame1)

print()

print("Type of r_dataFrame1 is : " , type(r_dataFrame1))
r_head1 = v_dataFrame1.head()

print(r_head1)

print()

print("Type of r_head1 is :" ,type(r_head1))

print(r_dataFrame1.head(100))
r_tail1 = r_dataFrame1.tail()

print(r_tail1)

print()

print("Type of r_tail1 is :" ,type(r_tail1))
r_columns1 = r_dataFrame1.columns

print(r_columns1)

print()

print("Type of r_columns is : " , type(r_columns1))
r_info1 = r_dataFrame1.info()

print(r_info1)

print()

print("Type of r_info1 is : " , type(r_info1))
r_dtypes1 = r_dataFrame1.dtypes

print(r_dtypes1)

print()

print("Type of r_dtypes1 is : " , type(r_dtypes1))

r_descr1 = r_dataFrame1.describe()

print(r_descr1)

print()

print("Type of r_descr1 is : " , type(r_descr1))

r_country1 = r_dataFrame1["COUNTRY"]

print(r_country1)

print()

print("Type of r_country1 is : " , type(r_country1))
r_currenyList1 = ["TRY","GBP","EUR","EUR","USD","AZN","IRR"]

r_dataFrame1["CURRENCY"] = r_currenyList1



print(r_dataFrame1.head())
r_AllCapital = r_dataFrame1.loc[:,"CAPITAL"]

print(r_AllCapital)

print()

print("Type of v_AllCapital is : " , type(r_AllCapital))
r_top3Currency = r_dataFrame1.loc[0:3,"CURRENCY"]

print(r_top3Currency)
r_CityCountry = r_dataFrame1.loc[:,["CAPITAL","COUNTRY","BLABLA"]] #--> BLABLA not defined !!!

print(r_CityCountry)
r_Reverse1 = r_dataFrame1.loc[::-1,:]

print(r_Reverse1)
print(r_dataFrame1.loc[:,:"POPULATION"])

print()

print(r_dataFrame1.loc[:,"POPULATION":])
print(r_dataFrame1.iloc[:,2])
r_filter1 = r_dataFrame1.POPULATION > 4

print(r_filter1)

r_filter2 = r_dataFrame1["POPULATION"] < 9

print(r_filter2)
print(r_dataFrame1[r_filter1 & r_filter2])
print(r_dataFrame1[r_dataFrame1["CURRENCY"] == "EUR"])
r_meanPop =r_dataFrame1["POPULATION"].mean()

print(r_meanPop)



r_meanPopNP = np.mean(r_dataFrame1["POPULATION"])

print(r_meanPopNP)
for a in r_dataFrame1["POPULATION"]:

    print(a)
r_dataFrame1["POP LEVEL"] = ["Low" if r_meanPop > a else "HIGH" for a in r_dataFrame1["POPULATION"]]

print(r_dataFrame1)
print(r_dataFrame1.columns)



r_dataFrame1.columns = [a.lower() for a in r_dataFrame1.columns]



print(r_dataFrame1.columns)
r_dataFrame1.columns = [a.split()[0]+"_"+a.split()[1] if (len(a.split())>1) else a for a in r_dataFrame1.columns]

print(r_dataFrame1.columns)
r_dataFrame1["test1"] = [-1,-2,-3,-4,-5,-6,-7]



print(r_dataFrame1)
r_dataFrame1.drop(["test1"],axis=1,inplace = True) #--> inplace = True must be written

print(r_dataFrame1)
r_data1 = r_dataFrame1.head()

r_data2 = r_dataFrame1.tail()



print(r_data1)

print()

print(r_data2)
r_dataConcat1 = pd.concat([r_data1,r_data2],axis=0) # axis = 0 --> VERTICAL CONCAT

r_dataConcat2 = pd.concat([r_data2,r_data1],axis=0) # axis = 0 --> VERTICAL CONCAT



print(r_dataConcat1)

print()

print(r_dataConcat2)
r_CAPITAL = r_dataFrame1["capital"]

r_POPULATION = r_dataFrame1["population"]



r_dataConcat3 = pd.concat([r_CAPITAL,r_POPULATION],axis=1) #axis = 1 --> HORIZONTAL CONCAT

r_dataConcat4 = pd.concat([r_POPULATION,r_CAPITAL],axis=1) #axis = 1 --> HORIZONTAL CONCAT

print(r_dataConcat3)

print()

print(r_dataConcat4)
r_dataFrame1["test1"] = [a*2 for a in r_dataFrame1["population"]]

print(r_dataFrame1)
def f_multiply(r_population):

    return r_population*3



r_dataFrame1["test2"] = r_dataFrame1["population"].apply(f_multiply)

print(r_dataFrame1)