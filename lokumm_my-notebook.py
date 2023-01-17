print("acaba oldumu")
v_massage = "hello world"

print("hi")
v_name = "lokum"

v_surname = "ayyıldız"



v_fullname = v_name + v_surname

print(v_fullname)
v_fullname = (v_name +" "+v_surname)



print(v_fullname)
v_num1 = "1005"

v_num2 = "1014"

v_numSun1 = v_num1 +v_num2



print(v_numSun1)
v_lenFull = len(v_fullname)



print("v_fullname : " ,v_fullname, "and lengt is:" ,v_lenFull )


#upper

v_upperF = v_fullname.upper()



#lower

v_lowerF = v_name.lower()

print("v_fullname : " , v_upperF ,"lower : " , v_lowerF)
v_titlef = v_fullname.title()

print("v_fullname : ", " and title is : ",v_titlef)
v_num1 = 1015

v_num2 = 1004

v_sum1 = v_num1 + v_num2



print(v_sum1 , "and type : " , type(v_sum1))
v_fl1 = 55.9

v_fl2 = 20.5

v_s3 = v_fl1 + v_fl2



print(v_s3 , type (v_s3))
def f_SayHello():

    print("Hi. I am coming from f_SayHello")

    

def f_SayHello2():

    print("Hi. I am coming from from f_SayHello2")

    print("Good")

    

f_SayHello()
f_SayHello2()
def f_sayMassage(v_Massage1):

    print(v_Massage1 , " came from 'f_sayMassage'")

    

def f_getFullName(v_FirstName , v_Surname , v_Age):

    print("wellcome" , v_FirstName , " " , v_Surname , "your age : " , v_Age)
f_getFullName("lokum" , "ayyıldız" , 13)
def f_Calc1(f_Num1 , f_Num2 , f_Num3):

    v_cevap = f_Num1 + f_Num2 + f_Num3

    print("cevap = " ," " , v_cevap)

    
f_Calc1(123 , 456 , 890)
# return function

def f_Calc2(v_Num1 , v_Num2 , v_Num3):

    v_Out = v_Num1+v_Num2+v_Num3*2

    print("Hi from f_Calc2")

    return v_Out
v_gelen =  f_Calc2(4,7,9)

print("Score is : " , v_gelen)
# Default Functions :

def f_getSchoolInfo(v_Name,v_StudentCount,v_City = "ISTANBUL"):

    print("Name : " , v_Name , " St Count : " , v_StudentCount

          , " City : " , v_City)
f_getSchoolInfo("AAIHL" , 353)

f_getSchoolInfo("RIL" , 353 , "RİZE")
# Flexible Functions :



def f_Flex1(v_Name , *v_messages):

    print("Hi " , v_Name , " your first message is : " , v_messages[2])

f_Flex1("Lokum" , "Nbr" , "Aramıyosun?" , "Geçmişolsun")
# Lambda Function :



v_result1 = lambda x : x*5

print("Result is : " , v_result1(401))
def f_alan(kenar1,kenar2):

    print(kenar1*kenar2)
f_alan(6,9)
l_list1 = [1,3,0,2,4,7]

print(l_list1)

print("Type of 'l_list1' is : " , type(l_list1))
v_list1_4 = l_list1[0]

print(v_list1_4)

print("Type of 'v_list1_4' is : " , type(v_list1_4))
l_list2 = ["gün1","gün2","gün3","gün4","gün5"]

print(l_list2)

print("Type of 'l_list2' is : " , type(l_list1))
v_list2_4 = l_list2[3]

print(v_list2_4)

print("Type of 'v_list2_4' is : " , type(v_list2_4))
v_list2_x3 = l_list2[-3]

print(v_list2_x3)
l_list2_2 = l_list2[0:3]

print(l_list2_2)
#Len

v_len_l_list2_2 = len(l_list2_2)

print("Size of 'l_list2_2' is : ",v_len_l_list2_2)

print(l_list2_2)
#Append

l_list2_2.append("gün9")

print(l_list2_2)



l_list2_2.append("gün95")

print(l_list2_2)
#Reverse

l_list2_2.reverse()

print(l_list2_2)
#Sort

l_list2_2.sort()

print(l_list2_2)
#Remove



#First add 'Saturday' then Remove 'Saturday'

l_list2_2.append("gün53")

print(l_list2_2)
d_dict1 = {"Tower":"Kule" , "msuıo" : "Okul" , "Worker": "Çalışan"}



print(d_dict1)

print(type(d_dict1))
v_school = d_dict1["Tower"]

print(v_school)

print(type(v_school))
#Keys & Values



v_keys = d_dict1.keys()

v_values = d_dict1.values()





print(v_keys)

print(type(v_keys))



print()

print(v_values)

print(type(v_values))
v_var1 = 2394823

v_var2 = 1238967



if v_var1 > v_var2:

    print(v_var1 , "bu büük bundan " , v_var2)

elif v_var1 < v_var2:

    print(v_var1 , "bu güccük bundan " , v_var2)

else : 

    print("bu 2 degişken")
# < , <= , > , >= , == , <>

def f_Comparison1(v_Comp1 , v_Comp2):

    if v_Comp1 > v_Comp2:

        print(v_Comp1 , " bu bundan büük " , v_Comp2)

    elif v_Comp1 < v_Comp2:

        print(v_Comp1 , " bu bundan küçük " , v_Comp2)

    else :

        print("These " , v_Comp1 , " bir birilerine eşitler")

        

f_Comparison1(53,11)

f_Comparison1(61,99)

f_Comparison1(11,11)
# using 'IN' with LIST





def f_IncludeOrNot(v_search, v_searchList):

    if v_search in v_searchList :

        print("Buldum!! ",v_search , "is in list.")

    else :

        print(v_search , " özür dileirm bulamdım yaaa :(")

        

l_list = list(d_dict1.keys()) 

print(l_list)

print(type(l_list))



f_IncludeOrNot("Tower" , l_list)

f_IncludeOrNot("uçak" , l_list)
for a in range(0,54):

    print("Denek " , a)

v_happyMessage = "BEN RIZE LIYIM"

print(v_happyMessage)
for v_chrs in v_happyMessage:

    print(v_chrs)

    print("********************")
for v_chrs in v_happyMessage.split():

    print(v_chrs)
i = 0

while(i < 20):

    print("selam denek" , i)

    i = i+1
l_list1 = ("1","4","5","3")
print(l_list1)

print()



i = 0

k = len(l_list1)



while(i<k):

    print(l_list1[i])

    i=i+1
#Let's find minimum and maximum number in list



l_list2 = [9,1,4,-2,-912903801938,135,912389123347,910923819,-189000]



v_min = 0

v_max = 0



v_index = 0

v_len = len(l_list2)



while (v_index < v_len):

    v_current = l_list2[v_index]

    

    if v_current > v_max:

        v_max = v_current

    

    if v_current < v_min:

        v_min = v_current

    

    v_index = v_index+1



print ("Maximum number is : " , v_max)

print ("Minimum number is : " , v_min)
# Import libary to use

import numpy as np 
v_array1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

v_array2_np = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
print("v_array1 : " , v_array1)

print("Type of v_array1 : " , type(v_array1))
print("v_array2_np : " , v_array2_np)

print("Type of v_array2_np : " , type(v_array2_np))
# shape

v_shape1 = v_array2_np.shape

print("v_shape : " , v_shape1 , " and type is : " , type(v_shape1))
# Reshape

v_array3_np = v_array2_np.reshape(5,4)

print(v_array3_np)
# Dimension

v_dimen1 = v_array3_np.ndim

print("v_dimen1 : " , v_dimen1 , " type is : " , type(v_dimen1))
# Size

v_size1 = v_array3_np.size

print("v_size1 : " , v_size1 , " and type : " , type(v_size1))
# Let's create 3x4 array

v_array4_np = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]])

print(v_array4_np)

print("---------------")

print("Shape is : " , v_array4_np.shape)
# Zeros

v_array5_np = np.zeros((10,5))

print(v_array5_np)
# Update an item on this array 

v_array5_np[9,4] = 99

print(v_array5_np)
# Ones

v_array6_np = np.ones((99,99))

print(v_array6_np)
# Empty

v_array7_np = np.empty((3,2))

print(v_array7_np)
# Arrange

v_array8_np = np.arange(10,45,5)

print(v_array8_np)

print(v_array8_np.shape)
# Linspace

v_array9_np = np.linspace(10,30,5)

v_array10_np = np.linspace(10,40,30)



print(v_array9_np)

print(v_array9_np.shape)

print("-----------------------")

print(v_array10_np)

print(v_array10_np.shape)
# Sum , Subtract , Square

v_np1 = np.array([4,7,9])

v_np2 = np.array([5,7,8])



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

v_np1 = np.array([5,6,7])

v_np2 = np.array([8,9,11])

print(v_np1 * v_np2)
# Transpose

v_np5 = np.array([[1,2,5],[9,5,3]])

v_np5Transpose = v_np5.T

print(v_np5)

print(v_np5.shape)

print()

print(v_np5Transpose)

print(v_np5Transpose.shape)
# Matrix Multiplication

v_np6 = v_np5.dot(v_np5Transpose)

print(v_np6)
# Exponential --> We will use on Statistics Lesson

v_np5Exp = np.exp(v_np5)



print(v_np5)

print(v_np5Exp)
# Random 

v_np8 = np.random.random((6,6)) # --> It will get between 0 and 1 random numbers

print(v_np8)
#Sum , Max ,Min

v_np8Sum = v_np8.sum()

print("Sum of array : ", v_np8Sum)  #--> Remember ! If you get sum of array we can use that :  sum(array1)

print("Max of array : ", v_np8.max()) #--> Remember ! If you get max of array we can use that :  max(array1)

print("Min of array : ", v_np8.min()) #--> Remember ! If you get min of array we can use that :  min(array1)
# Sum with Column or Row

print("Sum of Columns :")

print(v_np8.sum(axis=0)) # --> Sum of Columns

print()

print("Sum of Rows :")

print(v_np8.sum(axis=1)) #Sum of Rows
# Square , Sqrt

print(np.sqrt(v_np8))

print()

print(np.square(v_np8))
# Add

v_np10 = np.array([1,2,3,4,5])

v_np11 = np.array([52,51,50,49,48])



print(np.add(v_np10,v_np11))
v_np12 = np.array([4,3,5,8,5,6,7,8,8])



print("First item is : " , v_np12[8])

print("Third item is : " , v_np12[1])
# Get top 4 rows :

print(v_np12[0:4])

# Reverse

v_np12_Rev = v_np12[::-1]

print(v_np12_Rev)
v_np13 = np.array([[5,4,3,2,1],[15,14,153,16,15]])

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

v_np14 = np.array([[53,34,61],[1,6,45],[14,51,32],[33,19,13]])

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
v_np20 = np.array([[9,8],[7,6],[5,4]])



print(v_np20)

print()

print(v_np20.reshape(2,3))



print()

print(v_np20) #--> It has not changed !!
# Resize

v_np20.resize((2,3))

print(v_np20) # --> Now it changed !  Resize can change its shape
v_np21 = np.array([[33,71],[37,43]])

v_np22 = np.array([[13,64],[71,83]])



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
v_list1 = [3,4,9,8]

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
v_list2[0] = 55



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
v_list2[0] = 71



print(v_list2)

print()

print(v_list5) # --> Not same address with list2

print()

print(v_list6) # --> Not same address with list2