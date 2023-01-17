v_message = "hello world"

print("Hi")
print(v_message)
v_name = "miray"

v_surname = "türkyılmaz"



v_fullname = v_name + v_surname

print(v_fullname)
v_fullname = v_name + " " + v_surname



print(v_fullname)
v_num1 = "150"

v_num2 = "200"

v_numSum1 = v_num1 + v_num2

print(v_numSum1)
#length



v_lenFull = len(v_fullname)

print("v_fullname : " , v_fullname , "and length is :" ,v_lenFull)
v_titleF = v_fullname.title()

print("v_fullname :" , v_fullname , "and title is :" , v_titleF)
#upper  :

v_upperF = v_fullname.upper()



#lower

v_lowerF = v_fullname.lower()

print("v_fullname :" , v_fullname , "Upper :" , v_upperF , "Lower : " , v_lowerF)
v_2ch = v_fullname[9]

print(v_2ch)
v_num1 = 150

v_num2 = 200

v_sum1 = v_num1 + v_num2  



print(v_sum1 , "and type : " , type(v_sum1))
v_num1 = v_num1 + 50

v_num2 = v_num2 - 20.5

v_sum1 = v_num1 + v_num2

print(v_num1)
print("v_sum1 : ",v_sum1 , "type : " , type(v_sum1))
v_fl1 = 25.5

v_fl2 = 15.5

v_s3  = v_fl1 + v_fl2



print(v_s3 , type (v_s3))
def f_SayHello():

    print("Hi.I am coming from f_SayHello")

    

def f_SayHello2():

    print("Hi.I am coming from f_SayHello2")

    print("Good")

    

f_SayHello()
f_SayHello2()


def f_sayMessage(v_Message1):

    print(v_Message1 , " came from 'f_sayMessage'")

    

def f_getFullName(v_FirstName , v_Surname , v_Age):

    print("Welcome " , v_FirstName , " " , v_Surname , " your age : " , v_Age)

f_sayMessage("How are you?")
f_getFullName("Miray","TÜRKYILMAZ",16)
def f_Calc1(f_Num1 , f_Num2 ,f_Num3):

    v_Sonuc = f_Num1 + f_Num2 +f_Num3

    print("Sonuç = "," ",v_Sonuc)
f_Calc1(10,20,30)
#return functıon

def f_Calc2(v_Num1,v_Num2,v_Num3):

    v_Out = v_Num1 + v_Num2 + v_Num3*2

    print("Hi from f_Calc2")

    return v_Out
v_gelen = f_Calc2(1,2,3)

print("Score is :",v_gelen)
#Default functions :

def f_getSchoolInfo(v_Name,v_StudentCount,v_City = "ISTANBUL"):

    print("Name .",v_Name,"St Count:" ,v_StudentCount,"City:",v_City)
f_getSchoolInfo("AAIHL",521)

f_getSchoolInfo("Ankara Fen",521,"ANKARA")
#Flexible Functions :



def f_Flex1(v_Name,*v_messages):

    print("Hi",v_Name,"your first message is:",v_messages[2])
f_Flex1("Miray","Selam","Naber","İyisindir İnşAllah")
#Lambda Function :



v_result =lambda x : x*3

print("Result is :",v_result(6))
def f_alan(kenar1,kenar2):

    print(kenar1*kenar2)
f_alan(5,6)
l_list1 = [1,2,3,4,5,6]

print(l_list1)

print("type of'l_list1' is:",type(l_list1))
v_list1_4 = l_list1[3]

print(v_list1_4)

print("type of 'l_list_4'is:",type(v_list1_4))
l_list2 =["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

print(l_list2)

print("type of 'l_list2' is: ",type(l_list2))
v_list2_4 = l_list2[3]

print(v_list2_4)

print("type of 'v_list2_4' is:",type(v_list2_4))
v_list2_x3 =l_list2[-3]

print(v_list2_x3)
l_list2_2 = l_list2[0:3]

print(l_list2_2)
#Len

v_len_l_list2_2 =len(l_list2_2)

print("size of 'l_list2_2'is:",v_len_l_list2_2)

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

l_list2_2.append("Saturday")

print(l_list2_2)
l_list2_2.remove("Saturday")

print(l_list2_2)
d_dict1 = {"Turkey":"Ankara","Spain":"Madrid","Russia":"Moskow"}

print(d_dict1)

print(type(d_dict1))
v_Spain = d_dict1["Spain"]

print(v_Spain)

print(type(v_Spain))
#Keys & Values



v_keys = d_dict1.keys()

v_values = d_dict1.values()



print(v_keys)

print(type(v_keys))



print(v_values)

print(type(v_values))
v_var1 = 16

v_var2 = 17



if v_var1 > v_var2:

    print(v_var1,"is greater then",v_var2)

elif v_var1 < v_var2:

    print(v_var1,"is smaller then",v_var2)

else:

    print("This 2 variables are equal")

# < , <= , > , >= , == , <>



def f_Comparison1(v_Comp1 , v_Comp2):

    if v_Comp1 > v_Comp2:

        print(v_Comp1 , " is greater then " , v_Comp2)

    elif v_Comp1 < v_Comp2:

        print(v_Comp1 , " is smaller then " , v_Comp2)

    else :

        print("These " , v_Comp1 , " variables are equal")



f_Comparison1(12,13)

f_Comparison1(13,12)

f_Comparison1(10,10)



# using 'IN' with LIST



def f_IncludeOrNot(v_search,v_searchList):

    if v_search in v_searchList :

        print("Good news !",v_search,"is in list.")

    else :

            print(v_search,"is not in list.Sorry :(")

        

        

l_list = list(d_dict1.keys())

print(l_list)

print(type(l_list))



f_IncludeOrNot("Turkey",l_list)

f_IncludeOrNot("Serbia",l_list)

for a in range(1,8):

    print("Studied",a,"hour in the weekend")

    
v_warningMessage = "Warning! You should work harder"

print(v_warningMessage)
for v_chrs in v_warningMessage:

    print(v_chrs)

    print("----")
for v_chrs in v_warningMessage.split():

    print(v_chrs)
print(l_list1)

v_sum_list1 = sum(l_list1)

print("Sum of l_list1 is:",v_sum_list1)



print()

v_cum_list1 = 0

v_loopindex = 0



for v_current in l_list1:

    v_cum_list1 = v_cum_list1 + v_current

    print(v_loopindex , " nd value is : " , v_current)

    print("Cumulative is : " , v_cum_list1)

    v_loopindex = v_loopindex + 1

print("---")

i = 0

while(i < 8):

    print("Hello World",i)

    i = i + 1
print(l_list1)

print()



i = 0

k = len(l_list1)



while(i < k):

    print(l_list1[i])

    i = i + 1
#Let's find minimum and maximum number in list



l_list2 = [1,3,5,8,0,-8,-5,-3,-5]



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

# Import library to use

import numpy as np
v_array1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

v_array2_np = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
print("v_array1 : " , v_array1)

print("Type of v_array1 : " , type(v_array1))
print("v_array2_np : " , v_array2_np)

print("Type of v_array2_np : " , type(v_array2_np))
# shape

v_shape1 = v_array2_np.shape

print("v_shape1 : " , v_shape1 , " and type is : " , type(v_shape1))
#Reshape



v_array3_np = v_array2_np.reshape(3,5)

print(v_array3_np)
v_shape2 = v_array3_np.shape

print("v_shape2 : " , v_shape2 , " and type is : " , type(v_shape2))
#Dimension



v_dimen1 = v_array3_np.ndim

print("v_dimen1 :" , v_dimen1 , "type is :", type(v_dimen1))
# Dtype.name



v_dtype1 = v_array3_np.dtype.name

print("v_dtype1 : " , v_dtype1 , " and type is : " , type(v_dtype1))
# Size



v_size1 = v_array3_np.size

print("v_size1 :","an type :",type(v_size1))
# Let's create 3x4 array



v_array4_np = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

print(v_array4_np)

print("--------------")

print("Shape is : " , v_array4_np.shape)
#Zeros



v_array5_np = np.zeros((3,4))

print(v_array5_np)
# Update an item on this array 



v_array5_np[0,3] = 21

print(v_array5_np)
# Ones



v_array6_np = np.ones((3,4))

print(v_array6_np)
# Empty



v_array7_np = np.empty((2,3))

print(v_array7_np)
#Arrange



v_array8_np = np.arange(10,45,5)

print(v_array8_np)

print(v_array8_np.shape)
# Linspace



v_array9_np = np.linspace(10,30,5)

v_array10_np= np.linspace(10,20,30)



print(v_array9_np)

print(v_array9_np.shape)

print("----------")

print(v_array10_np)

print(v_array10_np.shape)
# Sum , Subtract , Square



v_np1 = np.array([1,2,3])

v_np2 = np.array([7,8,9])



print(v_np1 + v_np2)

print(v_np1 - v_np2)

print(v_np2 - v_np1)

print(v_np1 ** 2)
#Sinus



print(np.sin(v_np2))
#True / False



v_np2_TF = v_np2 < 8

print(v_np2_TF)

print(v_np2_TF.dtype.name)
# Element wise Prodcut



v_np1 = np.array([3,5,9])

v_np2 = np.array([4,7,2])

print(v_np1 * v_np2)
v_np5 = np.array([[3,5,9],[4,7,2]])

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



v_np10 = np.array([1,3,5,7,9])

v_np11 = np.array([10,30,50,70,90])



print(np.add(v_np10,v_np11))
v_np12 = np.array([1,2,3,4,5,6,7,8,9])



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



v_np14 = np.array([[2,4,6],[8,10,12],[14,16,18],[20,22,24]])

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
 #Resize



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
v_list1 = [3,5,8,9]

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
v_list2[0] = 16



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
v_list2[0] = 17



print(v_list2)

print()

print(v_list5) # --> Not same address with list2

print()

print(v_list6) # --> Not same address with list2
# Import Library

import pandas as pd
v_dict1 = {"LESSON":["MATH","BIOLOGY","PHYSICS","CHEMISTRY","BIOLOGY","CHEMISTRY"],"HOUR":[6,2,4,4,2,4],"NOTE":[88.2,92.3,75.4,83.6,92.3,83.6]}



v_dataFrame1 = pd.DataFrame(v_dict1)

print(v_dataFrame1)

print()

print("type of v_dataFrame1 is:",type(v_dataFrame1))
v_head1 = v_dataFrame1.head()

print(v_head1)

print()

print("type of v_head1 is:",type(v_head1))
print(v_dataFrame1.head(100))
v_tail1 = v_dataFrame1.tail()

print(v_tail1)

print()

print("type of v_tail1 is:",type(v_tail1))
v_columns1 = v_dataFrame1.columns

print(v_columns1)

print()

print("type of v_columns1:",type(v_columns1))
v_info1 = v_dataFrame1.info()

print(v_info1)

print()

print("type of v_info1 is:",type(v_info1))
v_dtypes1 = v_dataFrame1.dtypes

print(v_dtypes1)

print()

print("type of v_dtypes is:",type(v_dtypes1))
v_descr1 = v_dataFrame1.describe()

print(v_descr1)

print()

print("type of v_dscr1 is:",type(v_descr1))
v_lesson1 = v_dataFrame1["LESSON"]

print(v_lesson1)

print()

print("type of v_lesson1 is:",type(v_lesson1))
#add new columns

v_topic1 = ["NUMBERS","CELL","VECTORS","ATOMS","COMMUNITY","GASES"]

v_dataFrame1["TOPIC"] = v_topic1

print(v_dataFrame1.head())
v_Allhour = v_dataFrame1.loc[:,"HOUR"]

print(v_Allhour)

print()

print("type of v_Allhour is:",type(v_Allhour))
v_Alltopic = v_dataFrame1.loc[:,"TOPIC"]

print(v_Alltopic)

print()

print("type of v_Alltopic is:",type(v_Alltopic))
v_top3topic = v_dataFrame1.loc[0:3,"TOPIC"]

print(v_top3topic)
v_top1note = v_dataFrame1.loc[0:1,"NOTE"]

print(v_top1note)
v_lessonhour = v_dataFrame1.loc[:,["LESSON","HOUR","UNKNOWN"]] #--> BLABLA not defined !!!

print(v_lessonhour)
v_reserve1 = v_dataFrame1.loc[::-1,:]

print(v_reserve1)
print(v_dataFrame1.loc[:,:"NOTE"])

print()

print(v_dataFrame1.loc[:,"NOTE":])
print(v_dataFrame1.iloc[:,2])
v_filter1 = v_dataFrame1.HOUR > 2

print(v_filter1)
v_filter2 = v_dataFrame1.HOUR < 3

print(v_filter2)
print(v_dataFrame1[v_filter1 & v_filter2])
print(v_dataFrame1[v_dataFrame1["LESSON"]=="HOUR"])
v_meanPop =v_dataFrame1["NOTE"].mean()

print(v_meanPop)



v_meanPopNP = np.mean(v_dataFrame1["NOTE"])

print(v_meanPopNP)

for a in v_dataFrame1["NOTE"]:

    print(a)
v_dataFrame1["POP LEVEL"] = ["Low" if v_meanPop > a else "HIGH" for a in v_dataFrame1["NOTE"]]

print(v_dataFrame1)
print(v_dataFrame1.columns)



v_dataFrame1.columns = [a.lower() for a in v_dataFrame1.columns]



print(v_dataFrame1.columns)
v_dataFrame1.columns = [a.split()[0]+"_"+a.split()[1] if (len(a.split())>1) else a for a in v_dataFrame1.columns]



print(v_dataFrame1.columns)
v_dataFrame1["test1"] = [-1,-2,-3,-4,-5,-6]



print(v_dataFrame1)
v_dataFrame1.drop(["test1"],axis=1,inplace = True) #--> inplace = True must be written

print(v_dataFrame1)
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
v_LESSON = v_dataFrame1["lesson"]

v_HOUR = v_dataFrame1["hour"]



v_dataConcat3 = pd.concat([v_LESSON,v_HOUR],axis=1) #axis = 1 --> HORIZONTAL CONCAT

v_dataConcat4 = pd.concat([v_HOUR,v_LESSON],axis=1) #axis = 1 --> HORIZONTAL CONCAT

print(v_dataConcat3)

print()

print(v_dataConcat4)
v_dataFrame1["test1"] = [a*2 for a in v_dataFrame1["hour"]]

print(v_dataFrame1)
def f_multiply(v_hour):

    return v_hour*3

v_dataFrame1["test2"] = v_dataFrame1["hour"].apply(f_multiply)

print(v_dataFrame1)