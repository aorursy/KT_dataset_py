v_message = "selamun aleyküm " 



print("Hi")
print(v_message)

v_name = "furkan"

v_surname = "BOYACI"

v_fullname = v_name + v_surname

print(v_fullname)
v_fullname = v_name + " " + v_surname



print(v_fullname)
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
v_3ch = v_fullname[11]

print(v_3ch)
v_num1 = "19"

v_num2 = "07"

v_numSum1 = v_num1 + v_num2

print(v_numSum1)
v_fl1 = 1900

v_fl2 = 7

v_s3 = v_fl1 + v_fl2



print(v_s3 , type(v_s3))
def f_SayHello():

    print("Hi. I am coming from sarıyer")

f_SayHello()
def f_sayMessage(v_Message1):

    print(v_Message1 , " came from 'f_sayMessage'")

    

def f_getFullName(v_FirstName , v_Surname , v_Age):

    print("gardaşım hoş geldin" , v_FirstName , " " , v_Surname , " your age : " , v_Age)

f_sayMessage("How are you la gardaşım nasılsın ?")
f_getFullName("furkan" , "BYC" , 15)
def f_Calc1(f_Num1 , f_Num2 , f_Num3):

    v_Sonuc = f_Num1 + f_Num2 + f_Num3

    print("Sonuç = " ," " , v_Sonuc)
f_Calc1(1000 , 900 , 7)
def f_alan(kenar1,kenar2):

    print(kenar1*kenar2)

f_alan(15,15)
l_list1 = [1,9,0,7,8,4]

print(l_list1)

print("Type of 'l_list1' is : " , type(l_list1))
v_list1_4 = l_list1[0]

print(v_list1_4)

print("Type of 'v_list1_4' is : " , type(v_list1_4))
l_list2 = ["1.","2.","3.","4.","5.",]

print(l_list2)

print("Type of 'l_list2' is : " , type(l_list1))
v_list2_4 = l_list2[3]

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
d_dict1 = {"Home":"Ev" , "School" : "Okul" , "Student": "Öğrenci"}



print(d_dict1)

print(type(d_dict1))
v_school = d_dict1["School"]

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
v_var1 = 10

v_var2 = 20



if v_var1 > v_var2:

    print(v_var1 , " is greater then " , v_var2)

elif v_var1 < v_var2:

    print(v_var1 , " is smaller then " , v_var2)

else :

    print("This 2 variables are equal")
# < , <= , > , >= , == , <>

def f_Comparison1(v_Comp1 , v_Comp2):

    if v_Comp1 > v_Comp2:

        print(v_Comp1 , " is greater then " , v_Comp2)

    elif v_Comp1 < v_Comp2:

        print(v_Comp1 , " is smaller then " , v_Comp2)

    else :

        print("These " , v_Comp1 , " variables are equal")

        

f_Comparison1(33,44)

f_Comparison1(66,22)

f_Comparison1(11,11)

# using 'IN' with LIST





def f_IncludeOrNot(v_search, v_searchList):

    if v_search in v_searchList :

        print("Good news ! ",v_search , " is in list.")

    else :

        print(v_search , " is not in list. Sorry :(")



l_list = list(d_dict1.keys())

print(l_list)

print(type(l_list))



f_IncludeOrNot("Home" , l_list)

f_IncludeOrNot("Pencil" , l_list)
for a in range(0,15):

    print("sa " , a)
v_unhappyMessage = "I AM UNHAPPY"

print(v_unhappyMessage)
for v_chrs in v_unhappyMessage:

    print(v_chrs)

    print("------")
for v_chrs in v_unhappyMessage.split():

    print(v_chrs)
print(l_list1)

v_sum_list1 = sum(l_list1)

print("Sum of l_list1 is : " , v_sum_list1)



print()

v_cum_list1 = 0

v_loopindex = 0

for v_current in l_list1:

    v_cum_list1 = v_cum_list1 + v_current

    print(v_loopindex , " nd value is : " , v_current)

    print("Cumulative is : " , v_cum_list1)

    v_loopindex = v_loopindex + 1

    print("------")
i = 0

while(i < 4):

    print("Hi" , i)

    i = i+1
print(l_list1)

print()



i = 0

k = len(l_list1)



while(i<k):

    print(l_list1[i])

    i=i+1
l_list2 = [3,5,7,-6,1907,255,71,34,-1907]



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
# Reshape

v_array3_np = v_array2_np.reshape(3,5)

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

print("---------------")

print("Shape is : " , v_array4_np.shape)
# Zeros

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
# Arrange

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



print(v_np1 + v_np2) #--> We can not addition of 2 arrays

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

v_np5 = np.array([[2,4,8],[3,6,1]])

v_np5Transpose = v_np5.T

print(v_np5)

print(v_np5.shape)

print()

print(v_np5Transpose)

print(v_np5Transpose.shape)