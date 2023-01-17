v_message = "Hello"



v_name = "gazi"

v_surname = "erdogan"

v_fullname = v_name + " " + v_surname



v_var1 = "100"

v_var2 = "200"

v_varSum = v_var1 + v_var2



print(v_message)



print()

print(v_fullname)



print()

print("v_varSum : " , v_varSum)
#len function :

print("Length of v_fullname is " , len(v_fullname))



#title function :

print()

print("Full Name is " , v_fullname.title())



#upper function :

print()

print("Upper of Full Name is " , v_fullname.upper())



#lower function :

print()

print("Full Name is " , v_fullname.lower())



#type

print()

print("Type of v_fullname is " , type(v_fullname))
v_chr1 = v_fullname[1]

v_chr2 = v_fullname[8]



print("v_chr1 : " , v_chr1 , " and v_chr2 : " , v_chr2)
# Integer

v_num1 = 30

v_num2 = 40

v_numSum = v_num1 + v_num2



print("v_num1 : " , v_num1 , " and type : " , type(v_num1))



print()

print("Sum of Num1 and Num2 is : " , v_numSum , " and type : " , type(v_numSum))
# Float

v_num3 = 30.5

v_numSum2 = v_num3 + v_num2



print("v_num3 : " , v_num3 , " and type : " , type(v_num3))



print()

print("Sum of Num2 and Num3 is : " , v_numSum2 , " and type : " , type(v_numSum2))
def f_SayHello():

    print("Hello")

    

f_SayHello()
def f_SayMessage(v_message):

    print(v_message)

    

f_SayMessage("I am Gazi")
def f_Sum1(v_num1 , v_num2):

    v_sum1 = v_num1 + v_num2

    print(v_num1 , " + " , v_num2, " = " , v_sum1)

    

f_Sum1(10,20)
#Let's create calculator of circle circumreference function



#Functions can give a value with RETURN



def f_CircleCircumreference(v_Radius):

    v_CircumRefenece = 2 * 3.14 * v_Radius

    return v_CircumRefenece



v_Circle1 = f_CircleCircumreference(2)

print("Reference is : " , v_Circle1)
# Default Functions :



def f_Students1(v_Name , v_Surname , v_ShoeSize = 42):

    print("Shoe size of ",v_Name , " " , v_Surname , " is : " , v_ShoeSize)

    

f_Students1("Gazi","Erdogan")

f_Students1("Muhammed","Dugun",43)
#Flexible Functions :



def f_SayMessage2(v_Name , *v_args):

    print("Hi ", v_Name , " Your 2nd message is : " , v_args[1])



f_SayMessage2("Gazi" , "Message 1", "Message 2", "Message 3", "Message 4")
# Lambda Function :



v_result1 = lambda x : x*3

print("Result is : " , v_result1(6))
print("Type of 'f_Students1' is : " , type(f_Students1))

print("Type of 'v_result1' is : " , type(v_result1))
l_list1 = [1,5,3,6,8,9]

print(l_list1)

print("Type of 'l_list1' is : " , type(l_list1))
v_list1_4 = l_list1[3]

print(v_list1_4)

print("Type of 'v_list1_4' is : " , type(v_list1_4))
l_list2 = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

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


for a in range(0,10):

    print("Hi " , a)


v_happyMessage = "I AM HAPPY"

print(v_happyMessage)
for v_chrs in v_happyMessage:

    print(v_chrs)

    print("------")
for v_chrs in v_happyMessage.split():

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
#Let's find minimum and maximum number in list



l_list2 = [3,5,7,-6,-100,255,71,34,-85]



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

v_np11 = np.array([10,20,30,40,50])



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
v_list1 = [1,2,3,4]

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
# Import Library

import pandas as pd
# Let's create Data Frame from Dictionary

v_dict1 = { "COUNTRY" : ["TURKEY","U.K.","GERMANY","FRANCE","U.S.A","AZERBAIJAN","IRAN"],

            "CAPITAL":["ISTANBUL","LONDON","BERLIN","PARIS","NEW YORK","BAKU","TAHRAN"],

            "POPULATION":[15.07,8.13,3.57,2.12,8.62,4.3,8.69]}



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
v_country1 = v_dataFrame1["COUNTRY"]

print(v_country1)

print()

print("Type of v_country1 is : " , type(v_country1))
# Add new column

v_currenyList1 = ["TRY","GBP","EUR","EUR","USD","AZN","IRR"]

v_dataFrame1["CURRENCY"] = v_currenyList1



print(v_dataFrame1.head())
# Get all rows ,  1 column

v_AllCapital = v_dataFrame1.loc[:,"CAPITAL"]

print(v_AllCapital)

print()

print("Type of v_AllCapital is : " , type(v_AllCapital))
# Get top 3 rows of Currency

v_top3Currency = v_dataFrame1.loc[0:3,"CURRENCY"]

print(v_top3Currency)
v_CityCountry = v_dataFrame1.loc[:,["CAPITAL","COUNTRY","BLABLA"]] #--> BLABLA not defined !!!

print(v_CityCountry)
v_Reverse1 = v_dataFrame1.loc[::-1,:]

print(v_Reverse1)
print(v_dataFrame1.loc[:,:"POPULATION"])

print()

print(v_dataFrame1.loc[:,"POPULATION":])
#Get data with column index (not column name)

print(v_dataFrame1.iloc[:,2])
v_filter1 = v_dataFrame1.POPULATION > 4

print(v_filter1)
v_filter2 = v_dataFrame1["POPULATION"] < 9

print(v_filter2)
print(v_dataFrame1[v_filter1 & v_filter2])
print(v_dataFrame1[v_dataFrame1["CURRENCY"] == "EUR"])
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
v_dataFrame1.columns = [a.split()[0]+"_"+a.split()[1] if (len(a.split())>1) else a for a in v_dataFrame1.columns]

print(v_dataFrame1.columns)
v_dataFrame1["test1"] = [-1,-2,-3,-4,-5,-6,-7]

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
v_dataFrame2 = pd.read_csv("../input/iris/Iris.csv")



print(v_dataFrame2.info())
v_SpeciesUnique = v_dataFrame2["Species"].unique()

print(v_SpeciesUnique)

print(type(v_SpeciesUnique))
print(v_dataFrame2.describe())
v_Setosa = v_dataFrame2[v_dataFrame2["Species"]=="Iris-setosa"]

v_Versicolor = v_dataFrame2[v_dataFrame2["Species"]=="Iris-versicolor"]

v_Virginica = v_dataFrame2[v_dataFrame2["Species"]=="Iris-virginica"]



print(v_Setosa.describe())

print(v_Versicolor.describe())
# We must import the library

import matplotlib.pyplot as plt
print(v_dataFrame2.head())
v_dataFrame2.drop(["Id"], axis=1,inplace=True)

print(v_dataFrame2.head())
v_dataFrame2.plot()

plt.show()
plt.plot(v_Setosa.index, v_Setosa.PetalLengthCm, color ="red",label ="setosa - PetalLegtnCm")

plt.xlabel("Index")

plt.ylabel("PetalLegtnCm")

plt.legend() # --> It will add Label into graphic

plt.show()
plt.plot(v_Virginica.index, v_Virginica.PetalLengthCm, color ="green",label ="virginica - PetalLegtnCm")

plt.xlabel("Index")

plt.ylabel("PetalLegtnCm")

plt.legend() # --> It will add Label into graphic

plt.show()
plt.plot(v_Setosa.index, v_Setosa.PetalLengthCm, color ="red",label ="setosa")

plt.plot(v_Versicolor.index, v_Versicolor.PetalLengthCm, color ="blue",label ="versicolor")

plt.plot(v_Virginica.index, v_Virginica.PetalLengthCm, color ="green",label ="virginica")

plt.xlabel("Index")

plt.ylabel("PetalLegtnCm")

plt.legend() # --> It will add Label into graphic

plt.show()
v_Setosa = v_dataFrame2[v_dataFrame2["Species"]=="Iris-setosa"]

v_Versicolor = v_dataFrame2[v_dataFrame2["Species"]=="Iris-versicolor"]

v_Virginica = v_dataFrame2[v_dataFrame2["Species"]=="Iris-virginica"]



plt.scatter(v_Setosa["PetalLengthCm"],v_Setosa["PetalWidthCm"], color="red",label="setosa")

plt.xlabel("PetalLengthCm")

plt.ylabel("PetalWidthCm")

plt.title("Scatter Plot")

plt.legend()

plt.show()


plt.scatter(v_Setosa["PetalLengthCm"],v_Setosa["PetalWidthCm"], color="red",label="setosa")

plt.scatter(v_Versicolor["PetalLengthCm"],v_Versicolor["PetalWidthCm"], color="blue",label="versicolor")

plt.scatter(v_Virginica["PetalLengthCm"],v_Virginica["PetalWidthCm"], color="green",label="virginica")

plt.xlabel("PetalLengthCm")

plt.ylabel("PetalWidthCm")

plt.title("Scatter Plot")

plt.legend()

plt.show()
plt.hist(v_Setosa["PetalLengthCm"],bins=17)

plt.xlabel("PetalLengthCm")

plt.ylabel("Frequency")

plt.title("Histogram")

plt.show()
test_x = np.array([1,2,3,4,5,6,7])

test_y = test_x*2+6

plt.bar(test_x,test_y)

plt.xlabel("test_x")

plt.ylabel("test_y")

plt.show()
v_dataFrame2.plot(grid=True,subplots=True,alpha = 0.8)

plt.show()
plt.subplot(2,1,1)

plt.plot(v_Setosa.index,v_Setosa["PetalLengthCm"],color="red",label="setosa")

plt.ylabel("PetalLengthCm")

plt.legend()

plt.subplot(2,1,2)

plt.plot(v_Versicolor.index,v_Versicolor["PetalLengthCm"],color="blue",label="versicolor")

plt.ylabel("PetalLengthCm")

plt.xlabel("index")

plt.legend()

plt.show()