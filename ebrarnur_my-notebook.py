print('hi kaggle')
e_message="hi kaggle"

print("hi")
print(e_message)
e_name="ebrar"

e_surname="altın"



e_full=e_name + e_surname



print(e_full)
e_full=e_name+" "+e_surname



print(e_full)
e_num1="120"

e_num2="40"

e_num3=e_num1+e_num2



print(e_num3)
e_length=len(e_full)

print("e_fullname:",e_full,"and length is:",e_length)
e_title=e_full.title()



print("e_fullname:",e_full, "and title is:" ,e_title)
e_upper=e_full.upper()



e_lower=e_full.lower()



print("e_fullname:",e_full,"e_upper:",e_upper,"e_lower:",e_lower)
e_one=e_full[0]



print(e_one)
e_number1=25

e_number2=75

e_number3=e_number1+e_number2



print(e_number3,"and type:",type  (e_number3))
e_num1=32.5

e_num2=22.5

e_num3=e_num1+e_num2



print(e_num3)
print("e_num3:",e_num3,"type:",type(e_num3))
e_n1=12.5

e_n2=32.5

e_n3=e_n1+e_n2



print(e_n3 ,type(e_n3))
def e_hikaggle():

    print("ı am here")

    

e_hikaggle()    
def e_hikaggle():

    print("ı am traying")

 

e_hikaggle()
def e_message(message1):

    print("you have a message",message1)

e_message("'hi'")
e_message("'how are you'")
def e_date(day,month,year):

    print("date", day,"/",month,"/",year)

    
e_date("27","09","2012")
def e_number(e_num1,e_num2):

    e_number2= e_num1 + e_num2

    print(e_number2)
e_number(12,12)
#return fuction



def e_number(num1,num2,num3):

    e_Out=num1+num2+num3*2

    print("score")

    return e_Out
e_score=e_number(2,3,5)

print(e_score)
#default functions:



def e_info(e_name,e_studentcount,e_city ="SARIYER"):

    print("name:",e_name,"studentcount:",e_studentcount,"city:", e_city)



e_info("AAİHL",123,)

e_info("AAİHL",123,"Beşiktaş")
#Flexible Functions



def e_Flex(e_name,*e_message):

    print(e_name,"you have a message:",e_message[1])
e_Flex("Ebrar:","nasılsın?","nerdesin?")
#Lambda Function:



e_example=lambda x: x*3

print("example:",e_example(5))

def e_area(edge1,edge2):

    print(edge1*edge2)

e_area(5,5)
e_answers=["1a","2e","3d","4a"]

print(e_answers)

print("type of answer key is :",type(e_answers))
e_list=e_answers[2]

print(e_list)

print("type of e_list is : ",type(e_list))
e_city=["istanbul","ankara","izmir","konya"]

print(e_city)

print("type of city list:",type(e_city))
e_clist=e_city[1]

print(e_clist)

print("type of e_clist is:",type(e_clist))
e_clist2=e_city[-1]

print(e_clist2)

print("type of e_clist2 is:",type(e_clist2))
e_clist3=e_city[0:2]

print(e_clist3)
#len

e_list2=len(e_clist3)

print("size of e_clist3 is :",e_list2)

print(e_clist3)

#append



e_clist3.append("ankara")

print(e_clist3)
e_city.append("istanbul")

print(e_city)

#reverse



e_city.reverse()

print(e_city)
#sort



e_answers.sort()

print(e_answers)
#remove



#first add '2e' then remove '2e'



e_answers.append("2e")

print(e_answers)
e_answers.remove("2e")

print(e_answers)
e_dict = {"bulut" : "облако" , "etek" : "юбка" , "fırça" : "щетка" , "top" : "мяч"}

print(e_dict)

print(type(e_dict))
e_bulut=e_dict["bulut"]

print(e_bulut)

print(type(e_bulut))
# keys & Values



e_key=e_dict.keys()





print(e_key)

print(type(e_key))

e_values = e_dict.values()



print(e_values)

print(type(e_values))
e_factor1=25

e_factor2=17



if e_factor1>e_factor2:

    print(e_factor1 ,"greater than", e_factor2)

    

elif e_factor1<e_factor2:

    print(e_factor1 ,"smaller than",e_fator2)

    

else:

    print("This 2 variables are equal")

    
e_factor1=45

e_factor2=56



if e_factor1>e_factor2:

    print(e_factor1 ,"greater than", e_factor2)

    

elif e_factor1<e_factor2:

    print(e_factor1 ,"smaller than",e_factor2)
# < , <= , > , >= , == , <>



def e_factor(e_öge1 , e_öge2):

    if e_öge1 > e_öge2:

        print(e_öge1 , " is greater then " , e_öge2)

    elif e_öge1 < e_öge2:

        print(e_öge1 , " is smaller then " , e_öge2)

    else :

        print("These " , e_öge1 , " variables are equal")

        

e_factor(56,84)

e_factor(54,32)

e_factor(99,99)
# Using "İN" with list





def e_factors(e_öge, e_searchList):

    if e_öge in e_searchList :

        print("Good news ! ",e_öge, " is in list.")

    else :

        print(e_öge , " is not in list. Sorry :(")



l_list = list(e_dict.keys())

print(l_list)

print(type(l_list))



e_factors("top" , l_list)

e_factors("bulut" , l_list)
for a in range (0,10):

    print("level",a)
j_message="WHY SO SERİOUS"

print(j_message)
for e_message in j_message:

    print(e_message)

    print("???")
for e_message in j_message.split():

    print(e_message)
e_list=[1,2,3,4,5]

print(e_list)

sum_list=sum(e_list)

print("sum of list is:",sum_list)



print()

e_cum_list=0

e_loopindex=0

for e_current in e_list:

    e_cum_list= e_cum_list+e_current

    print(e_loopindex,"nd values is:",e_current)

    print("cumulative is:",e_cum_list)

    e_loopindex=e_loopindex+1

    print("------")
i=0

while(i<6):

    print("hi",i)

    i=i+1
print(e_list)

print()



i=0

k=len(e_list)



while(i<k):

    print(e_list[i])

    i=i+1
e_list2=[10,-30,600,78,-43]



e_min=0

e_max=0



e_index=0

e_len=len(e_list2)



while(e_index<e_len):

    e_current=e_list2[e_index]

    

    if e_current>e_max:

        e_max=e_current

        

    if e_current<e_min:

        e_min=e_current

        

        

    e_index=e_index+1

    

    

print("max number is:",e_max)

print("min number is:",e_min)
#İmport library to use

import numpy as np
e_array=[1,2,3,4,5,6,7,8,9,0]

e_array_2=np.array([1,2,3,4,5,6,7,8,9,0])
print("e_arry :",e_array)

print("type of e_array:",type(e_array))
print("e_array_2:",e_array_2)

print("type of e_array_2:",type(e_array_2))
#shape

e_shape=e_array_2.shape



print("e_shape:",e_shape,"type of e_shape:",type(e_shape))
#reshape

e_reshape=e_array_2.reshape(5,2)



print("e_reshape:",e_reshape,"type of reshape:",type(e_reshape))
#dimension

e_dimen=e_array_2.ndim

print("e_dimen:",e_dimen,"type is:",type(e_dimen))
#dtype.name

e_dtype=e_array_2.dtype.name

print("e_dtype:",e_dtype,"and type is:",type(e_dtype))
#Size 

e_size=e_array_2.size

print("e_size:",e_size,"and type is:",type(e_size))
e_array3=np.array([[1,2],[3,4],[5,6],[7,8],[9,0]])

print(e_array3)



print()



print("shape is:",e_array3.shape)
#Zeros

e_array4=np.zeros((5,2))

print(e_array4)
# Update an item on this array 

e_array4[3,1]=11

print(e_array4)
#Ones

e_array5=np.ones((2,7))

print(e_array5)
#Empty

e_array6=np.empty((1,5))

print(e_array6)
#Arrange

e_array7= np.arange(5,10,15)

print(e_array7)

print(e_array7.shape)
#Linspace

e_array8=np.linspace(10,20,30)

e_array9=np.linspace(20,35,40)



print(e_array8)

print(e_array8.shape)



print("********")



print(e_array9)

print(e_array9.shape)
#Sum  Subtact  Square

e_np1=np.array([3,6,9])

e_np2=np.array([5,10,15])



print(e_np1+e_np2)

print(e_np1-e_np2)

print(e_np2-e_np1)

#Sinus

print(np.sin(e_np1))
#True False

e_np_TF=e_np2<6

print(e_np_TF)

print(e_np_TF.dtype.name)
# Element wise Prodcut

e_np_1 = np.array([2,4,5])

e_np_2 = np.array([5,10,3])



print(e_np_1*e_np_2)
#Transpose



e_np=np.array([[2,4,6],[3,6,9]])

e_transpose=e_np.T

print(e_np)

print(e_np.shape)

print()

print(e_transpose)

print(e_transpose.shape)
# Matrix Multiplication



e_np2=e_np.dot(e_transpose)

print(e_np2)
# Exponential --> We will use on Statistics Lesson



e_npexp=np.exp(e_np)

print(e_np)

print(e_npexp)
#Random



e_np2=np.random.random((6,6)) # --> It will get between 0 and 1 random numbers

print(e_np2)
#Sum Max Min



e_npsum=e_np2.sum()

print("sum of array:",e_npsum)#--> Remember ! If you get sum of array we can use that :  sum(array1)



print("max of array:",e_np2.max())#--> Remember ! If you get max of array we can use that :  max(array1)



print("min of array:",e_np2.min())#--> Remember ! If you get min of array we can use that :  min(array1)
# Sum with Column or Row



print("Sum of Columns :")

print(e_npsum.sum(axis=0))

# Square , Sqrt

print(np.square(e_npsum))

print()

print(np.sqrt(e_npsum))
e_np9=np.array([2,4,6,8])

e_np10=np.array([50,60,70,80])



print(np.add(e_np9,e_np10))
e_np11=np.array([1,2,3,4,5,6,7,8,9])



print("first item is:", e_np11[0])

print("sixth item is:",e_np11[5])
# Get top 4 rows :



print(e_np11[0:5])
# Reverse

e_np11_rev=e_np11[::-2]

print(e_np11_rev)
e_np12=np.array([[3,6,9,12,15,18],[6,12,18,24,30,36]])

print(e_np12)

print()

print(e_np12[1,3])#--> Get a row

print()

e_np12[1,3]=200 #--> Update a row

print(e_np12)
# Get all rows but 3rd columns :

print(e_np12[:,2])
#Get 2nd row but 2,3,4th columns

print(e_np12[1,1:3])
# Get last row all columns

print(e_np12[-2,:])
# Get last columns but all rows

print(e_np12[:,-2])
#Flatten



e_np13=np.array([[1,2,5],[5,10,15,],[20,25,30],[26,87,13]])

e_np14=e_np13.ravel()



print()



print(e_np13)

print("Shape of e_np13 is : ",e_np13.shape)

print(e_np14)

print("Shape of v_np14 is : ",e_np14.shape)
# Reshape

e_np15=e_np14.reshape(2,6)

print(e_np15)

print("Shape of v_np15 is : ",e_np15)
e_np16=e_np15.T

print(e_np16)

print("Shape of v_np16 is :",e_np16.shape)
e_np17=np.array([[11,12],[33,34],[55,56]])



print(e_np17)

print()

print(e_np17.reshape(2,3))

print()

print(e_np17)#--> It has not changed !!!





# Resize

e_np17.resize((2,3))

print(e_np17) # --> Now it changed !  Resize can change its shape

e_list1=[2,3,4,5]

e_np20=np.array(e_list1)

print()

print(e_list1)

print("type of e_list1:",type(e_list1))

print()

print(e_np20)

print("type of  e_np20:",type(e_np20))
e_list2=list(e_np20)

print(e_list2)

print("type of e_list2:",type(e_list2))
e_list3=e_list2

e_list4=e_list2



print(e_list2)

print()

print(e_list3)

print()

print(e_list4)
e_list2[2]=33



print(e_list2)

print()

print(e_list3)# --> Same address with list2

print()

print(e_list4)# --> Same address with list2
e_list5=e_list.copy()

e_list6=e_list2.copy()



print(e_list5)

print()

print(e_list6)
e_list[2]=13



print(e_list2)

print()

print(e_list5)# --> Not same address with list2

print()

print(e_list6)# --> Not same address with list2
# Import Library

import pandas as pd
e_dict1={"Region":["Marmara Region","Central Anatolia Region","Black Sea Region","Aegean Region","Eastern Anatolia Region"],

       "City":["İstanbul","Nevşehir","Samsun","Afyon","Ağrı"], 

        "Plate" :["34","50","55","03","04"]}



e_dataframe=pd.DataFrame(e_dict1)



print(e_dataframe)



print()



print("type of e_dataframe is:",type(e_dataframe))
# get top 5 rows



e_head=e_dataframe.head()



print(e_head)

print()

print("type of e_head is:", type(e_head))
# get top 100 rows

print(e_dataframe.head(55))
e_tail=e_dataframe.tail()

print(e_tail)

print()

print("type of e_tail is:",type(e_tail))
e_columns=e_dataframe.columns



print(e_columns)

print()

print("type of e_columns is:",type(e_columns))
e_info=e_dataframe.info()

print(e_info)

print()

print("type of e_info is:",type(e_info))
e_dtypes=e_dataframe. dtypes

print(e_dtypes)

print()

print("type of e_dtypes is :",type(e_dtypes))

e_describe=e_dataframe.describe()

print(e_describe)

print()

print("type of e_describe is:",type(e_describe))
e_region=e_dataframe["Region"]

print(e_region)

print()

print("type of e_region is:",type(e_region))
# Add new column



e_district=["Sarıyer","Acıgöl","Çarşamba","Emirdağ","Taşlıçay"]

e_dataframe["District"]=e_district

print(e_dataframe.head())
# Get all rows ,  1 column



e_city=e_dataframe.loc[:,"City"]

print(e_city)

print()

print("type of e_city is:",type(e_city))
e_plate=e_dataframe.loc[0:4,"Plate"]

print(e_plate)
e_f=e_dataframe.loc[:,["Plate","District","..."]] #--> "..." not defined !!!

print(e_f)

e_reverse=e_dataframe.loc[::-1,:]

print(e_reverse)
print(e_dataframe.loc[:,:"City"])

print()

print(e_dataframe.loc[:,"Plate":])
#Get data with column index (not column name)

print(e_dataframe.iloc[:,2])
e_altitude=[40,1.224,4,1010,1.640]

e_dataframe["Altitude"]=e_altitude



e_filter1=e_dataframe.Altitude>4



print(e_filter1)

e_filter2=e_dataframe["Altitude"]<7

print(e_filter2)
print(e_dataframe[e_filter1&e_filter2])
print(e_dataframe[e_dataframe["Plate"]=="50"])
e_mean1=e_dataframe["Plate"].mean()

print(e_mean1)



e_mean2=np.mean(e_dataframe["Plate"])

print(e_mean2)
for a in e_dataframe["Altitude"]:

    print(a)
e_dataframe["Altitude LEVEL"] = ["Low" if e_mean1 > a else "HIGH" for a in e_dataframe["Altitude"]]

print(e_dataframe)
print(e_dataframe.columns)



e_dataframe.columns = [a.lower() for a in e_dataframe.columns]



print(e_dataframe.columns)
e_dataframe.columns = [a.split()[0]+"_"+a.split()[1] if (len(a.split())>1) else a for a in e_dataframe.columns]

print(e_dataframe.columns)
e_dataframe.drop(["altitude"],axis=1,inplace = True) #--> inplace = True must be written

print(e_dataframe)
v_data1 = e_dataframe.head()

v_data2 = e_dataframe.tail()



print(v_data1)

print()

print(v_data2)
v_dataConcat1 = pd.concat([v_data1,v_data2],axis=0) # axis = 0 --> VERTICAL CONCAT

v_dataConcat2 = pd.concat([v_data2,v_data1],axis=0) # axis = 0 --> VERTICAL CONCAT



print(v_dataConcat1)

print()

print(v_dataConcat2)
e_region = e_dataframe["region"]

e_city = e_dataframe["city"]



v_dataConcat3 = pd.concat([e_region,e_city],axis=1) #axis = 1 --> HORIZONTAL CONCAT

v_dataConcat4 = pd.concat([e_region,e_city],axis=1) #axis = 1 --> HORIZONTAL CONCAT

print(v_dataConcat3)

print()

print(v_dataConcat4)
e_number1=[-1,-2,-3,-4,-5]

e_dataframe["Number"]=e_number1

print(e_dataframe)
e_dataframe["Number"] = [a*2 for a in e_dataframe["Number"]]

print(e_dataframe)
e_altitude=[40,1.224,4,1010,1.640]

e_dataframe["Altitude"]=e_altitude
def e_multiply(e_altitude):

    return e_altitude*3



e_dataframe["Number"] = e_dataframe["Altitude"].apply(e_multiply)

print(e_dataframe)