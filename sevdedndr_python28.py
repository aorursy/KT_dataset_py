v_message = 'Hello World'

print(v_message)



v_name='Sevde'

v_surname='Dundar'

v_fullname=v_name+' '+v_surname

print(v_fullname)
#len function:

print()

print("Lenght of v_fullname is" , len(v_fullname))



#title function:

print()

print("Full Name is" , v_fullname.title())



#upper function:

print()

print("Upper of Full Name is" , v_fullname.upper())



#lower function:

print()

print("Full Name is" , v_fullname.lower())



#type function:

print()

print("Type of v_fullname is" , type(v_fullname))



v_chr1 = v_fullname[4]

v_chr2 = v_fullname[6]

print('v_chr1 : ', v_chr1, 'and v_chr2  : ' , v_chr2)
#Integer

v_num3 = 10

v_num4 = 35

v_sum1 = v_num3 + v_num4

print(v_sum1)





print('v_num3 : ' , v_num3 , 'and type:' , type(v_num3))



print()

print('Sum of Num3 and Num4 is :',v_sum1 , 'and type :', type(v_sum1))

v_num1 = "480"

v_num2 = "840"

v_sum1 = v_num1 + v_num2



print(v_sum1)



v_num1 = 480

v_num2 = 840

v_Sum1 = v_num1 + v_num2 



print(v_Sum1)
#Float



v_num6=16.2

v_Sum2=v_num6+v_num3



print('Sum of Num and Num3 is :',v_Sum2, 'and type:' , type(v_Sum2))
dict1= {"Kitaplik":"Kitap" , "Suluk" : "Su" , "Kalemlik": "Kalem"}



print(dict1)

print(type(dict1))
v_kitaplik = dict1["Kitaplik"]

print(v_kitaplik)

print(type(v_kitaplik))
#Keys & Values



v_keys = dict1.keys()

v_values = dict1.values()





print(v_keys)

print(type(v_keys))



print()

print(v_values)

print(type(v_values))


v_Zahl1 = 100

v_Zahl2 = 270



if  v_Zahl1 > v_Zahl2:

    print(v_Zahl1 , " ist gröβer als " , v_Zahl2)

elif v_Zahl1 < v_Zahl2:

    print(v_Zahl1 , " ist kleiner als  " , v_Zahl2)

else :

    print("Diese 2 Variablen sind gleich")
def f_Vergleich1(v_verg1 , v_verg2):

    if v_verg1 > v_verg2:

        print(v_verg1 , " ist gröβer als " , v_verg2)

    elif v_verg1 < v_verg2:

        print(v_verg1 , " ist kleiner als " , v_verg2)

    else :

        print("Diese " , v_verg1 , " Variablen sind gleich")

        

f_Vergleich1(96,97)

f_Vergleich1(887,878)

f_Vergleich1(1020,1020)
def f_IncludeOrNot(v_search, v_searchList):

    if v_search in v_searchList :

        print("GOOD NEWS ! ",v_search , " İS İN LİST.")

    else :

        print(v_search , " İSN'T İN LİST. SORRYY :(((")



list1 = list(dict1.keys())

print(list1)

print(type(list1))



f_IncludeOrNot("Gözlük" , list1)

f_IncludeOrNot("Kalemlik" , list1)
for sevde in range(0,11):

    print("SELAMUNALEYKÜM " , sevde)
Message = "I AM SAD AS ALWAYS"

print(Message)
for chrs in Message:

    print(chrs)

    print("------")
for chrs in Message.split():

    print(chrs)


list1 = [2,6,4,7,9,10]

print(list1)

sum_list1 = sum(list1)

print("Sum of list1 is : " , sum_list1)



print()

v_cum_list1 = 0

v_loopindex = 0

for v_current in list1:

    v_cum_list1 = v_cum_list1 + v_current

    print(v_loopindex , " nd value is : " , v_current)

    print("Cumulative is : " , v_cum_list1)

    v_loopindex = v_loopindex + 1

    print("------")
s = 1

while(s < 26):

    print("HALLO" , s)

    s = s+1
print(list1)

print()



o = 0

k = len(list1)



while(o<k):



    print(list1[o])

    o=o+1
#minimum and maximum



list2 = [3,5,7,-6,-100,255,71,34,-85]



v_mini = 0

v_maxi = 0



index = 0

v_len = len(list2)



while (index < v_len):

    current = list2[index]

    

    if current > v_maxi:

        v_maxii = v_current

    

    if current < v_mini:

        v_mini = v_current

    

    index = index+1



print ("Maximum number is : " , v_maxi)

print ("Minimum number is : " , v_mini)
# Import library to use

import numpy as np
array = ['Tokyo ',' Delhi',' Şangay',' Sao Paulo',' Mexico City',' Kahire',' Dakka',' Mumbai',' Pekin',' Osaka',' Karaçi',' Çongçing',' Buenos Aires',' İstanbul',' Kalküta']

array_np = np.array(['Tokyo ',' Delhi',' Şangay',' Sao Paulo',' Mexico City',' Kahire',' Dakka',' Mumbai',' Pekin',' Osaka',' Karaçi',' Çongçing',' Buenos Aires',' İstanbul',' Kalküta'])


print('15 most populous cities in the world:',array)

print("------")

print("------")

print("Type of array : " , type(array))

                                    

print("array_np : " , array_np)

print("------")

print("Type of array_np : " , type(array_np))

# SHAPE

shape = array_np.shape

print("shape : " , shape , " and type is : " , type(shape))

#RESHAPE

reshape = array_np.reshape(3,5)

print(reshape)
shape2 = reshape.shape

print("shape2 : " , shape2 , " Type of shape2 : " , type(shape2))
#dimension

dimen1 = reshape.ndim

print("dimen1 : " , dimen1 , " Type of dimension : " , type(dimen1))
#dtype.name



dtype1 = reshape.dtype.name

print("dtype1 : " , dtype1 , " Type of dtype1 : " , type(dtype1))
#size

size1 = reshape.size

print("size1 : " , size1 ,  "Type of size1 : " , type(size1))
# Let's create 3x4 array

reshape2 = np.array([['Tokyo ',' Delhi',' Şangay',' Sao Paulo'],[' Mexico City',' Kahire',' Dakka',' Mumbai'],[' Pekin',' Osaka',' Karaçi',' Çongçing']])

print(reshape2)

print("---------------")

print("Shape is : " , reshape2.shape)
#zeros

zeros = np.zeros((6,9))

print(zeros)
#update an item on this array 

zeros[0,7] = 111

print(zeros)
#ones

ones = np.ones((9,5))

print(ones)
#empty

empty = np.empty((2,5))

print(empty)
# Arrange

arrange= np.arange(11,100,11)

print(arrange)

print(arrange.shape)
#linspace

linspace = np.linspace(10,30,5)

linspace2 = np.linspace(10,20,9)



print(linspace)

print(linspace.shape)

print("-----------------------")

print(linspace2)

print(linspace2.shape)
#sum,subtract,square



np1 = np.array([1,3,5])

np2 = np.array([7,9,11])



print(np1 + np2)

print(np1 - np2)

print(np2 - np1)

print(np1 ** 2)
np3 = np.array([13,15,17,19])

#sinus

print(np.sin(np3))
#true or false

np2_TF = np2 < 200000000000

np3_TF = np2 >9

print(np2_TF)

print(np2_TF.dtype.name)

print(np3_TF)

print(np3_TF.dtype.name)
#element wise prodcut



print(np1 * np2)