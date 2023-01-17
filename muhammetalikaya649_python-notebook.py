_message = "hello world"

print(_message)
_name = "muhammet ali"

_surname = "kaya"

_fullname = _name + " " + _surname

print(_fullname)
_var1 = "1"

_var2 = "611"

_varsum1 = _var1 + _var2

print(_varsum1)
_var3 = 100

_var4 = 61

_varsum2 = _var3 + _var4

print(_varsum2)
#len function

print()

print("length of fullname is" , len(_fullname))

#title function

print()

print("my name is" , _fullname.title())

print()

#upper function

print()

print("upper of fullname" , _fullname.upper())

#lower funtion

print()

print("lower of fullname" , _fullname.lower())

#type

print()

print("type of fullname" , type(_fullname))

print()
_chr1 = _fullname [3]

_chr2 = _fullname [2]

_chr3 = _fullname [0]

_chr4 = _fullname [6]

_chr5 = _fullname [7]

_chrfull1= _chr1 + _chr2 + _chr3 + _chr4 + _chr5

print(_chrfull1)
_chr6 = _surname [0]

_chr7 = _name [3]

_chr8 = _name [10]

_chr9 = _name [6]

_chrfull2 = _chr6 + _chr7 + _chr8 + _chr9

print(_chrfull2)
#integer

_num1 = 30

_num2 = 37

_num3 = 45

_numfull1 = _num1 + _num2 + _num3 

print(_numfull1)
print("_num1 + _num2 + _num3" , _numfull1 , "and type" , type(_numfull1))
#float

_num4 = 30.5

_numfull2 = _num4 + _numfull1

print(_numfull2)

print()

print("numfull1 + num4" , _numfull2 , type(_numfull2))
def _sayhello1():

    print("hello")



_sayhello1()
def _SayMessage1(_message):

    print(_message)

    

_SayMessage1("I am Muhammet")
def _sum1(_num3 , _num2):

    _sum2 = _num3 + _num2

    print(_num3 , "+" , _num2 , "=" , _sum2)

    

_sum1(43,37)
_list1 = [1,2,3,4,5,6,7,8,9,10,11,12]

print(_list1)

print("_list1" , type(_list1))
_list1_1 = _list1[3]

print(_list1_1)

print("list_1_1" , _list1_1 , type(_list1_1))
_list2 = ["istanbu","trabzon","sinop","bursa","adana"]

print(_list2)

print("list2" , _list2 , type(_list2))
_list2_1 = _list2[2]

print(_list2_1)

print("list2_1" , _list2_1 , type(_list2_1))
_list2_x3 = _list2[-3]

print(_list2_x3)
_list2_2 = _list2[0:3]

print(_list2_2)
#Append

_list2_2.append("Saturday")

print(_list2_2)



_list2_2.append("Tuesday")

print(_list2_2)
_dict1 = {"Home":"Ev" , "School" : "Okul" , "Student": "Öğrenci"}



print(_dict1)
_dict2 = {"türkiye" : "ankara" , "almanya" : "berlin" , "fransa" : "paris"}

print(_dict2)
_school = _dict1["School"]

print(_school)

print(type(_school))
_türkiye = _dict2["türkiye"]

print(_türkiye)

print(type(_türkiye))
_almanya = _dict2["almanya"]

print(_almanya)

print(type(_almanya))
#Keys & Values



_keys = _dict1.keys()

_values = _dict1.values()





print(_keys)

print(type(_keys))



print()

print(_values)

print(type(_values))

#Keys & Values



_keys = _dict2.keys()

_values = _dict2.values()





print(_keys)

print(type(_keys))



print()

print(_values)

print(type(_values))

_var5= 56

_var6 = 354



if _var5 > _var6:

    print(_var5 , " is greater then " , _var6)

elif _var5 < _var6:

    print(_var5 , " is smaller then " , _var6)

else :

    print("This 2 variables are equal")
for a in range(0,62):

    print("ADIM" , a)
for a in range(1,6):

    print("bizim aile" , a , "kişilikti")
_message = "I love you"

print(_message)
for _chrs in _message:

    print(_message)

    print("-*-*-*-*")
for _chrs in _message:

    print(_chrs)

    print("------")
_message2 = "BİZE HER YER TRABZON"

print(_message2)
for _chrs2 in _message2:

    print(_message2)

    print("!!!!!!!")
for _chrs2 in _message2:

    print(_chrs2)

    print("&&&")
for _chrs in _message.split():

    print(_chrs)
for _chrs2 in _message2.split():

    print(_chrs2)
print(_list1)

_sum_list1 = sum(_list1)

print("listenin içindeki tüm sayıları topla : " , _sum_list1)

print()

_cum_list1 = 0

_loopindex = 0 

for _current in _list1:

    _cum_list1 = _cum_list1 + _current

    print(_loopindex , "nasılsın : " , _current)

    print("iyiyim sen nasılsın : " , _cum_list1)

    _loopindex = _loopindex + 61

    print("_?=_?=_?=")
a = 0

while(a < 10):

    print(a,"kemençe")

    a = a+1
print(_list1)

print()



i = 0

k = len(_list1)



while(i<k):

    print(_list1[i])

    i=i+1

_list3 = [1,2,3,4,5,6,7,8,9,0,-123,-394,-319,9981,92839,8362]



_min = 0

_max = 0



_index = 0

_len = len(_list3)



while (_index < _len):

    _current = _list3[_index]

    

    if _current > _max:

        _max = _current

    

    if _current < _min:

        _min = _current

        

    _index = _index+1



print ("EN BÜYÜK SAYI : "  , _max)

print ("EN KÜÇÜK SAYI : " , _min)
# Import library to use

import numpy as np
_aray1 = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]

_aray2_np = np.array([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])
print("_aray1" , _aray1)

print(type(_aray1))
print("_aray2_np" , _aray2_np)

print(type(_aray2_np))
# shape

_shape1 = _aray2_np.shape 

print(_shape1 , type(_shape1))
# reshape

_aray3_np = _aray2_np.reshape(4,5)

print(_aray3_np , type(_aray3_np))
_shape2 = _aray3_np.shape

print(_shape2 , type(_shape2))
# Dimension

_dimen1 = _aray3_np.ndim

print(_dimen1 , type(_dimen1))

_dtype1 = _aray3_np.dtype.name

print(_dtype1 , type(_dtype1))
# Size

_size1 = _aray3_np.size

print(_size1 , type(_size1))
# Let's create 5*4 array

_array4_np = np.array([[5,10,15,20,25],[30,35,40,45,50],[55,60,65,70,75],[80,85,90,95,100]])

print(_array4_np)

print("---------------")

print("Shape is : " , _array4_np.shape)
# Zeros

_aray5_np = np.zeros ((2,10))

print(_aray5_np)
# ones 

_aray6_np = np.ones((2,10))

print(_aray6_np)
# empty

_aray7_np = np.empty((5,4))

print(_aray7_np)
# Arrange

_array8_np = np.arange(5,30,45)

print(_array8_np)

print(_array8_np.shape)
# Linspace

_array9_np = np.linspace(10,30,5)

_array10_np = np.linspace(10,30,20)



print(_array9_np)

print(_array9_np.shape)

print("-----------------------")

print(_array10_np)

print(_array10_np.shape)
# Sum , Subtract , Square

_np1 = np.array([5,10,15])

_np2 = np.array([10,15,20])



print(_np1 + _np2)

print(_np1 - _np2)

print(_np2 - _np1)

print(_np1 ** 2)
# Sinus

print(np.sin(_np2))
# True / False

_np2_TF = _np2 < 8

print(_np2_TF)

print(_np2_TF.dtype.name)
# Element wise Prodcut

_np3 = np.array([1,2,3,4,5,6,7,8,9,10])

_np4 = np.array([11,12,13,14,15,16,17,18,19,20])

print(_np3 * _np4)

print(type(_np3 * _np4))
# Transpose

_np5 = np.array([[1,2,3],[4,5,6],[7,8,9]])

_np5Transpose = _np5.T

print(_np5)

print(_np5.shape)

print()

print()

print(_np5Transpose)

print(_np5Transpose.shape)
 # Matrix Multiplication

_np6 = _np5.dot(_np5Transpose)

print(_np6)
# Exponential --> We will use on Statistics Lesson

_np5exp = np.exp(_np5)

print(_np5)

print(_np5exp)
# random

_np6 = np.random.random((61,61))

print(_np6)
_np6sum = _np6.sum()

print("sum : " , _np6sum)

print("max : " , _np6.max)

print("min : " , _np6.min)
# Square , Sqrt

print(np.sqrt(_np6))

print()

print(np.square(_np6))
# Add

_np7 = np.array([1,2,3,4,5])

_np8 = np.array([5,10,15,20,25])



print(np.add(_np7,_np8))
_np9 = np.array([2,4,6,8,10,12,14,16,18,20])

print(_np9[0])

print(_np9[4])

print(_np9[7])
print(_np9[4:9])
# Reverse

_np9_Rev = _np9[::-1]

print(_np9_Rev)

_np10 = np.array([[1,2,3,4,5],[11,12,13,14,15]])

print(_np10)

print()

print(_np10[1,3]) #--> Get a row



print()

_np10[1,3] = 314 #--> Update a row

print(_np10)
# Get all rows but 3rd columns :

print(_np10[:,2])
#Get 2nd row but 2,3,4th columns

print(_np10[1,1:4])
#Flatten

_np11 = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])

_np12 = _np11.ravel()



print(_np11)

print(_np11.shape)

print()

print(_np12)

print(_np12.shape)

print()

_np13 = np.array([[1,2],[3,4],[5,6]])



print(_np13)

print()

print(_np13.reshape(2,3))



print()

print(_np13) 
# Resize

_np13.resize((2,3))

print(_np13)
_np14 = np.array([[2,4,5],[7,9,10]])

_np15 = np.array([[12,14,15],[17,19,20]])



print(_np14)

print()

print(_np15)
# Vertical Stack

_np16 = np.vstack((_np14,_np15))

_np17 = np.vstack((_np15,_np14))



print(_np16)

print()

print(_np17)
_list1 = [1,2,3,4]

_np18 = np.array(_list1)



print(_list1)

print(type(_list1))

print()

print(_np18)

print(type(_np18))
_list2 = list(_np18)

print(_list2)

print()

print(type(_list2))
_list3 = _list2

_list4 = _list2



print(_list2)

print()

print(_list3)

print()

print(_list4)
_list2[0] = 61



print(_list2)

print()

print(_list3)

print()

print(_list4)
_list5 = _list2.copy()

_list6 = _list2.copy()



print(_list5)

print()

print(_list6)
_list2[0] = 61000000061



print(_list2)

print()

print(_list5) 

print()

print(_list6) 
# pandas 

import pandas as pd
# Let's create Data Frame from Dictionary

_dict1 = { "team" : ["trabzon","galatasaray","fenerbahçe","beşiktaş","başakşehir","kasımpaşa","sarıyer"],

            "goalkeeper":["uğurcan","muslera","altay","karius","mert günok","fatih öztürk","ali türkan"],

            "number":[1.91,1.90,1.98,1.89,1.96,1.91,1.80]}



_dataFrame1 = pd.DataFrame(_dict1)



print(_dataFrame1)

print()

print("Type of v_dataFrame1 is : " , type(_dataFrame1))
_head1 = _dataFrame1.head()

print(_head1)

print()

print("Type of v_head1 is :" ,type(_head1))