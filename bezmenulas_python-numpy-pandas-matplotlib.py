# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
sayi = 10

ondalik = 10.0

yazi = "yzlı"



variable_type = type(sayi)

print(variable_type)
var1 = "okul"

var2 = "tatil"

var3 = var1+var2

print(var3)
uzunluk = len(var3)

print(var3, "=", uzunluk)

print("var3 5. karakteri = ",var3[4])
integer_syi = -50

float_syi = 32.2

topla = integer_syi + float_syi

print(topla)
#ondalıklı sayi yuvarlama

yuvarla = round(float_syi)

print(yuvarla)
# input 



var1 = int(input("enter a number:")) # 123

print(var1)
var2 = input("write a post:") # hello world!

print(var2)
# Fonksiyon oluşturma



var1 = 40

var2 = -10



output = (((var1+var2)*50)/100.0)*var1/var2



def islem_yap(c,b):

    islem = (((c+b)*50)/100.0)*c/b

    return islem



sonuc = islem_yap(var1,var2)



print(output," = ",sonuc)
def yazi_yaz():

    print("merhaba dünya")



yazi_yaz()
# Çember çevresi hesaplayan fonk.



def cember_cevre(r,pi=3.14):

    output = 2*pi*r

    return output



var1 = cember_cevre(5)

var2 = cember_cevre(5,3)

print(var1," != ", var2)
# flexible

def hesapla(boy,kilo,*args):

    print(args)

    output = (kilo+boy)*args[0]

    return output



var1 = hesapla(40,10,5)

var2 = hesapla(40,10,5,3,47)

print(var1," = ",var2)
# Ouiz



yas = 10

isim = "Mehmet"

soyisim = "Ak Bulut"

def bilgilendir(yas,isim,*args,notu=79):

    print("Adı:",isim,", Soyisim:",*args,", Yaş:",yas,", Notu:", notu)

    print(type(isim))

    print(len(soyisim))

    print(float(yas))

    

    output = args[0]*yas

    return output



sonuc = bilgilendir(yas,isim,soyisim)

#sonuc = bilgilendir(yas,isim,soyisim, notu=88)



print(sonuc)
# Lambda fonksiyonu



sonuc = lambda x:x*3

print(sonuc(8))

print(sonuc("yzi"))
# List



liste = [1,2,3,4,5,6]

print(type(liste))

print(liste)



liste_str = ["ptesi","salı","çarş"]

print("\n",liste_str)



gun = liste_str[1]

print(gun)
var1 = liste[3]

last_var = liste[-1]

liste_divide = liste[0:3]

print(var1,"\n",last_var,"\n",liste_divide)
liste.append(-99)

print(liste)

liste.remove(-99)

print(liste)

liste.append(123)

liste.remove(3)

print(liste)
liste.reverse()

print(liste)
liste = [1,5,9,3,2,44,-56,12,990]

liste.sort()

print(liste)
list_str_int = [1,2,3,"abc","Micheal"]

list_str_int
# tuple

t = (1,22,3,4,5,167,5,5,6,9)

print(t)

print("Count = ",t.count(5)) # kaç tne var?

print("index = ",t.index(5))
# dictionary



dictionary = {"micheal":32,"Ellie":41,"John":19}

# keys = micheal,ellie,john

# values = 32,41,19

print(dictionary)

print(dictionary.keys())

print(dictionary.values())
def dictionary_cre():

    a = {"micheal":32,"Ellie":41,"John":19}

    return a 

    

dic = dictionary_cre()

print(dic)
# if -else 



var1 = 10

var2 = 20



if var1>var2:

    print(var1)

elif var1==var2:

    print(var1,"=",var2)

else:

    print(var1<var2) # True
liste = [1,2,3,4,5,6]

var1 = 3



if var1 in liste:

    print("var1 in list = {}".format(var1))

else:

    print("nope")
keys = dictionary.keys()

print(keys)



if "can" in liste:

    print("yes")

else:

    print("nope")
# For loops



for each in range(1,11):

    print(each)

# each = 10
for each in "ankara ist":

    print(each)
for each in "ank ara ist     sss".split():

    print(each)
liste = [1,2,3,4,5,6,99,2,33,5,1]

print("Summation=",sum(liste))



count = 0

for each in liste:

    count = count+each

    print(count)



print(sum(liste),"=",count)
# while loop



i = 0

while i<4:

    print(i)

    i = i+1
limit = len(liste)

each = 0

count = 0

while each<limit:

    count = count+liste[each]

    each = each+1

print("Count = ",count)  
# Find minimum number ???

liste = [1,2,3,55,678,123,-546,-9,0]



# first way;

print(min(liste)) # -546



# second way;

min_number = liste[0]

for each in liste:

    if each < min_number:

        min_number = each

    else:

        continue

print(min_number)
# Error

    

a = 10

b = "2"

# print(a+b) # error 

print(str(a)+b)



k = 10

zero = 0

#a = k/zero # error



if(zero==0):

    a = 0

else:

    a = k/zero



try: 

    a = k/zero

except ZeroDivisionError:

    a = 0

print(a)
import numpy as np



array = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])# 1*15 vector

print(array)

print(type(array)) # numpy.ndarray



print(array.shape)
a = array.reshape(3,5)

print(a)

print("shape:",a.shape)

print("dimension",a.ndim) # Dizi boyutlarının sayısı.



print("data type:",a.dtype.name)

print("size:",a.size)



print("type:",type(a))



# array1 = np.array([[1,2,3,4],[5,6,7,8],[9,8,7,5]])
zeros = np.zeros((3,4))

zeros[0,0] = 5

print(zeros,"\n")



one = np.ones((3,2))

print(one,"\n")



emp = np.empty((2,3))

print(emp,"\n")



eye = np.eye(5)

print(eye,"\n")



a = np.arange(10,50,5)

print(a,"\n")



# 20 points from 10 to 50

a = np.linspace(10,50,20)

print(a,"\n")
# numpy basic operations

a = np.array([1,2,3])

b = np.array([4,5,6])



print("a+b=",a+b)

print("a-b=",a-b)

print("a**2=",a**2)



print("sin(a)=",np.sin(a))

print("a<2=",a<2)
a = np.array([[1,2,3],[4,5,6]])

b = np.array([[1,2,3],[4,5,6]])



print("a*b=",a*b)



b=b.T #transpoze

print(b) 
c = a.dot(b) # matrix product , matris çarpımı

print(c)
print(np.exp(a)) # exp = e^a
a = np.random.random((2,2)) # 0,1

print(a)
a = np.random.random((2,2))*10 # 0,10

print(a)
a = np.random.random((2,2))*10+5 # 5,15

print(a)
a = a.round()

print(a,"\n")



print(a.sum())

print(a.max())

print(a.min())

print(a.std()) # standart sapma

print(a.var()) # varyans
print(a.sum(axis=0))
print(a.sum(axis=1))
print(np.sqrt(a)) # square root
print(np.square(a)) # a**2
print(np.add(a,a)) # a+a
# indexing and slicing

array = np.array([1,2,3,4,5,6,7])



print(array[0])

print(array[2:5])



reverse_array = array[::-1]

print(reverse_array)
array = np.array([[1,2,3,4,5],[6,7,8,9,10]])

print(array[1,1])

print(array[:,1])

print(array[1,1:4])
print(array,"\n")

print(array[-1,:])

print(array[:,-1])
# shape manipulation

print(array)

array = np.array([[1,2,3],[4,5,6],[7,8,9]])

print("shape manipulation;\n",array)
# flatten

a = array.ravel()

print(a)
array2 = a.reshape(3,3)

print(array2,"\n")



print(array2.T,"\n")



print(array2.shape)
array1 = np.array([[1,2],[3,4],[4,5]])

array2 = np.array([[9,5],[2,7],[8,3]])



print("array1;\n",array1,"\n")

print("array2;\n",array2,"\n")



array3 = np.column_stack((array1,array2))

print("column_stack((array1,array2)\n",array3)
# stacking arrays

array1 = np.array([[1,2],[3,4]])

array2 = np.array([[-1,-2],[-3,-4]])



# veritical

array3 = np.vstack((array1,array2))

print("veritical \n",array3)



array3 = np.hstack((array1,array2))

print("\n hortizal \n",array3)
# convert and copy



liste = [1,2,3,4] # list

array = np.array(liste) # np.array

liste2 = list(array)



print("liste;\n",liste)

print("array;\n",array)

print("liste2;\n",liste2)
a = np.array([1,2,3])

print("a:",a)



b = a

b[0] = 55

c = a



print("\na:",a,"\nb:",b,"\nc:",c)
d = np.array([1,2,3])

e=d.copy()

f=d.copy()



print("d:",d,"\ne:",e,"\nf:",f)



print("\nnew:")

e[1] = 66

print("d:",d,"\ne:",e,"\nf:",f)
import pandas as pd
dictionary = {"name":["John","Harry","Ron","Micheal","Ellie","Fred","Saitama"],

             "age":[20,19,43,12,30,25,31],

             "point":[100,85,121,53,149,78,200]}



dataFrame1 = pd.DataFrame(dictionary)

print(dataFrame1)
dataFrame1.head() # top five
dataFrame1.tail() # last five
# pandas basic method



print(dataFrame1.columns)
print(dataFrame1.info())
print(dataFrame1.dtypes)
print(dataFrame1.describe()) # numeric feature = columns (age,point)
# indexing and slicing



print(dataFrame1["age"])



print("\n", dataFrame1.age)
dataFrame1["new feature"] = [-1,-2,-3,-4,-5,-6,-7]



print(dataFrame1["new feature"])
print(dataFrame1.loc[:,"name"])
print(dataFrame1.loc[:3,"name"])
print(dataFrame1.loc[:3,"name":"point"])
print(dataFrame1.loc[:3,["name","point"]])
print(dataFrame1.loc[::-1,:])
print(dataFrame1.loc[:,:"name"])
print(dataFrame1.loc[:,"name"])
print(dataFrame1.iloc[:,2]) # point
# filtering



filtre1 = dataFrame1["age"] > 25 # dataFrame1.age > 25

print(filtre1)
filtre_data = dataFrame1[filtre1]

print(filtre_data)
filtre2 = dataFrame1.point > 130

print(filtre2)
filtre_data = dataFrame1[filtre1 & filtre2]

print(filtre_data)
print(dataFrame1[dataFrame1.point > 100])
# list comprehension



# import numpy as np



mean_point = dataFrame1.point.mean()

# mean_point_np = np.mean(dataFrame1.point)



print(mean_point)
dataFrame1["point_level"] = ["lower" if mean_point>each else "high" 

                             for each in dataFrame1.point]



print(dataFrame1)



#for each in dataFrame1["point"]:

#    if mean_point > each:

#        print("lower")

#    else:

#        print("high")
dataFrame1.columns
dataFrame1.columns = [each.lower() for each in dataFrame1.columns]

dataFrame1.columns
dataFrame1.columns = [each.split()[0]+"_"+each.split()[1] 

                      if(len(each.split())>1) else each 

                      for each in dataFrame1.columns]

dataFrame1.columns

# new featur ----> new_feature
# drop and concatenating



dataFrame1.drop(["new_feature"],axis=1,inplace=True)

dataFrame1



# dataFrame1 = dataFrame1.drop(["yeni_feature"],axis=1)
data1 = dataFrame1.head()

data2 = dataFrame1.tail()



print(data1)

data2
# vertical



data_concat = pd.concat([data1,data2],axis=0)

data_concat
# horizontal



name = dataFrame1.name

age = dataFrame1.age



data_h_concat = pd.concat([name,age],axis=1)

data_h_concat
# transforming data 

dataFrame1["list_comp"] = [each*2 for each in dataFrame1.age]



dataFrame1
# apply()



def multiply(age):

    return age*2



dataFrame1["apply_method"] = dataFrame1.age.apply(multiply)



dataFrame1
import pandas as pd

df = pd.read_csv("../input/Iris.csv")



df
print(df.columns)
df.Species.unique() # unique ---> benzersiz,eşsiz
df.info()
df.describe()
setosa = df[df.Species == "Iris-setosa"]

versicolor = df[df.Species == "Iris-versicolor"]



setosa
setosa.describe()
versicolor.describe()
import matplotlib.pyplot as plt

df1 = df.drop(["Id"],axis=1)



setosa = df[df.Species == "Iris-setosa"]

versicolor = df[df.Species == "Iris-versicolor"]

virginica = df[df.Species == "Iris-virginica"]



setosa.columns
df1.plot(grid=True,alpha=0.9)



setosa.plot()

versicolor.plot()

virginica.plot()



plt.show()
plt.plot(setosa.Id,setosa.PetalLengthCm,color="red",label="setosa")

plt.plot(versicolor.Id,versicolor.PetalLengthCm,color="green",label="versicolor")

plt.plot(virginica.Id,virginica.PetalLengthCm,color="blue",label="virginica")

plt.legend()

plt.xlabel("Id")

plt.ylabel("PetalLengthCm")

plt.show()



plt.plot(setosa.Id,setosa.PetalLengthCm,color="red",label="setosa")

plt.legend()

plt.xlabel("setosa.Id")

plt.ylabel("setosa.PetalLengthCm")

plt.show()



plt.plot(versicolor.Id,versicolor.PetalLengthCm,color="green",label="versicolor")

plt.legend()

plt.xlabel("versicolor.Id")

plt.ylabel("versicolor.PetalLengthCm")

plt.show()



plt.plot(virginica.Id,virginica.PetalLengthCm,color="blue",label="virginica")

plt.legend()

plt.xlabel("virginica.Id")

plt.ylabel("virginica.PetalLengthCm")

plt.show()
# scatter plot



plt.scatter(setosa.PetalLengthCm,setosa.PetalWidthCm,color="red",label="setosa")

plt.scatter(versicolor.PetalLengthCm,versicolor.PetalWidthCm,color="green",label="versicolor")

plt.scatter(virginica.PetalLengthCm,virginica.PetalWidthCm,color="blue",label="virginica")

plt.legend()

plt.xlabel("PetalLengthCm")

plt.ylabel("PetalWidthCm")

plt.title("Scatter Plot")

plt.show()
# histogram



plt.hist(setosa.PetalLengthCm,bins=50)

plt.xlabel("PetalLengthCm values")

plt.ylabel("frekans")

plt.title("Histogram")

plt.show()
# bar plot



import numpy as np



plt.bar(setosa.Id,setosa.PetalLengthCm)

plt.title("Bar plot")

plt.xlabel("setosa.Id")

plt.ylabel("setosa.PetalLengthCm")

plt.show()
# subplots



df1.plot(grid=True,alpha=0.9,subplots=True)

plt.show()
setosa = df[df.Species == "Iris-setosa"]

versicolor = df[df.Species == "Iris-versicolor"]
plt.subplot(2,1,1)

plt.plot(setosa.Id,setosa.PetalLengthCm,color="red",label= "setosa")

plt.ylabel("setosa -PetalLengthCm")



plt.subplot(2,1,2)

plt.plot(versicolor.Id,versicolor.PetalLengthCm,color="green",label= "versicolor")

plt.ylabel("versicolor -PetalLengthCm")



plt.show()