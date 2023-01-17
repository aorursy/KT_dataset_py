var1 = 10 # integer = int

ay = "temmuz"

var3 = 10.3 # double (float)



s = "bugun gunlerden pazartesi"

variable_type = type(s)   # str = string

print(variable_type)


def benim_ilk_func(a,b):   

    """

    bu benim ilk denemem

    parametre: 

    return: 

    """

    output = (((a+b)*50)/100.0)*a/b

    return output



print(benim_ilk_func(20,50))
# %% 

# default f: cemberin cevre uzunlugu = 2*pi*r

def cember_cevresi_hesapla(r,pi=3.14):

    """

    cember cevresi hesapla

    input(parametre): r,pi

    output = cemberin cevresi

    """

    output = 2*pi*r

    return output



# flexible

def hesapla(boy,kilo,*args):

    print(args)

    output = (boy+kilo)*args[0]

    return output



print(cember_cevresi_hesapla(5))

print(hesapla(5,5,10,20,30,40))


def hesapla(x):

    return x*x

print(hesapla(3))



sonuc2 = lambda x: x*x

print(sonuc2(3))


def square():

    """ return square of value """

    def add():

        """ add two local variable """

        x = 2

        y = 3

        z = x + y

        return z

    return add()**2

print(square())
number_list = [1,2,3]

y = map(lambda x:x**2,number_list)

print(list(y))
liste = [1,2,3,4,5,6]

print(type(liste))



liste_str = ["ptesi","sali","cars"]

print(type(liste_str))



print(liste[1])

print(liste[-1])

print(liste[0:3])



liste.append(7)

print(liste)

liste.remove(7)

print(liste)

liste.reverse()

print(liste)



liste2 = [1,5,4,3,6,7,2]

liste2.sort()

print(liste2)



string_int_liste = [1,2,3,"aa","bb"]


t = (1,2,3,3,4,5,6)



print(t.count(5))

print(t.index(3))


def deneme():

    dictionary = {"ali":32,"veli":45,"ayse":13}

    # ali ,veli ,ayse = keys

# 32,45,13 = values

    return dictionary



dic = deneme()

print(dic)



dictionary = {'spain' : 'madrid','usa' : 'vegas'}

print(dictionary.keys())

print(dictionary.values())

dictionary['spain'] = "barcelona"    # update existing entry

print(dictionary)

dictionary['france'] = "paris"       # Add new entry

print(dictionary)

del dictionary['spain']              # remove entry with key 'spain'

print(dictionary)

print('france' in dictionary)        # check include or not

dictionary.clear()                   # remove all entries in dict

print(dictionary)
# if else statement



var1 = 10

var2 = 20



if(var1 > var2):

    print("var1 buyuktur var2")

elif(var1 == var2):

    print("var and var2 esitler")

else:

    print("var1 kucuktur var2")





liste = [1,2,3,4,5]



value = 3

if value in liste:

    print("evet {} degeri listenin icinde".format(value))

else:

    print("hayir")



dictionary = {"ali":32,"veli":45,"ayse":13}

keys = dictionary.keys()



if "veli" in keys:

    print("evet")

else:

    print("hayir")
# for loop

for each in range(1,3):

    print(each)

    

for each in "ank ist":

    print(each)

    

for each in "ank ist".split(): 

    print(each)

    

liste = [1,4,5,6,8,3,3,4,67]

 

print(sum(liste))  



count = 0

for each in liste:

    count = count + each

    print(count)



# while loop

i = 0

while(i <4):

    print(i)

    i = i + 1
class Calisan:

    zam_orani = 1.8

    counter = 0

    def __init__(self,isim,soyisim,maas): # constructor

        self.isim = isim

        self.soyisim = soyisim

        self.maas = maas

        self.email = isim+soyisim+"@asd.com"

        Calisan.counter = Calisan.counter + 1

    

    def giveNameSurname(self):

        return self.isim +" " +self.soyisim

        

    def zam_yap(self):

        self.maas = self.maas + self.maas*self.zam_orani

# class variable

calisan1 = Calisan("ali", "veli",100) 

print("giveNameSurname: ",calisan1.giveNameSurname())

print("maas: ",calisan1.maas)

calisan1.zam_yap()

print("yeni maas: ",calisan1.maas)



#  class example

calisan2 = Calisan("ayse", "hatice",200) 

calisan3 = Calisan("ayse", "yelda",600) 

liste  = [calisan1,calisan2,calisan3]



maxi_maas = -1

index = -1

for each in liste:

    if(each.maas>maxi_maas):

        maxi_maas = each.maas

        index = each

        

print(maxi_maas)

print(index.giveNameSurname())
# importing

import numpy as np

# numpy basics

array = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])  # 1*15 vector

print(array.shape)

a = array.reshape(-1,1)

print("shape: ",a.shape)

print("dimension: ", a.ndim)

print("data type: ",a.dtype.name)

print("size: ",a.size)

print("type: ",type(a))



array1 = np.array([[1,2,3,4],[5,6,7,8],[9,8,7,5]])

print(array1)

zeros = np.zeros((3,4))

zeros[0,0] = 5

print(zeros)



print(np.ones((3,4)))

print(np.empty((2,3)))



a = np.arange(10,50,5) # 10 dan 50 ye 5er 5 er artır

print(a)

a = np.linspace(10,50,5) #10 50 ye 5 tane yerleştir

print(a)


a = np.array([1,2,3])

b = np.array([4,5,6])



print(a+b)

print(a-b)

print(a**2)



a = np.array([[1,2,3],[4,5,6]])

b = np.array([[1,2,3],[4,5,6]])

# element wise prodcut

print(a*b)

# matrix prodcut

print(a.dot(b.T))



a = np.random.random((2,2)) # 2 2 lık 0-1 arasında sayı uretiyor

print(a.sum())

print(a.max())

print(a.min())

print(a.sum(axis=0)) # sutunları topla

print(a.sum(axis=1)) # satırları topla

print(np.sqrt(a))

print(np.square(a)) # a**2

print(np.add(a,a))


import numpy as np

array = np.array([1,2,3,4,5,6,7])   #  vector dimension = 1

print(array[0])

print(array[0:4])



reverse_array = array[::-1]

print(reverse_array)



array1 = np.array([[1,2,3,4,5],[6,7,8,9,10]])

print(array1[1,1]) # 1 satır 1 sutun 7 

print(array1[:,1]) # tum satır 1 sutun 2,7 

print(array1[1,1:4]) # 1 satır 1-4 sutun 7-8-9

print(array1[-1,:]) #son satırın tum sutunları

print(array1[:,-1]) # tum satırların son sutunu 5,10


array = np.array([[1,2,3],[4,5,6],[7,8,9]])



# flatten

array1 = array.ravel() # duz hale getırıldı

print(array1)

array2 = array1.reshape(3,3) # matrıse cevır

print(array2)



# %% stacking arrays

array1 = np.array([[1,2],[3,4]])

array2 = np.array([[-1,-2],[-3,-4]])



# veritical

#array([[1, 2],

#       [3, 4]])

#array([[-1, -2],

#       [-3, -4]])

array3 = np.vstack((array1,array2))

print(array3)

# horizontal

#array([[1, 2],[-1, -2],

#       [3, 4]],[-3, -4]]

array4 = np.hstack((array1,array2))

print(array4)


liste = [1,2,3,4]   # list

array = np.array(liste) #np.array

liste2 = list(array) # list



a = np.array([1,2,3])

b = a

b[0] = 5

print(b[0])

print(a[0]) # a da degıstı



d =  np.array([1,2,3])

e = d.copy()

d[0] = 5

print(d[0])

print(e[0]) # e da degısmedi
import pandas as pd



dictionary = {"NAME":["ali","veli","kenan","hilal","ayse","evren","isim1","isim2","isim3"],

              "AGE":[15,16,17,33,45,66,70,70,70],

              "MAAS": [100,150,240,350,110,220,300,300,300]} 



dataFrame1 = pd.DataFrame(dictionary)



print(dataFrame1.head()) # ilk 5 kısım, içerindeki görmek için

print(dataFrame1.tail()) # sondaki 5 tane
# pandas basic method

print(dataFrame1.columns)

print("------")

print(dataFrame1.info())

print("------")

print(dataFrame1.dtypes)

print("------")

print(dataFrame1.describe())  # numeric feature = columns (age,maas)


print(dataFrame1["AGE"])

print(dataFrame1.AGE)



dictionary = {"NAME":["ali","veli","kenan",],

              "AGE":[15,16,17,],

              "MAAS": [100,150,240]} 



dataFrame1 = pd.DataFrame(dictionary)

dataFrame1["yeni_feature"] = [-1,-2,-3]

print("---1---")

print(dataFrame1.loc[:, "AGE"])

print("---2---")

print(dataFrame1.loc[:1, "AGE"])

print("---3---")

print(dataFrame1.loc[:1, "AGE":"NAME"])

print("---4---")

print(dataFrame1.loc[:1, ["AGE","NAME"]])

print("---5---")

print(dataFrame1.loc[::-1,:]) #ters yazdı

print("---6---")

print(dataFrame1.loc[:,:"NAME"])

print("---7---")

print(dataFrame1.loc[:,"NAME"])

print("---8---")

print(dataFrame1.iloc[:,2]) #i integer location name yerıne sutun ındexı verdık


dictionary = {"NAME":["ali","veli","kenan","hilal","ayse","evren"],

              "AGE":[15,16,17,33,45,66],

              "MAAS": [100,150,240,350,110,220]} 

dataFrame1 = pd.DataFrame(dictionary)

filtre1 = dataFrame1.MAAS > 200

print(filtre1)

filtrelenmis_data = dataFrame1[filtre1]

print(filtrelenmis_data)

filtre2 = dataFrame1.AGE <20

dataFrame1[filtre1 & filtre2]

print(dataFrame1[dataFrame1.AGE > 60])


import numpy as np

dataFrame1 = pd.DataFrame(dictionary)

ortalama_maas = dataFrame1.MAAS.mean()

# ortalama_maas_np = np.mean(dataFrame1.MAAS)

dataFrame1["maas_seviyesi"] = ["dusuk" if ortalama_maas > each else "yuksek" for each in dataFrame1.MAAS]

print(dataFrame1)

print(dataFrame1.columns)

dataFrame1.columns = [ each.lower() for each in dataFrame1.columns] 

print(dataFrame1.columns)

dataFrame1.columns = [each.split()[0]+"_"+each.split()[1] if(len(each.split())>1) else each for each in dataFrame1.columns]

print(dataFrame1) # bosluklu sutun adı varsa _ ekledı ama bızım verıler zaten bosluksuz


dataFrame1["yeni_feature"] = [-1,-2,-3,-4,-5,-6]

dataFrame1.drop(["yeni_feature"],axis=1,inplace = True)

# dataFrame1 = dataFrame1.drop(["yeni_feature"],axis=1)

data1 = dataFrame1.head()

print(data1)

data2 = dataFrame1.tail()

print(data2)

# vertical

data_concat = pd.concat([data1,data2],axis=0) #dusey bırlestırdı yanı ust uste

print(data_concat)

# horizontal

maas = dataFrame1.maas

age = dataFrame1.age

data_h_concat = pd.concat([maas,age],axis=1) #yatay bırlestırdı yanı yanyana

print(data_h_concat)


def multiply(age):

    return age*2

dataFrame1["apply_metodu"] = dataFrame1.age.apply(multiply)

dataFrame1["list_comp"] = [ each*2 for each in dataFrame1.age]

print(dataFrame1)


name = "ronaldo"

it = iter(name)

print(next(it))    # print next iteration

print(*it)         # print remaining iteration


list1 = [1,2,3,4]

list2 = [5,6,7,8]

z = zip(list1,list2)

print(z)

z_list = list(z)

print(z_list)

print("-----")

un_zip = zip(*z_list)

un_list1,un_list2 = list(un_zip) # unzip returns tuble

print(un_list1)

print(un_list2)

print(type(un_list2))


num1 = [1,2,3]

num2 = [i + 1 for i in num1 ]

print(num2)

# Conditionals on iterable

num1 = [5,10,15]

num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]

print(num2)
import pandas as pd

data = pd.read_csv("../input/Iris.csv")

print(data.columns)
print(data.Species.unique())
data.info()
data.describe()
df1 = data.drop(["Id"],axis=1)

df1.corr()
setosa = data[data.Species == "Iris-setosa"]

versicolor = data[data.Species == "Iris-versicolor"]

print(setosa.describe())

print(versicolor.describe())
import matplotlib.pyplot as plt



setosa = data[data.Species == "Iris-setosa"]

versicolor = data[data.Species == "Iris-versicolor"]

virginica = data[data.Species == "Iris-virginica"]



plt.plot(setosa.Id,setosa.PetalLengthCm,color="red",label= "setosa")

plt.plot(versicolor.Id,versicolor.PetalLengthCm,color="green",label= "versicolor")

plt.plot(virginica.Id,virginica.PetalLengthCm,color="blue",label= "virginica")

plt.legend()

plt.xlabel("Id")

plt.ylabel("PetalLengthCm")

plt.show()



# clf() = cleans it up again you can start a fresh

#plt.clf()


# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.SepalLengthCm.plot(kind = 'line', color = 'g',label = 'SepalLengthCm',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.PetalLengthCm.plot(color = 'r',label = 'PetalLengthCm',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()


setosa = data[data.Species == "Iris-setosa"]

versicolor = data[data.Species == "Iris-versicolor"]

virginica = data[data.Species == "Iris-virginica"]



plt.scatter(setosa.PetalLengthCm,setosa.PetalWidthCm,color="red",label="setosa")

plt.scatter(versicolor.PetalLengthCm,versicolor.PetalWidthCm,color="green",label="versicolor")

plt.scatter(virginica.PetalLengthCm,virginica.PetalWidthCm,color="blue",label="virginica")

plt.legend()

plt.xlabel("PetalLengthCm")

plt.ylabel("PetalWidthCm")

plt.title("scatter plot")

plt.show()


plt.hist(setosa.PetalLengthCm,bins= 10)

plt.xlabel("PetalLengthCm values")

plt.ylabel("frekans")

plt.title("hist")

plt.show()


import numpy as np



x = np.array([1,2,3,4,5,6,7])

a = ["turkey","usa","a","b","v","d","s"]

y = x*2+5



plt.bar(a,y)

plt.title("bar plot")

plt.xlabel("x")

plt.ylabel("y")

plt.show()


df1 = data.drop(["Id"],axis=1)

df1.plot(grid=True,alpha= 0.9,subplots = True)

plt.show()



setosa = data[data.Species == "Iris-setosa"]

versicolor = data[data.Species == "Iris-versicolor"]

virginica = data[data.Species == "Iris-virginica"]



plt.subplot(2,1,1)

plt.plot(setosa.Id,setosa.PetalLengthCm,color="red",label= "setosa")

plt.ylabel("setosa -PetalLengthCm")

plt.subplot(2,1,2)

plt.plot(versicolor.Id,versicolor.PetalLengthCm,color="green",label= "versicolor")

plt.ylabel("versicolor -PetalLengthCm")

plt.show()
