message = "hello world"

print(message)
def ucgenhesapla(x,y,z):

    ucgencevresi= x + y + z

    print(ucgencevresi)

ucgenhesapla(12,23,34)



    

    
def karehesapla(k1,k2,k3,k4):

    karecevresi=k1+k2+k3+k4

    print(karecevresi)

karehesapla(1,1,1,1)
print("merhaba")
name = "sevde elif"

surname = "hacıosmanoğlu"

fullname= name + surname

print(fullname)
fullname = name + " " + surname

print(fullname)
num1 = "1837"

num2= "837"

numSum1 = num1 + num2

print(numSum1)
#length

lenfull = len(fullname)

print("fullname:" , fullname , "and lenght is:" , lenfull )
titlefull = fullname.title()

print("fullname:" , fullname, " and title is", titlefull)
#upper

upperfull = fullname.upper()



#lower

lowerfull = fullname.lower()

print("fullname:", fullname , "upper:", upperfull, "lower:" , lowerfull)
v_2ch =fullname[13]

print(v_2ch)
num1 = 100

num2 = 200

sum1 = num1 + num2



print(sum1 , " and  type : " , type(sum1))
num1 = num1 + 50

num2 = num2 - 47.5

sum1 = num1 + num2



print(num1)
print("sum1 : ",sum1 , " type : " , type(sum1))
fl1 = 28.5

fl2 = 39.5

s3 = fl1 + fl2



print(s3 , type(s3))
int1= 25

fl1=39.2

sum2= fl1 + int1

print(sum2, type(sum2))
def sayhello():

    print("hello")

def sayhello2():

    print("merhaba")

    print("ben sayhello2"),

    

sayhello()
sayhello2()
def message(message1):

    print(message1, "nasılsın")

message("merhaba")
def fullname(firstname, surname, age):

    print("welcome", firstname,surname , "your age", age)

    
fullname("sevde elif","hacıosmanoğlu",16)
def dikdörtgenC(kenar1, kenar2, kenar3, kenar4):

    sonuc = kenar1 + kenar2 + kenar3 + kenar4

    print("sonuç=",sonuc)
dikdörtgenC(10 , 15 , 10 , 15)

def ucgenc(a,b,c):

    ucgenc=a+b+c

    
#return



def sayılar(Num1 , Num2 , Num3):

    sonuc1 = Num1+Num2+Num3*8

    return sonuc1
sayı =  sayılar(2,8,6)

print("Sonuç : " , sayı)
#Default

def information(Name,age,date_of_birth,City,identification_number,favorite_color= "yellow"):

    print("Name : ", Name , " age : " ,age , " date of birth : " , date_of_birth, "City:",City,

          "identification number:",identification_number, "favorite color:",favorite_color)
information( "sevde elif",16, 2003, "istanbul",12312312312)

information( "sevde elif",16, 2003, "istanbul",12312312312,"green")
#Flexible



def Flexible(Name , *messages):

    print("Hi " , Name , " your first message is : " , messages[1])
Flexible("sevde elif" , "hi" , "welcome" , "how are you?")
#Lambda

lam1=lambda x : x*60

print(lam1(120))


def merhaba_de():

    print("merhaba")

merhaba_de()

merhaba_de()

merhaba_de()

merhaba_de()

merhaba_de()
def toplam(sayi1,sayi2):

    return sayi1+sayi2

 

sayi1 = int(input("Birinci sayıyı girin :"))

sayi2 = int(input("İkinci sayıyı girin :"))



print("Sonuç :",toplam(sayi1,sayi2))

 

def toplama(sayi1,sayi2):

    toplam=sayi1+sayi2

    print(toplam)

 

sayi1 = int(input("Birinci sayıyı girin :"))

sayi2 = int(input("İkinci sayıyı girin :"))

 

print("Sonuç :",toplam(sayi1,sayi2))

 
liste = [65,3,589,516,2,3]

print(liste)

print(type(liste))

liste2 = liste[2]

print(liste2)

print( type(liste2))
dersler = ["matematik","edebiyat","felsefe","fizik","biyoloji","kimya","geometri"]

print(dersler)

print( type(dersler))
dersler1 = dersler[0]

print(dersler1)

print( type(dersler1))
karsk=["deneme", 56,"deneme2",369]

print(karsk)

print(type(karsk))
karsk1 = karsk[-3]

print(karsk1)
dersler3 = dersler[0:5]

print(dersler3)
#Len

lendeneme = len(dersler)

print(lendeneme)

print(dersler)
#Append

dersler3.append("matematik")

print(dersler3)



dersler3.append("edebiyat")

print(dersler3)
#Reverse

dersler3.reverse()

print(dersler3)
#Sort

dersler3.sort()

print(dersler3)


dersler3.append("felsefe")

print(dersler3)

dersler3.remove("felsefe")

print(dersler3)
kitaplar = {"kitap1":562,"kitap2" :856 , "kitap3": 963,"kitap4":119}



print(kitaplar)

print(type(kitaplar))
kitap2 = kitaplar["kitap2"]

print(kitap2)

print(type(kitap2))
keyler = kitaplar.keys()

values_1 = kitaplar.values()





print(keyler)

print(type(keyler))



print()

print(values_1)

print(type(values_1))
değer1 = 253.6

değer2 = 253.5



if değer1 > değer2:

    print(değer1 , " daha büyük " , değer2)

elif değer1 < değer2:

    print(değer1 , " daha küçük " , değer2)

else:

    print("2 değişkende birbiri ile eşittir")
def karşılaştırma(sayı1 ,sayı2):

    if sayı1 > sayı2:

        print(sayı1 , " daha büyük" , sayı2)

    elif sayı1 < sayı2:

        print(sayı1 , " daha küçük " , sayı2)

    else :

        print("These " , sayı1 , " değişkenler eşit")

        

karşılaştırma(24,3456)

karşılaştırma(546,34)

karşılaştırma(345,345)       
def arama(arama1, aramalistesi):

    if arama1 in aramalistesi :

        print(arama1, " listede mevcut.")

    else :

        print(arama1 , "listede bulunamadı.")



list1 = list(kitaplar.keys())

print(list1)

print(type(list1))



arama("kitap4", list1)

arama("book" , list1)
sozluk1 ={"Computer":"Bilgisayar",

"Driver":"Sürücü",

"Memory":"Hafıza",

"Output":"Çıktı",

"Software":"Yazılım",

"Printer":"Yazıcı"}

 

print(sozluk["Computer"])
sozluk = {"ANK":"Ankara","İST":["Sarıyer","Beşiktaş","Şişli"],"nüfus":{"istanbul":5445000 ,"ankara":15070000}}

print(sozluk["nüfus"])


ing_sözlük = {"dil": "language", "bilgisayar": "computer", "masa": "table"}



sorgu = input("Please enter the word you want to know the meaning.:")



if sorgu not in ing_sözlük:

    print("This word is not in our database!")



else:

    print(ing_sözlük[sorgu])
for a in range(5,15):

    print("Hi " , a)
mesaj1 = "bu bir deneme mesajıdır"

print(mesaj1)
for b in mesaj1:

    print(b)

    print("------")
for b in mesaj1.split():

    print(b)
list1=[1,2,3,4,5,6,7,8,9]

sum_dersler = sum(list1)

print("list1 toplamı : " , list1)



print()

cum_list1 = 0

loopindex = 0

for current in list1:

    cum_list1 = cum_list1 + current

    print(loopindex , " nd value is : " , current)

    print("Cumulative is : " , cum_list1)

    loopindex = loopindex + 1

    print("------")
i = 0

while(i < 10):

    print("Hi" , i)

    i = i+1
print(list1)

print()



i = 0

k = len(list1)



while(i<k):

    print(list1[i])

    i=i+1
list2 = [5,7,9,-642,-255,-642.0,24,-5]



min_ = 0

max_ = 0



index = 0

len1 = len(list2)



while (index < len1):

    current = list2[index]

    

    if current > max_:

        max_ = current

        

    

    if current < min_:

        min_ = current

    

    index = index+1



print ("en büyük sayı: " , max_)

print ("en küçük sayı : " , min_)
import numpy as np
array1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

array2_np = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
print("array1 : " ,array1)

print("array1 tipi : " , type(array1))
print("array2_np : " , array2_np)

print("array2_np tipi : " , type(array2_np))
shape1 = array2_np.shape

print("shape1 : " , shape1 , " and type is : " , type(shape1))
array3_np = array2_np.reshape(3,5)

print(array3_np)
shape2 = array3_np.shape

print("shape2 : " , shape2 , " and type is : " , type(shape2))
dimen1 = array3_np.ndim

print("dimen1 : " , dimen1 , " type is : " , type(dimen1))
dtype1 = array3_np.dtype.name

print("dtype1 : " , dtype1 , " and type is : " , type(dtype1))
size1 = array3_np.size

print("size1 : " , size1 , " and type : " , type(size1))
array4_np = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

print(array4_np)

print("Shape is : " , array4_np.shape)
array5_np = np.zeros((3,4))

print(array5_np)
array5_np[0,3] = 12

print(array5_np)
array6_np = np.ones((3,4))

print(array6_np)
array7_np = np.empty((2,3))

print(array7_np)
array8_np = np.arange(5,60,5)

print(array8_np)

print(array8_np.shape)
array9_np = np.linspace(10,30,5)

array10_np = np.linspace(10,30,20)



print(array9_np)

print(array9_np.shape)

print(array10_np)

print(array10_np.shape)
np1 = np.array([1,2,3])

np2 = np.array([7,8,9])



print(np1 + np2)

print(np1 - np2)

print(np2 - np1)

print(np1 ** 2)
print(np.sin(np2))
np2_TF = np2 < 6

print(np2_TF)

print(np2_TF.dtype.name)
v_np1 = np.array([56,12.1,60])

v_np2 = np.array([8,2,4])

print(v_np1 * v_np2)
v_np5 = np.array([[5,7,2],[4,66,19]])

v_np5Transpose = v_np5.T

print(v_np5)

print(v_np5.shape)

print()

print(v_np5Transpose)

print(v_np5Transpose.shape)
v_np6 = v_np5.dot(v_np5Transpose)

print(v_np6)
v_np5Exp = np.exp(v_np5)



print(v_np5)

print(v_np5Exp)
v_np8 = np.random.random((5,4))

print(v_np8)
v_np8Sum = v_np8.sum()

print("Sum of array : ", v_np8Sum)

print("Max of array : ", v_np8.max())

print("Min of array : ", v_np8.min())
v_np8Sum = v_np8.sum()

print("Sum of array : ", v_np8Sum)

print("Max of array : ", v_np8.max())

print("Min of array : ", v_np8.min())
print("Sum of Columns :")

print(v_np8.sum(axis=0))

print("Sum of Rows :")

print(v_np8.sum(axis=1))
print(np.sqrt(v_np8))

print(np.square(v_np8))
v_np10 = np.array([1,2,3,4,5])

v_np11 = np.array([10,20,30,40,50])



print(np.add(v_np10,v_np11))
v_np12 = np.array([1,2,3,4,5,6,7,8,9])



print("First item is : " , v_np12[0])

print("Third item is : " , v_np12[2])
print(v_np12[0:4])
v_np12_Rev = v_np12[::-1]

print(v_np12_Rev)
v_np13 = np.array([[1,2,3,4,5],[11,12,13,14,15]])

print(v_np13)

print(v_np13[1,3])

v_np13[1,3] = 314

print(v_np13)
print(v_np13[:,2])
print(v_np13[1,1:4])
print(v_np13[-1,:])
print(v_np13[:,-1])
v_np14 = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])

v_np15 = v_np14.ravel()



print(v_np14)

print("Shape of v_np14 is : " ,v_np14.shape)

print(v_np15)

print("Shape of v_np15 is : " ,v_np15.shape)
v_np16 = v_np15.reshape(3,4)

print(v_np16)

print("Shape of v_np16 is : " ,v_np16.shape)
v_np17 = v_np16.T

print(v_np17)

print("Shape of v_np17 is : " ,v_np17.shape)
v_np20 = np.array([[1,2],[3,4],[5,6]])



print(v_np20)

print(v_np20.reshape(2,3))

print(v_np20)
v_np20.resize((2,3))

print(v_np20)
v_np21 = np.array([[1,2],[3,4]])

v_np22 = np.array([[5,6],[7,8]])



print(v_np21)

print(v_np22)
v_np23 = np.vstack((v_np21,v_np22))

v_np24 = np.vstack((v_np22,v_np21))



print(v_np23)

print(v_np24)
v_np25 = np.hstack((v_np21,v_np22))

v_np26 = np.hstack((v_np22,v_np21))



print(v_np25)

print(v_np26)
v_list1 = [1,2,3,4]

v_np30 = np.array(v_list1)



print(v_list1)

print("Type of list : " , type(v_list1))

print(v_np30)

print("Type of v_np30 : " , type(v_np30))
v_list2 = list(v_np30)

print(v_list2)

print("Type of list2 : " , type(v_list2))
v_list3 = v_list2

v_list4 = v_list2



print(v_list2)

print(v_list3)

print(v_list4)
v_list2[0] = 55



print(v_list2)

print(v_list3)

print(v_list4)
v_list5 = v_list2.copy()

v_list6 = v_list2.copy()



print(v_list5)

print(v_list6)
v_list2[0] = 71



print(v_list2)

print(v_list5)

print(v_list6)
import pandas as pd
dict1 = { "ülkeler" : ["Türkiye","Almanya","Fransa","Amerika","Azerbaycan","Kore"],

            "Başkentler":["İstanbul","Berlin","Paris","New york","Bakü","Seul"],

            "population":[15.07,3.57,2.12,8.62,4.3,10.2]}



dataFrame1 = pd.DataFrame(dict1)



print(dataFrame1)

print()

print("dataframe1 tipi : " , type(dataFrame1))
head1 = dataFrame1.head()

print(head1)

print("head1 tipi:" ,type(head1))
print(dataFrame1.head(100))
tail1 = dataFrame1.tail()

print(tail1)

print("tail1 tipi :" ,type(tail1))
columns1 = dataFrame1.columns

print(columns1)

print("columns1 tipi : " , type(columns1))
info1 = dataFrame1.info()

print(info1)

print("info1 tipi: " , type(info1))
dtypes1 = dataFrame1.dtypes

print(dtypes1)

print("dtypes1 tipi : " , type(dtypes1))
descr1 = dataFrame1.describe()

print(descr1)

print("descr1 tipi: " , type(descr1))
country1 = dataFrame1["ülkeler"]

print(country1)

print("country1 : " , type(country1))