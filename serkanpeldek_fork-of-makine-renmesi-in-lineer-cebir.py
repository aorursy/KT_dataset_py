import numpy as np # lineer cebir işlemleri için

import scipy as sp

import pandas as pd # verinin organizasyonu için



#Grafik çizdirme kütüphanesi

import matplotlib.pyplot as plt

#Pratik grafik çizdirme kütüphanesi

import seaborn as sns



#Makine öğrenmesi modelinin performans değerlendirmesi için gerekli

#fonksiyonlar

from sklearn import metrics



from sklearn.model_selection import train_test_split



import os #Sistem 

import warnings #uyarılar

print(os.listdir("../input"))

warnings.filterwarnings("ignore")
arr1=list([1,2.0, 'karakter'])

print(arr1)
print("Dönüşümden önce veri yapısının türü:",type(arr1))

arr1=np.array(arr1)

print("Dönüşümden sonra veri yapısının türü:",type(arr1))
arr2=np.array([1,2,3,4])

print("Veri yapısının türü:",type(arr2))

arr2=list(arr2)

print("Veri yapısının türü:",type(arr2))
def get_public_attributes(aclass):

    #liste tanımlanıyor

    public_attributes=[]

    

    #dir ön tanımlı olarak Python içerisinde yer alan bir fonksiyondur. 

    #parametre olarak aldığı nesnenin geçerli özelliklerine(attributes) döner.

    attributes=dir(aclass)

    

    #özellik listesi içerisindeki herbir özellik alınıyor

    for attribute in attributes:

        

        #önünde iki altçizgi(__) olmayan özellik isimleri kontrol ediliyor

        if "__" not in attribute:

            #önünde iki altçizgi(__) olmayan özellik isimleri ekleniyor

            public_attributes.append(attribute)

            

    

    #Elde edilen özellikler listesine geri dönülüyor

    return public_attributes
list_public_attributes=get_public_attributes(list)

print("Pyton dizisinin sahip olduğu dışa açık özelliklerin sayısı:",len(list_public_attributes))

print("Pyton dizisinin sahip olduğu dışa açık özellikler:\n",list_public_attributes)

print()

ndarray_public_attributes=get_public_attributes(np.array([]))

print("Numpy dizisinin sahip olduğu dışa açık özellikler sayısı:",len(ndarray_public_attributes))

print("Numpy dizisinin sahip olduğu dışa açık özellikler:\n",ndarray_public_attributes)

print(list.append.__doc__)

print(help(list.append))
def attributes_functionality(object_name, attributes):

    for attribute in attributes:

        full_name=object_name+"."+attribute+".__doc__"

        print(eval(full_name))

        print("*"*90)
attributes_functionality('list', list_public_attributes)
attributes_functionality('np.ndarray', ndarray_public_attributes)
#tüm değişkenler int tipinde olduğu için dtype int64 olacaktır.

#ndarray dizisinde varsayılan tam sayı tipi int64'tür

arr1=np.array([1,2,3,4])

print("{} dizisinin".format(arr1))

print("değişken tipi:{}\nşekli:{}\n".format(arr1.dtype, arr1.shape))



#tüm değişkenler noktalı sayı tipinde olduğu için dtype float64 olacaktır.

#ndarray dizisinde varsayılan noktalı sayı tipi float64'tür

arr2=np.array([1., 2., 3., 4.])

print("{} dizisinin".format(arr2))

print("değişken tipi:{}\nşekli:{}\n".format(arr2.dtype, arr2.shape))



#Değişkenlerden üçü tam sayı ve biri noktalı sayı olduğu için dtype float64 olacaktır

#float64, int64'ten daha kapsalı olduğu için dtype float64 olmuştur

arr3=np.array([1, 2, 3, 4.])

print("{} dizisinin".format(arr3))

print("değişken tipi:{}\nşekli:{}\n".format(arr3.dtype, arr3.shape))



#ndarray dizisinin değişen tipini kendimiz belirleyebiriz.

#Dizideki tüm değişkenler -128 ile 127 arasında olduğu için int8 yeterli olacaktır

arr4=np.array([1,2,3,4], dtype=np.int8)

print("{} dizisinin".format(arr4))

print("değişken tipi:{}\nşekli:{}".format(arr4.dtype, arr4.shape)) 
d1=4

d2=5

d3=3

arr=np.zeros((d1, d2,d3),dtype=np.int32)

print("dizinin şekli:",arr.shape)

print("dizinin kaç boyutlu olduğu:",arr.ndim)

print("dizinin kaç boyutlu olduğu:",len(arr.shape))
arr_arange=np.arange(10)

print("varsayılan dtype:",arr_arange.dtype)

print(arr_arange, end="\n\n")



arr_empty=np.empty((3,3))

print("varsayılan dtype:",arr_empty.dtype)

print(arr_empty, end="\n\n")



arr_zeros=np.zeros((3,3))

print("varsayılan dtype:",arr_zeros.dtype)

print(arr_zeros, end="\n\n")



arr_ones=np.ones((3,3))

print("varsayılan dtype:",arr_ones.dtype)

print(arr_ones, end="\n\n")



any_number=24

arr_any=np.ones((3,3))*any_number

print("varsayılan dtype:",arr_any.dtype)

print(arr_any)
empty_zeros=np.vstack(([arr_empty,arr_zeros]))

print(empty_zeros, end="\n\n")



ones_any=np.hstack(([arr_ones, arr_any]))

print(ones_any)
arr_arange=np.arange(0,20,3)

i=3

print(arr_arange)

print("arr_arange[{}]:{}".format(i, arr_arange[i]), end="\n\n")



i, j= 3,2

arr_nd=np.array([[1,2,3],

                    [4,5,6],

                    [7,8,9],

                    [10, 11, 12]])

print(arr_nd)

print("arr_nd[{}][{}]:{}".format(i, j, arr_nd[i][j]))

print("arr_nd[{},{}]:{}".format(i, j, arr_nd[i,j]))
i, j=-3, -2

print(arr_arange)

print("arr_arange[{}]:{}".format(i, arr_arange[i]), end="\n\n")



arr_nd=np.array([[1,2,3],

                    [4,5,6],

                    [7,8,9],

                    [10, 11, 12]])

print(arr_nd)

print("arr_nd[{}][{}]:{}".format(i, j, arr_nd[i][j]))

print("arr_nd[{},{}]:{}".format(i, j, arr_nd[i,j]))


row=0

col=2

print(arr_nd, end="\n\n")

print("{}. satır, arr_nd[{}]:{}".format(row+1,row,  arr_nd[row]), end="\n\n")

print("{}. satır, arr_nd[{},]:{}".format(row+1,row,  arr_nd[row,]), end="\n\n")

print("{}. sütün, arr_nd[:,{}]:{}".format(col+1,col,  arr_nd[:,col]), end="\n\n")
import random

#20 uzunluğunda bir ndarray dizisi oluşturuluyor

#dizinin elemanları 0 ile 19 arasındaki değerlerden oluşmaktadır.

arr=np.arange(20)

#random.shuffle(arr)



#dizi tüm elemanlar alınıyor

arr_part=arr[:]

print("{:<10}:{}".format("arr",arr))

print("{:<10}:{}".format("arr[:]",arr_part),end="\n\n")



start=4

stop=19

#4. ve 18. indeks arasındaki elemanlar alınıyor

arr_part=arr[start:stop]

print("{:<10}:{}".format("arr",arr))

print("{:<10}:{}".format("arr[{}:{}]".format(start,stop),arr_part),end="\n\n")



start=4

stop=-1

#4. indeksten sonuncu elemana kadar olan elemanlar alınıyor

arr_part=arr[start:stop]

print("{:<10}:{}".format("arr",arr))

print("{:<10}:{}".format("arr[{}:{}]".format(start,stop),arr_part),end="\n\n")



#4. indekste sonraki elemanlar alınıyor

arr_part=arr[start:]

print("{:<10}:{}".format("arr",arr))

print("{:<10}:{}".format("arr[{}:]".format(start),arr_part),end="\n\n")



#dizi sondan başa olacak şekilde alınıyor,yani ters döndürülüyor.

arr_part=arr[::-1]

print("{:<10}:{}".format("arr",arr))

print("{:<10}:{}".format("arr[::-1]",arr_part),end="\n\n")



start=4

stop=19

step=2

#4. ve 18. indeks arasındaki elemanlar alınıyor

arr_part=arr[start:stop:step]

print("{:<11}:{}".format("arr",arr))

print("{:<10}:{}".format("arr[{}:{}:{}]".format(start,stop,step),

                         arr_part),end="\n\n")
#yeni bir dizi oluşturuluyor

arr=np.arange(20)

#Dizi 4X5 bir matrisi dönüştürülüyor

#reshape(yeniden şekillendirmeyi) bir sonraki bölümde inceleyeceğiz

arr=arr.reshape(4,5) # veya arr=np.reshape(arr, (4,5))



print("orijinal arr:\n",arr, end="\n\n")



n=1

print("{}.satır: {}".format(n, arr[n]), end="\n\n")

print("{}.sütün: {}".format(n, arr[:,n]), end="\n\n")



row=3

col=4

arr_sub=arr[:row,:col]#üç satır al, dört sütün al

print("{}:\n{}".format("arr[:{},:{}]".format(row, col),arr_sub), end="\n\n")

step=2

arr_sub=arr[:,::2]#tüm satırları al, sütünlar 2 atlamalı al 

print("{}:\n{}".format("arr[:,::{}]".format(step),arr_sub), end="\n\n")



arr_sub=arr[::-1,::-1]#Matrisi ters dönder

print("{}:\n{}".format("arr[::-1,::-1]",arr_sub), end="\n\n")
#Bir boyutlu bir dizi oluşturuluyor

arr1=np.arange(12)



#yeni bir alt dizi elde ediliyor.

arr1_sub=arr1[1:10:2]

print("Alt Dizide Değişiklik Yapmadan Önce")

print("arr1    :",arr1)

print("arr1_sub:",arr1_sub, end="\n"*2)



arr1_sub[:]=255

print("Alt Dizide Değişiklik Yaptıktan Sonra")

print("arr1    :",arr1)

print("arr1_sub:",arr1_sub)
#Bir boyutlu bir dizi oluşturuluyor

arr1=np.arange(12)



#yeni bir alt dizi orijinal diziden bağımsız olarak elde ediliyor.

#arr1_sub=arr1[1:10:2].copy() #1. yöntem

arr1_sub=np.copy(arr1[1:10:2]) #2. yöntem

print("Alt Dizide Değişiklik Yapmadan Önce")

print("arr1    :",arr1)

print("arr1_sub:",arr1_sub, end="\n"*2)



arr1_sub[:]=255

print("Alt Dizide Değişiklik Yaptıktan Sonra")

print("arr1    :",arr1)

print("arr1_sub:",arr1_sub)
arr=np.arange(24)



arr_default=arr.reshape((6,4))# varsayılan olarak order="C"

arr_C=arr.reshape((6,4), order='C')

arr_F=arr.reshape((6,4), order="F")



print("arr:",arr, end="\n\n")

print("arr.reshape((6,4)):\n",arr_default,end="\n\n")

print("arr.reshape((6,4), order='C'):\n",arr_C,end="\n\n")

print("arr.reshape((6,4), order='F'):\n",arr_F,end="\n\n")

arr_F.reshape((24))
arr_F.reshape((24), order='F')
arr_C.reshape((24))
#24 -->2x3x4 dönüşüm

arr_3B=arr.reshape((2,3,4))



#2x3x4 --> 12x2 dönüşüm

arr_2B=arr_3B.reshape((12,2))



print("arr:",arr, end="\n\n")

print("24 -->2x3x4 dönüşüm:\n",arr_3B, end="\n\n")

print("2x3x4 --> 12x2 dönüşüm:\n",arr_2B, end="\n\n")
arr1=np.ones((4), dtype=np.int32)

arr2=np.arange(4)

arr3=np.ones((4,4),dtype=np.int32)



print("arr1:",arr1)

print("arr2:",arr2)

#Burada farklı bir durum yok, eşit boyutlara sahip iki vektör 

#toplanıyor. 

print("arr1+arr2:",arr1+arr2, end="\n\n")#



print("arr:",arr1)

print("arr+2:",arr1+2, end="\n\n")



print("arr3:\n",arr3)

print("arr3+4:\n",arr3+4, end="\n\n")
arr4=np.arange(4).reshape((4,1))

print('arr4:', arr4)

arr5 =arr4.reshape(-1)

print('arr5:', arr5)

print('arr5.shape', arr5.shape)

arr6 =arr5.reshape((arr5.shape[0], 1))

print('arr6:',  arr6)



#satır vektörle matris toplanıyor

print("arr3+arr2:\n",arr3+arr2, end="\n\n")



#sütün vektörle matris toplanıyor

print("arr3+arr4:\n",arr3+arr4)
arr = np.arange(20).reshape(4,5)

print(arr)

print()

arr1 = arr.reshape(4, 5, 1)

print(arr1)

print()

print(arr1[0])
row_vector=np.array([1,2,3,4])

col_vector=np.array([[1],[2],[3],[4]])

print("satır vektör:",row_vector)

print("sütün vektör:",col_vector)
#satır vektörü sütün vektörüne dönüştürüyor

#Dönüşüm yapılırken satır vektörle sütün vektörün uzunluklarının eşit olduğundan emin olunmalıdır.

#Vektörü uzunluklarının eşit olduğunu garantilemek için shape özelliği kullanılmıştır.

print(row_vector.reshape((row_vector.shape[0],1)))
#sütün vektörü satır vektörüne dönüştürülüyor.

#Dönüşüm yapılırken sütün vektörle satır vektörün uzunluklarının eşit olduğunda emin olunmalıdır.

#Vektörü uzunluklarının eşit olduğunu garantilemek için shape özelliği kullanılmıştır.

print(col_vector.reshape(col_vector.shape[0]))
a=np.array([2, 4, 6, 8])

b=np.array([1, 2, 3, 4])



print("a  :",a)

print("b  :",b)

print("a+b:",a+b, end="\n\n")



print("a  :",a)

print("b  :",b)

print("a-b:",a-b, end="\n\n")



print("a  :",a)

print("b  :",b)

print("a*b:",a*b, end="\n\n")



print("a  :",a)

print("b  :",b)

print("a/b:",a/b, end="\n\n")



print("a  :",a)

print("b  :",b)

print("a**b:",a**b, end="\n\n")



print("a  :",a)

print("b  :",b)

print("a%b:",a%b, end="\n\n")
a=np.array([3, 11, 7])

s=3



print("a   :", a)

print("s   :", s)

print("a+s :", a+s, end="\n\n")



print("a   :", a)

print("s   :", s)

print("a-s :", a-s, end="\n\n")



print("a   :", a)

print("s   :", s)

print("a*s :", a*s, end="\n\n")



print("a   :", a)

print("s   :", s)

print("a/s :", a/s, end="\n\n")



print("a   :", a)

print("s   :", s)

print("a**s :", a**s, end="\n\n")



print("a   :", a)

print("s   :", s)

print("a%s :", a%s, end="\n\n")
a=np.array([1, 5, 3])

b=np.array([2, 6, 1])



c=a*b

a_dot_b= np.sum(c)



print("a   :", a)

print("b   :", b)

print("a.b :", a_dot_b)
a_dot_b=a.dot(b)

print("a   :", a)

print("b   :", b)

print("a.b :", a_dot_b)
a=np.array([1,-2,4,-5,1,-8,7,-2])

l1=np.sum(np.absolute(a))

l2=np.sqrt(np.sum(a**2))

max_norm=np.max(np.abs(a))

min_norm=np.min(np.abs(a))

frobenius_norm=np.sqrt(np.sum(a**2))



print("a       :",a)

print("l1 norm :",l1.astype(np.float32))

print("l2 norm :{:.2f}".format(l2))

print("max norm:", max_norm.astype(np.float32))

print("min norm:", min_norm.astype(np.float32))

print("Frobenious norm:{:.2f}".format(frobenius_norm))
import math

l1=np.linalg.norm(a, ord=1)

l2=np.linalg.norm(a, ord=2)

max_norm=np.linalg.norm(a, ord=math.inf)

min_norm=np.linalg.norm(a, ord=-math.inf)

frobenius_norm=np.linalg.norm(a, ord=None)



print("a       :",a)

print("l1 norm :",l1)

print("l2 norm :{:.2f}".format(l2))

print("max norm:", max_norm)

print("min norm:",min_norm)

print("Frobenius norm:{:.2f}".format(frobenius_norm))
#Sıfırdan matris oluşturuluyor.

#İki boyutlu bir liste ndarray dizisine dönüştürülüyor

A=np.array([[1, 2, 3, 4],

            [5, 6, 7, 8],

            [9, 10, 11, 12]])



#Bir boyutlu dizi oluşturluyor

b=np.arange(1,13)

#Dizi matrise dönüştürülüyor

B=b.reshape(3,4)



#Tüm elemanları 0 olan matris oluşturuluyor

C=np.zeros((3,4), dtype=np.int32)



#Tüm elemanların 1 olan matris oluşturuluyor

D=np.ones((3,4), dtype=np.int32)



#Tüm elemanları istenen sayıdan oluşan matris oluşturulor

number=6

E=np.ones((3, 4), dtype=np.int32)*number



print("A:\n", A, end="\n\n")



print("b:",b)

print("b.reshape(3,4):\n", B, end="\n\n")



print("np.zeros((3,4), dtype=np.int32):\n", C, end="\n\n")



print("np.ones((3,4), dtype=np.int32):\n", D, end="\n\n")



print("np.ones((3, 4), dtype=np.int32)*{}:\n".format(number), E, end="\n\n")
#[0,1) arasıdan rasgele noktalı değerler içeren matris oluşturuluyor

F=np.random.rand(3,4)



#istenen aralıkta rasgele tam sayı değerler içeren matris oluşturuluyor

G=np.random.randint(low=0,high=100,size=(3,4))



#istenen aralıkta rasgele noktalı değer içeren matris oluşturuluyor

H=np.random.uniform(low=0, high=100, size=(3,4))



print("np.random.rand(3,4):\n",F, end="\n\n")

print("np.random.randint(low=0,high=100,size=(3,4)):\n",G, end="\n\n")

print("np.random.uniform(low=0, high=100, size=(3,4)):\n",H, end="\n\n")
A=np.random.randint(low=2, high=10,size=(2,3))

B=np.ones((2,3),dtype=np.int32)*2



print("A:\n",A,end="\n\n")

print("B:\n",B,end="\n\n")

print("="*80)



print("A+B:\n",A+B,end="\n\n")

print("="*80)



print("A-B:\n",A-B,end="\n\n")

print("="*80)



print("A*B:\n",A*B,end="\n\n")

print("="*80)



print("A/B:\n",A/B,end="\n\n")

print("="*80)



print("A^B:\n",A**B,end="\n\n")

print("="*80)



print("A%B:\n",A%B,end="\n\n")
A=np.random.randint(low=2, high=10,size=(2,3))

s=np.random.randint(1,5)



print("A:\n",A,end="\n")

print("s:",s,end="\n")

print("="*80)



print("A+s:\n",A+s,end="\n\n")

print("="*80)



print("A-s:\n",A-s,end="\n\n")

print("="*80)



print("A*s:\n",A*s,end="\n\n")

print("="*80)



print("A/s:\n",A/s,end="\n\n")

print("="*80)



print("A^s:\n",A**s,end="\n\n")

print("="*80)



print("A%s:\n",A%s,end="\n\n")
def dot_product(first_matris, second_matris):

    C=[]

    C_row=[]



    #Birinci matrisin satırları alınıyor

    for row in first_matris:

        

        #İkinci matrisin sütünları alınıyor

        for col in second_matris.T:# T matrisin transpozu alınıyor

            #Birinci matrisin ilgili satırıyla ikinci matrisin tüm sütünları

            #çarpılarak sonuç matrisin satırları elde ediliyor

            C_row.append(row.dot(col))

        

        #Satırlar sonuç matrisine ekleniyor

        C.append(C_row)

        #Bir sonraki satır için satır sıfırlanıyor

        C_row=[]



    return np.array(C)
A=np.random.randint(low=0, high=3,size=(2,3))

B=np.random.randint(low=0, high=3, size=(3,4))



C=dot_product(A,B)



print("A:\n",A, end="\n\n")

print("B:\n",B, end="\n")

print("="*80)

print("A.B:\n",C)
C=A.dot(B)

print(C)
"""

Kare matrisler satır ve sütünları eşit 

matrislerdir. 

"""

n=5

A=np.arange(n*n).reshape((n,n))

print(A)
n=4

S=np.zeros((n,n),dtype=np.int32)



#Rasgele değer içeren simetrik matris oluşturuluyor

for i in range(n):

    for j in range(0, n-i):

        S[i,j]=np.random.randint(1,5)

        S[n-j-1,n-i-1]=S[i,j]

print(S)
n=4

LT=np.zeros((n,n),dtype=np.int32)



#Rasgele değer içeren alt üçgensel matris oluşturuluyor

for i in range(n):

    for j in range(n):

            if j<=i:

                LT[i,j]=np.random.randint(1,5)

print(LT)
n=4

UT=np.zeros((n,n),dtype=np.int32)



#Rasgele değer içeren üst üçgensel matris oluşturuluyor

for i in range(n):

    for j in range(n):

            if j>=i:

                UT[i,j]=np.random.randint(1,5)

print(UT)
n=4

m=6

DM=np.zeros((n,m),dtype=np.int32)



#Rasgele değer içeren köşegen matris oluşturuluyor

#Köşegen matrislerin kare matris olma zorunluluğu yoktur

#Dikdörtgen matrislerde köşegen kısa kenarın uzunluğu kadardır

for i in range(n):

    for j in range(m):

            if j==i:

                DM[i,j]=np.random.randint(1,5)

print(DM)
n=4



I=np.zeros((n,n),dtype=np.int32)



#Birim matris, köşegeni 1 olan kare matristir. 

for i in range(n):

    I[i,i]=1

print(I)
#Alt üçgensel matris

LT=np.tril(A)



print("A:\n",A, end="\n\n")

print("A alt üçgen matrisi\n",LT)
#Üst üçgensel matris

UT=np.triu(A)



print("A:\n",A, end="\n\n")

print("A alt üçgen matrisi\n",UT)
diagonal=np.diag(A)

A_diagonal=np.diag(diagonal)



print("A:\n",A,end="\n\n")

print("A köşegeni:\n",diagonal, end="\n\n")

print("Köşegenden oluşturulan matris:\n",A_diagonal)
#Birim matris

n=3

I=np.identity(n)

print(I)
A=np.arange(15).reshape((3,5))



A_T=np.zeros((A.shape[1],A.shape[0]), dtype=np.int32)



for i, row in enumerate(A):

    A_T[:,i]=row



print("A:\n",A, end="\n\n")

print("A transpozu:\n",A_T)
from numpy.random import RandomState

rnd=RandomState(42)

n=4

A=np.zeros((n,n))

for i in range(n):

    for j in range(n):

            A[i,j]=rnd.randint(0,3)



A_inv=np.linalg.inv(A)

print("A:\n",A,end="\n\n")

print("A matrisin tersi A^-1:\n",A_inv,end="\n\n")

print("A matrisinin tersiyle çarpımı A.A^-1:\n",A.dot(A_inv))
n=4

A=np.arange(n*n).reshape((n,n))



#A_trace=np.sum(np.diag(A))

A_trace=np.trace(A)

print("A:",A, end="\n\n")

print("A matrisinin izi:",A_trace)
rnd=RandomState(24)

n=4

A=np.zeros((n,n))

for i in range(n):

    for j in range(n):

            A[i,j]=rnd.randint(0,3)



A_det=np.linalg.det(A)           

print("A:\n",A,end="\n\n")

print("A matrisinin determinantı:",A_det)
print(np.linalg.matrix_rank(A))
n=4

A=np.arange(n*n).reshape((n,n))

print(np.linalg.matrix_rank(A))
A=np.arange(1,17).reshape(4,4)





P, L, U=sp.linalg.lu(A)

print("A:\n",A,end="\n\n")

print("P:\n",P, end="\n\n")

print("L:\n",L,end="\n\n")

print("U:\n",U, end="\n\n")

print("L.U.P:\n",P.dot(L).dot(U))
A=A.reshape((8,2))

Q, R =sp.linalg.qr(A)



print("A:\n",A,end="\n\n")

print("Q:\n",Q, end="\n\n")

print("R:\n",R,end="\n\n")

print("Q.R:\n",Q.dot(R))
#Cholesky ayrışımına uygun matris elde etmenin kendimce

#kolay yolu olduğu için pozitif tanımlı matrisi aşağıdaki gibi elde ettim.

#Pozitif tanımlı matrisler benim elde ettiğimden çok daha detaylı yapıya sahiptirler.



n=4

S=np.zeros((n,n), dtype=np.uint32)



kosegen=np.random.randint(1,4)

other=np.random.randint(1,4)



while kosegen==other:

    other=np.random.randint(1,4)



for i in range(n):

    for j in range(n):

        if i ==j:

            S[i,j]=kosegen

        else:

            S[i,j]=other

            S[j,i]=other
L=np.linalg.cholesky(S)
print("S:\n",S,end="\n\n")

print("L:\n",L,end="\n\n")

print("L.L^T:\n",L.dot(L.T))

n=3

A=np.zeros((n,n),dtype=np.int32)

for i in range(n):

    for j in range(n):

        A[i,j]=np.random.randint(1,10)



egienvalues, egienvectors=np.linalg.eig(A)



print("Öz değerler:\n",egienvalues,end="\n\n")

print("Öz vektörler matrisi:\n",egienvectors, end="\n\n")

for i, egienvector in enumerate(egienvectors.T):

    print("{}. öz vektör:{}".format(i+1,egienvector))
n=1

print("egienvectors[:,{}]*egienvalues[{}]:{}".format(

    n,

    n,

    egienvectors[:,n]*egienvalues[n])

     )

print("A.dot(egienvectors[:,{}])        :{}".format(n,A.dot(egienvectors[:,n])))
LAMBDA=np.diag(egienvalues)

print("A:\n",A,end="\n\n")

print("Q:\n",egienvectors,end="\n\n")

print("LAMBDA:\n",LAMBDA,end="\n\n")

print("Q^-1:\n",np.linalg.inv(egienvectors),end="\n\n")

print("Q.LAMBDA.Q^-1:\n",egienvectors.dot(LAMBDA).dot(np.linalg.inv(egienvectors)))
n=3

m=4

A=np.zeros((n,m),dtype=np.int32)

for i in range(n):

    for j in range(m):

        A[i,j]=np.random.randint(1,10)



#scipy kütüphanesindeki svd fonksiyonunu kullanarak SVD 

#bileşenlerini elde ediliyor

U, singular_value, V=sp.linalg.svd(A)



row, col=A.shape

lesser=min(row,col)

SIGMA=np.zeros((row, col))



#SIGMA kare matris olmadığı için köşegen matrisi

#bu şekilde oluşturuyoruz

SIGMA[:lesser, :lesser]=np.diag(singular_value)





print("A:\n",A, end="\n\n")

print("U:\n",U, end="\n\n")

print("SIGMA:\n",SIGMA,end="\n\n")

print("V:\n",V, end="\n\n")

print(U.dot(SIGMA.dot(V)))
iris=pd.read_csv("../input/iris/Iris.csv")
iris.head()
sns.countplot(iris['Species'])
dataset=iris[iris['Species']!="Iris-virginica"]

sns.countplot(dataset['Species'])
dataset['Species']=dataset.Species.map({'Iris-setosa':-1,

              'Iris-versicolor':1})
X=dataset.drop(["Id","Species"],  axis=1)

y=dataset['Species']



print("X.shape:",X.shape)

print("y.shape:",y.shape)
X_train, X_test, y_train, y_test=train_test_split(X.values, 

                                                  y.values, 

                                                  stratify=y, 

                                                  test_size=0.4,

                                                 random_state=42) 
class Perceptron:

    

    

    def __init__(self, learning_rate, number_of_iterations, classes):

        self.learning_rate=learning_rate

        self.number_of_iterations=number_of_iterations

        self.w=None

        self.errors=[]

        if len(classes)!=2:

            raise ValueError("Number of class should be 2")

        else:

            self.class1=classes[0]

            self.class2=classes[1]

            self.thres=sum(classes)/len(classes)

            print("class1:",self.class1)

            print("class2:",self.class2)

            print("thres:",self.thres)

    

    def _get_error(self, update):

        if update==0.0:

            return 0

        return 1

    

    def fit(self, X_train, y_train):

        feature_size=X_train.shape[1]

        self.w=np.zeros(1+feature_size)

        

        for i in range(self.number_of_iterations):

            error=0

            

            for xi, yi in zip(X_train, y_train):

                update=self.learning_rate*(yi-self.predict(xi))

                self.w[1:]=self.w[1:]+update*xi

                self.w[0]=self.w[0]+update

                error=error+ self._get_error(update)

            

            self.errors.append(error)

        

    def net_input(self, X):

        

        net=np.dot(X, self.w[1:])+self.w[0]

        return net

    

    def predict(self, X):

        pred=self.step(X)

        return pred

    

    def step(self,X):

        return np.where(self.net_input(X)>=self.thres, self.class1, self.class2)

perceptron=Perceptron(learning_rate=0.01, number_of_iterations=10, classes=[1,-1])



perceptron.fit(X_train, y_train)

y_pred=perceptron.predict(X_test)
def performance_metrics(y_true, y_pred,

            accuracy=True, confusion_matrix=True, classification_report=True):

    if accuracy:

        print("Başarı oranı(%):",metrics.accuracy_score(y_true, y_pred)*100,end="\n\n")

    if confusion_matrix:

        print("Karışıklık Matrisi:\n",

              metrics.confusion_matrix(y_true, y_pred),end="\n\n")

   

    if classification_report:

        print("Sınıflandırma Raporu:\n",

              metrics.classification_report(y_true, y_pred),end="\n\n")
performance_metrics(y_true=y_test, y_pred=y_pred)
fig, ax=plt.subplots(1,2,figsize=(12,6))

sns.swarmplot(data=dataset,

              x="SepalLengthCm", 

              y="SepalWidthCm", 

              hue="Species", 

              ax=ax[0])

sns.swarmplot(data=dataset,

              x="PetalLengthCm", 

              y="PetalWidthCm", 

              hue="Species", 

              ax=ax[1])


X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])



y_and = np.array([[0], [0], [0], [1]])

y_and=np.array([0, 0, 0, 1])



perceptron=Perceptron(learning_rate=0.01, number_of_iterations=20, classes=[1,0])



perceptron.fit(X_and, y_and)

y_pred=perceptron.predict(X_and)



print("\n\nAND İçin Perceptron Modelinin Sınıflandırma Performansı\n")

performance_metrics(y_true=y_and, y_pred=y_pred)
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])



y_or = np.array([[0], [1], [1], [1]])



perceptron=Perceptron(learning_rate=0.01, number_of_iterations=20, classes=[1,0])



perceptron.fit(X_or, y_or)

y_pred=perceptron.predict(X_or)



print("\n\nOR İçin Perceptron Modelinin Sınıflandırma Performansı\n")

performance_metrics(y_true=y_or, y_pred=y_pred)
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])



y_xor = np.array([[0], [1], [1], [0]])



perceptron=Perceptron(learning_rate=0.01, number_of_iterations=20, classes=[1,0])



perceptron.fit(X_xor, y_xor)

y_pred=perceptron.predict(X_xor)



print("\n\nXOR İçin Perceptron Modelinin Sınıflandırma Performansı\n")

performance_metrics(y_true=y_xor, y_pred=y_pred)
fig, axarr=plt.subplots(nrows=1, ncols=3, figsize=(15,4))



#AND grafiği çizdiriliyor

axarr[0].scatter(x=X_and[:,0], y=X_and[:,1], color=['green', 'green', 'green', 'red'] )

axarr[0].plot([0,1],[1.2,0.5])



#OR grafiği çizdiriliyor

axarr[1].scatter(x=X_or[:,0], y=X_or[:,1], color=['green', 'red', 'red', 'red'] )

axarr[1].plot([0,1],[0.5,-0.2])



#XOR grafiği çizdiriliyor

axarr[2].scatter(x=X_xor[:,0], y=X_xor[:,1], color=['green', 'red', 'red', 'green'] )

axarr[2].plot([0,1],[0.5,0.5])



for ax, title in zip(axarr.flatten(),['AND', 'OR', 'XOR']):

    ax.set_xticks([0,1])

    ax.set_yticks([0,1])

    ax.set_title(title)

plt.show()
class MyPCA():

    def __init__(self, n):

        self.number_of_component=n

    

    def fit(self, X):

        self.M=np.mean(X.T,axis=1)

        

        C=X-self.M

        

        V=np.cov(C.T)

        

        self.eigen_values, self.eigen_vectors=np.linalg.eig(V)

        

        self.__calculate_explained_variance()

        

        self.__construct_projection_matrix()

    

    def transform(self, X):

        C=X-self.M

        X_new=self.projection_matrix.T.dot(C.T)

        return X_new.T

    

    def fit_transform(self,X):

        self.fit(X)

        return self.transform(X)

    

    def __construct_projection_matrix(self):

        eigen_pairs=self.__get_eigen_pairs()

        

        

        n_eigen_pairs=eigen_pairs[:self.number_of_component]

        

        sorted_eigen_vectors=[eigen_pair[1][:,np.newaxis] for eigen_pair in n_eigen_pairs]

        

        self.projection_matrix=np.hstack(sorted_eigen_vectors)

    

    def __get_eigen_pairs(self):

        eigen_pairs=list()

        for i in range(len(self.eigen_values)):

            eigen_pairs.append((np.abs(self.eigen_values[i]),self.eigen_vectors[:,i]))

        

        eigen_pairs.sort(key=lambda k:k[0], reverse=True)

        

        return eigen_pairs

    

    def __calculate_explained_variance(self):

        sum_of_eigen_values=sum(self.eigen_values)

        

        self.variance_=[(eigen_value/sum_of_eigen_values) for eigen_value in self.eigen_values]

        

        self.explained_variance_=np.cumsum(self.variance_)

        
A=np.arange(1,13).reshape(4,3)
my_pca=MyPCA(n=2)

my_pca.fit(A)

P=my_pca.transform(A)



print(A)

print("P:\n",P)

print(my_pca.eigen_vectors)
from sklearn.decomposition import PCA

pca=PCA(n_components=2)

pca.fit(A)

P=pca.transform(A)



print(A)

print("P:\n",P)

print(pca.components_)