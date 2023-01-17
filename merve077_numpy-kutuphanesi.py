# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# python list

python_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# numpy array

numpy_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

print("Python listesi :")

print(python_list)

print("Numpy dizisi :")

print(numpy_array)
numpy_array1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

print(numpy_array1.ndim)

numpy_array2 = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])

print(numpy_array2.ndim)

numpy_array1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

print(numpy_array1.shape) #10 tane elemandan oluşan 1 boyutlu bir dizi(vektör).

print(numpy_array1.ndim)
numpy_array2 = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])

print(numpy_array2.ndim)

print(numpy_array2.shape)

# 1 satır ve 10 sütundan oluşan 2 boyutlu bir dizi(matris).
numpy_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

print(numpy_array.reshape(1,10))

print(numpy_array.reshape(10,1))

print(numpy_array.reshape(5,2))

print(numpy_array.reshape(2,5))

np.arange(0,10,3)

np.arange(10)

np.arange(0,10,1)
numpy_array= np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

numpy_array = numpy_array.reshape(5,2)

print(numpy_array)



#Dizinin herhangi bir satırını seçmek

#1.satır

first_row = numpy_array[0]

#1. ve 2. satır 

first_and_second_rows = numpy_array[0:2]

print(first_row)

print(first_and_second_rows)



#Dizinin herhangi bir kolonunu seçmek

#1. sütun

first_column = numpy_array[:,0]

#1. ve 2. sütun

first_and_second_column = numpy_array[:,0:2]

print(first_column)

print(first_and_second_column)



#Dizinin herhangi bir elemanını seçmek

selecting_item = numpy_array[3,1]

print(selecting_item)

numpy_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

numpy_array = numpy_array.reshape(5,2)



print(numpy_array)

print(numpy_array[::-1])

print(np.zeros((5,4)))

#np.ones(): zeros() fonksiyonuna benzer şeklide verilen büyüklükte 1'lerden oluşan bir matris döndürür.

print(np.ones((3,3,3)))

print(np.eye(4))
numpy_array = np.array([0,1, 2, 3, 4, 5, 6, 7, 8, 9])

numpy_array = numpy_array.reshape(5,2)



#Satır bazlı birleştirme

print(np.concatenate([numpy_array, numpy_array], axis =0))



#Sütun bazlı birleştirme

print(np.concatenate([numpy_array, numpy_array], axis =1))
numpy_array = np.array([0,1, 2, 3, 4, 5, 6, 7, 8, 9])

numpy_array = numpy_array.reshape(5,2)

print(numpy_array)



print(numpy_array.max())



print(numpy_array.min())



print(numpy_array.sum())



#Satırların toplamı

print(numpy_array.sum(axis = 1))



#Sütunların toplamı

print(numpy_array.sum(axis = 0))
numpy_array = np.array([0,1, 2, 3, 4, 5, 6, 7, 8, 9])

print(numpy_array.mean())

print(np.median(numpy_array))

print(numpy_array.var())

print(numpy_array.std())



#!!! Median kullanımının diğerlerinden farklı olduğu dikkatinizi çekmiş olabilir.

#Bunun sebebi ndarray nesnesinin np.median(ndarray) fonksiyonuna sahip olması fakat ndarray.median() metoduna sahip olmamasıdır.
numpy_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

numpy_array = numpy_array.reshape(3,3)

#dizimizi 3x3 lük, 2 boyutlu bir matrise dönüştürdük.



print(numpy_array)



print(numpy_array + numpy_array)



print(numpy_array - numpy_array)



print(numpy_array * numpy_array)



print(numpy_array / numpy_array)



print(numpy_array + 5)



print(numpy_array * 2)



numpy_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

numpy_array = numpy_array.reshape(3,3)

#dizimizi 3x3 lük, 2 boyutlu bir matrise dönüştürdük.

print(np.sin(numpy_array))

print(np.cos(numpy_array))

print(np.sqrt(numpy_array)) 

print(np.exp(numpy_array))

print(np.log(numpy_array))
numpy_array = np.array([0,1, 2, 3, 4, 5, 6, 7, 8, 9])

numpy_array1 = numpy_array.reshape(5,2)

numpy_array2 = numpy_array.reshape(2,5)

print(np.dot(numpy_array1,numpy_array2))
numpy_array = np.array([0,1, 2, 3, 4, 5, 6, 7, 8, 9])

numpy_array = numpy_array.reshape(5,2)

print(numpy_array)
print(numpy_array.T)
dizin=np.full((2,3),8)

dizin
import numpy as np

liste=[1,2,3] #normal python listesi

numpyliste=np.array([1,2,3]) #numpy dizisi

for e in liste:

 print(e)

 

for e in numpyliste:

 print(e)









import numpy as np

liste=[1,2,3]

numpyliste=np.array([1,2,3])

print(liste+liste)

print(numpyliste+numpyliste)

#listeler toplandığında yan yana yazar string gibi kabul eder

#numpy diziler toplandığında ise matematiksel olarak toplar



import numpy as np

liste=[1,2,3]

numpyliste=np.array([1,2,3])

print(3*liste)

print(3*numpyliste)



import numpy as np

a= np.array([5,6,9])

print(a[0])

print(a[1])



import numpy as np

a=np.array([[1,2],[3,4],[5,6]])

print(a.ndim)

b=np.array([5,6,9])

print(b.ndim)



import numpy as np

a=np.array([[1,2],[3,4],[5,6]])

print(a.shape)

print(a.reshape(2,3))



import numpy as np

print(np.linspace(3,18,15))

# 3 ile 18 dahil 3-18 arasında 15 adet sayı üretir



import numpy as np

a=np.array([[1,2],[3,4],[5,6]])

print(a.ravel())





import numpy as np

a=np.array([[1,2],[3,4],[5,6]])

print(a.min())

print(a.max())

print(a.sum())

#axis matristeki verilerin sütunlarını toplar

print(a.sum(axis=0))

#axis matristeki verilerin satırlarını toplar

print(a.sum(axis=1))

#tüm elemanların kare köklerini alır.

print(np.sqrt(a))

#tüm elemanların standart sapmasını alır

print(np.std(a))





import numpy as np

a=np.array([[1,2],[3,4]])

b=np.array([[5,6],[7,8]])

print(a+b)



import numpy as np

a=np.array([[1,2],[3,4]])

b=np.array([[5,6],[7,8]])

print(a*b)










