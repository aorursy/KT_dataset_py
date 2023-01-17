# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dat=[1,2,3]

arr=np.array(dat)

arr
dat2=[[1,2,3],[4,5,6],[7,8,9]]

arr2=np.array(dat2)

arr2
print(arr2[2][1],arr2[2],sep="--")
np.arange(10,20,2) #10dan 20ye kadar array oluşturur, 2şer aralıklarla
np.arange(25).reshape(5,5) 

#arange 0-25 arası değer arrayledi

#reshape 5x5 matrisledi
np.zeros(5)  #0lardan oluşan array verdiğin değer kadar
np.zeros((3,3))  # 3x3 lük her elemanı 0
np.ones(6)   # 1lerden oluşan array, verdiğin değer kadar
np.ones((4,4))  # 4x4lük her elemanı 1, verdiğin değer kadar
np.eye(5)   #5x5 lik birim matris
np.linspace(0,100,5)  # 0ile100 arasını 5 eşit parçaya bölerek arrayler
np.random.randint(1,50,6)  # 1ile500 arası rasgele 6 değer arrayler
np.random.rand(6)  #0ile1 arası rasgele sayı arrayler
np.random.randn(4)  #gauss dist göre değer arrayler
arr2.max()

arr2.min()

arr2.mean()

arr2.sum()

arr2.argmax()  #max değerin indexini verir

arr2.argmin()

np.linalg.det(arr2)  #detarminantı verir
arr=np.array([1,2,3,4,5])



arr1=arr    # bu şekildeaynı veri üzerinden hareket edecekler değişklikler 2sini de etkiler.



arr2=arr.copy()  # arr2 hafızada farklı yer kaplar duplicate edersin.
arr>2    # koşulu bu şekilde yazarsan her ddeğer için true false döndürür
arr[arr>3]  # bu şekilde filtrelemiş olursun true olanları döndürür
new_arr=np.arange(25).reshape(5,5)

new_arr
# (12,13) ve (17,18) almaya çalışalım indeks işlemleri önemli

new_arr[2:4,2:4]