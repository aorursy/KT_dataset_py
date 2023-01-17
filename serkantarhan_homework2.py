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
data=pd.read_csv ("/kaggle/input/fifa19/data.csv")  #datayı oluşturduk
# oyuncuların Overall değerlerini 7 kategoriye ayıralım ve gruplayalım



df=pd.read_csv("/kaggle/input/fifa19/data.csv") 

threshold1 = sum(data.Overall) / len(data.Overall) + 25

threshold2 = sum(data.Overall) / len(data.Overall) + 15

threshold3 = sum(data.Overall) / len(data.Overall) + 5

threshold4 = sum(data.Overall) / len(data.Overall)

threshold5 = sum(data.Overall) / len(data.Overall) - 5

threshold6 = sum(data.Overall) / len(data.Overall) - 15

threshold7 = sum(data.Overall) / len(data.Overall) - 25

print("average1" ,threshold1)

print("average2" ,threshold2)

print("average3" ,threshold3)

print("average4" ,threshold4)

print("average5" ,threshold5)

print("average6" ,threshold6)

print("average7" ,threshold7)





data["Oyuncu_Overall_Sınıfı"]= ["SuperStar" if i>=threshold1 else "Star" if threshold1> i >= threshold2 else "Yetenekli" if threshold2> i >= threshold3 else "Vasat" if threshold3> i >= threshold4 else "Gelişmez" if threshold4> i >= threshold5 else "Kötü" if threshold5> i >= threshold6 else "Rezil" for i in data.Overall]

data.loc[:10000,["Oyuncu_Overall_Sınıfı","Overall"]]

# nested Funtion

# bi r bölme işleminin sonucunun kare kökünü hesapladığımız iç içe fonksiyon tanımlayalım

def a ():

    def divide ():

        """ divide to local value"""

        x = 100

        y = 4

        z = x // y

        return z    

    return divide() **.5

print(a())
#Anonymous Function

# bir liste oluşturup o listedeki tüm itemlara fonksiyonumuzu uygulatıp kambda fonksiyonyu ile sonuç alalım

# aylık masraf tutarı $32.600 ise

# Yıllık cirolardan aylık net kar tutarlarını bulalım

Yearly_Sales_Amounts_List = [1000000, 2000000, 3000000, 4000000, 5000000,6000000]

x = map (lambda x:x//12 - 32600,Yearly_Sales_Amounts_List)



print ('$',list(x))