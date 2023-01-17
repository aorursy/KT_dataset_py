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
data = pd.read_csv("../input/us-major-league-soccer-salaries/mls-salaries-2017.csv")#Verisetini okuma
data #Verisetine göz atma
data.head(10) # ilk 10 veri
len(data.index) # Satır sayısı 
data["base_salary"].mean() # Ortalama Maaş
data["base_salary"].max() # En yüksek maaş
data[data["guaranteed_compensation"] == data["guaranteed_compensation"].max()] # En yüksek Tazminata sahip kişi
data[data["guaranteed_compensation"] == data["guaranteed_compensation"].max()]["last_name"].iloc[0] ## En yüksek Tazminata sahip kişinin soyadi
Gonz_position = soccer[data["last_name"] == "Gonzalez Pirez"]["position"].iloc[0]

if(Gonz_position == "F"):

    print("Forvet")

elif(Gonz_position == "M"):

    print("Orta Saha")

elif(Gonz_position == "D"):

    print("Defans")

else:

    print("Diğer")

data.groupby("position")["base_salary"].mean() #Pozisyona göre ortalama maaşlar
data["position"].nunique() #kaç farklı pozisyon(mevki) var.
data["position"].value_counts() # Her mevkideki oyuncu sayısı
data["club"].value_counts()
#Soyadında 'SON' geçen futbolcuları bulur

def contains_word(last_name):

    return "son" in last_name

data[data["last_name"].apply(contains_word)]