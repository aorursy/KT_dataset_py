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
data = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")

data.head()
data.columns
def average():

    """ Return value of average age docstring denemesi """ 

    avrg = sum(data.age)/len(data.age)

    return avrg

print(average())

print(average.__doc__)

help(average)
age1 = data[['age']][4:5]

print(age1)

def globalvar():

    age1*2

    age2 = (age1*2)+1

    return age2

print(globalvar())
def localvar():

    age1 = data[['age']][1:2]

    age2 = (age1*2)+1

    return age2

print(localvar())

print(age1)

#NESTED FUNCTION : iç içe yazılmış fonksiyonlara denir. thalach verilerinin ortalamasını daha kolay yöntemle alabilirdik

#fakat NESTED FUNCTION işlemini göstermek açısından böyle yaptım.



def ortalama():

    """ thalach Ortalamasını return eder """

    def toplam(): # Nested Function kısmı. En içteki function....

        """ NESTED function kısmı thalach veri setinin toplamının alındığı kısım """

        tplm = 0

        for index,value in data[['thalach']].iterrows():

            tplm = tplm + value

        return tplm

    return toplam()/len(data.thalach)

print(ortalama())

            

            
def den(age1,age2=52):

    c = age1 + age2

    print(c," ",age1," ",age2)

    return c

print(den(50))

print(den(70,70))
data["real_sex"] = ["man" if i != 0 else "women" for i in data['sex']]

data.loc[0:50, ["sex","real_sex"]]