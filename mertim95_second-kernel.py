# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
datta = pd.read_csv("/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv")

datta.info()
print(datta) 
datta.columns
list1 = datta.country.loc[:20]

list2 = datta.age.loc[:20]

list3 = datta.sex.loc[:20]

list4 = datta.suicides_no.loc[:20]

list5 = datta.year.loc[:20]

new_list = zip(list1,list2,list3,list4,list5)

print(new_list)

alist = list(new_list)

#print(alist) # you may use this one,too.

for i in alist:

    print(i)    # this looks better.
# a trial over unzip

not_zip = zip(*alist)

not_list1,not_list2,not_list3,not_list4 = list(not_zip)

print(not_list1)

print(not_list2)

print(not_list3)

print(not_list4)

print(type(not_list2))

print(type(list(not_list1)))

print(list(not_list1))
ilk="Bursa"

sayac=0

ulkeler=[]

for i in datta.country:

    if(i!=ilk):

        ulkeler.append(i)

        ilk=i

        sayac+=1        

print("")

print("There are",sayac,"countries in the list.")

print(ulkeler)

# To see which countries are involved.....



#for i in ulkeler:

#    print(i)      # for a better view.  
CompObject = "Albania"

intihar = 0

ListIndex = 0

intiharlar = []

for i in datta.country:

    if i == CompObject:

        intihar += datta.suicides_no[ListIndex]

    else:

        intiharlar.append(intihar)

        intihar = 0

        intihar += datta.suicides_no[ListIndex]

        CompObject = i

    ListIndex += 1

intiharlar.append(intihar)  # if do not add this, I will not be able to see the rate of Uzbekistan(the last country.)

print(intiharlar)
listCountries = ulkeler

listsuicides = intiharlar

nelist = zip(listCountries,listsuicides)

print(nelist)

ulkeveintihar = list(nelist)

print("Countries and Total Suicides")

print("")

for i in ulkeveintihar:

    print(i)

threshold = sum(datta.suicides_no)/len(datta.suicides_no)

print("Threshold is: ",threshold)

datta["Derece"] = ["Low" if i<threshold else "High" for i in datta.suicides_no]

datta.loc[:10000,["Derece","country","suicides_no","age","year"]]