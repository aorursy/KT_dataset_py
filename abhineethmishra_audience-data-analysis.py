# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

'''

This is an example of some basic preliminary data analysis of the traffic coming to a website(say youtube).

The Dataset is randomly generated, so the insights might not be authentic/pertain to how data is in the real world.

'''

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#creating a dataset with features audience age, location, video category, gender, device category

import random as r

r.seed(47)

location_set=['India','USA','China','Russia','Israel','UK','Germany','UAE',"Poland",'Nepal','Nigeria','Vietnam','Australia']

device_cat=['Web','Mobile']

video_cat_set=["Gaming","TV","Movies","Trailer","Music","Tech","Review"]

gender_set=['Male','Female']

location=[]

age=[]

dev_cat=[]

video_cat=[]

gender=[]

count=0

#generating the data

while count<1000:

    age.append(r.randint(10,80))

    location.append(r.choice(location_set))

    dev_cat.append(r.choice(device_cat))

    video_cat.append(r.choice(video_cat_set))

    gender.append(r.choice(gender_set))

    count+=1

#age=[x for x in random.randint(15,80)]

#create visitor frequency
#after generating the dataset, we try to get some insights

import matplotlib.pyplot as plt

from collections import Counter

#showing the distribution of viewers with their location

loc_freq=[]

for i in location_set:

    loc_freq.append(Counter(location)[i])

plt.figure(figsize=(20, 10))

plt.xlabel('Countries',fontsize=30)

plt.ylabel('Users',fontsize=30)

plt.bar(location_set,loc_freq)
#splitting age into categories

age_cat=['10-15','15-20','20-25','25-30','30-35','35-40','40-45','45-50','50-55','55-60','60-65','65-70','70-75','75-80']

age_freq=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]

count=0

#splitting age into categories

age_cat=['10-15','15-20','20-25','25-30','30-35','35-40','40-45','45-50','50-55','55-60','60-65','65-70','70-75','75-80']

age_freq=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]

count=0

for idx in range(len(age)):

    if age[idx]>=10 and age[idx]<15:

        age_freq[0]+=1

    elif age[idx]>=15 and age[idx]<20:

        age_freq[1]+=1 

    elif age[idx]>=20 and age[idx]<25:

        age_freq[2]+=1

    elif age[idx]>=25 and age[idx]<30:

        age_freq[3]+=1

    elif age[idx]>=30 and age[idx]<35:

        age_freq[4]+=1

    elif age[idx]>=35 and age[idx]<40:

        age_freq[5]+=1

    elif age[idx]>=40 and age[idx]<45:

        age_freq[6]+=1

    elif age[idx]>=45 and age[idx]<50:

        age_freq[7]+=1

    elif age[idx]>=50 and age[idx]<55:

        age_freq[8]+=1

    elif age[idx]>=55 and age[idx]<60:

        age_freq[9]+=1

    elif age[idx]>=60 and age[idx]<65:

        age_freq[10]+=1

    elif age[idx]>=65 and age[idx]<70:

        age_freq[11]+=1

    elif age[idx]>=70 and age[idx]<75:

        age_freq[12]+=1

    elif age[idx]>=75 and age[idx]<=80:

        age_freq[13]+=1
plt.pie(age_freq,labels=age_cat)
#the above pie chart shows that there is not much difference in the number of viewers based on their age

mob=0

web=0

for i in dev_cat:

    if i=='Mobile':

        mob+=1

    else:

        web+=1

lst=[web,mob]

print("Web",web,"Mobile",mob)
#The chart shows that there is an almost equal number of web and mobile users(because of random function)

#manipulating web and mobile for a relaistic view

plt.pie([600,400],labels=['Web Users','Mobile Users'])
#number of male and female visitors

m=0

f=0

for i in gender:

    if i=='Male':

        m+=1

    else:

        f+=1

print("Male",m,"Female",f)

lst=[m,f]

plt.pie(lst,labels=['Male','Female'])
#Looks like the number of female visitors is slightly higher than number of male visitors

cat_freq=[]

for i in video_cat_set:

    cat_freq.append(Counter(video_cat)[i])

plt.figure(figsize=(20, 10))

plt.xlabel('Category',fontsize=30)

plt.ylabel('Viewers',fontsize=30)

plt.bar(video_cat_set,cat_freq)