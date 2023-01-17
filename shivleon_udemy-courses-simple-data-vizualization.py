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
#importing necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mat
#Reading the csv file

datau=pd.read_csv(os.path.join(dirname, filename))

datau
datau.info()
print(datau.shape, datau.ndim, datau.size)
#lets see the number of unique values in each column

def unique_in_csv(data):

    for o in data:

        print("{0}={1}".format(o,data[o].unique()), end="\n\n")

        

unique_in_csv(datau)
#lets make a deep copy 1st

datau1=datau.copy(deep=True).reset_index()
plt.hist(datau1["course_id"], edgecolor="yellow")
#Lets see Number of lectures we have

plt.hist(datau1["num_lectures"])

plt.show()
#Lets see Number of Subscribers we have

plt.hist(datau1["num_subscribers"])

plt.show()
#Lets see Number of Reviews we have

plt.hist(datau1["num_reviews"])

plt.show()
plt.figure(figsize=(20,4))

plt.hist(datau1["content_duration"].sort_values(), edgecolor="red")

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(16,5))

plt.hist(datau1["price"].sort_values(), edgecolor="red")

plt.xticks(rotation=90)

plt.show()
datau1["price"].unique()
uni=len(datau1["course_id"].unique())

unt=len(datau1["course_title"].unique())

unr=len(datau1["url"].unique())

print(uni,unt,unr)
plt.bar(["course_id","course_title","url"],[uni,unt,unr])
datau1["is_paid"].value_counts()
T,F=datau1["is_paid"].value_counts()

labels="Paid","Unpaid"

explode=(0,0)

fig, ax=plt.subplots()

ax.pie([T,F], explode=explode, labels=labels, autopct="%1.1f%%", shadow=True, startangle=90)

ax.axis('equal')

plt.show()
labels="Web Develpoment","Business Finance","Musical Instruments","Graphics Design"

explode=(0.09,0.09,0.09,0.09)

fig, ax=plt.subplots()

ax.pie(datau1["subject"].value_counts(), explode=explode, labels=labels, autopct="%1.1f%%", shadow=True, startangle=90)

ax.axis('equal')

plt.show()
u=datau1["level"].value_counts()
#For understanding the composition of level, we wull make a Tree map

import squarify #for making treemap, we need squarify

plt.figure(figsize=(20,8))

labels=["ALl Levels \n"+str(u[0]),"Beginner Level \n"+str(u[1]),"Intermediate Level \n"+str(u[2]),"Expert Level\n"+str(u[3])]

colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]

squarify.plot(sizes=datau1["level"].value_counts(),color=colors, label=labels, alpha=.8)