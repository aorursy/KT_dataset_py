# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



file ="../input/all.csv"

df=pd.read_csv(file,sep=",")

print(type(df))
plt.figure(figsize=(10,5))

count =df['State'].value_counts()

count.plot("bar")

plt.xlabel("State name",size=20)

plt.ylabel("No. of cities", size=20)

plt.title("Cities per state", size= 20)

plt.savefig("census1.jpg")





print(count.values)

person=df['Persons']

cols = ['Persons','Below.Primary', 'Primary', 'Middle', 'Matric.Higher.Secondary.Diploma',

                'Graduate.and.Above']

temp = df[cols + ['State']].groupby('State').sum()



density =temp.iloc[:,1].values/count.values

#plt.figure(figsize=(10,8))



#plt.ylabel("Density", size=20)

#plt.title("Totals Persons per city in a State", size= 20)



#plt.bar(count.values, density,color="red")

#plt.xticks(count.values, count.index)

#plt.savefig("census2.jpg")

print(count+density)

# Scattered plot of count per city

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



file ="../input/all.csv"

df=pd.read_csv(file,sep=",")



plt.figure(figsize=(10,5))

count =df['State'].value_counts()

plt.xlabel("State name",size=20)

plt.ylabel("No. of cities", size=20)

plt.title("Cities per state", size= 20)

plt.savefig("census1.jpg")

plt.scatter(range(len(count)), count)

plt.show()