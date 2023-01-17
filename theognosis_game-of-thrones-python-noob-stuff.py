# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output



print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
f=pd.read_csv("../input/character-deaths.csv")



Book1 = 0

Book2 = 0

Book3 = 0

Book4 = 0

Book5 = 0

deaths_bybook=f["Book of Death"]

for deaths in deaths_bybook:

    if deaths == 1.0:

         Book1 = Book1 + 1

print(Book1)



deaths_bybook=f["Book of Death"]

for deaths in deaths_bybook:

    if deaths == 2.0:

         Book2 = Book2 + 1

print(Book2)



deaths_bybook=f["Book of Death"]

for deaths in deaths_bybook:

    if deaths == 3.0:

         Book3 = Book3 + 1

print(Book3)



deaths_bybook=f["Book of Death"]

for deaths in deaths_bybook:

    if deaths == 4.0:

         Book4 = Book4 + 1

print(Book4)



deaths_bybook=f["Book of Death"]

for deaths in deaths_bybook:

    if deaths == 5.0:

         Book5 = Book5 + 1

print(Book5)

    

    
Allbooks = [Book1, Book2, Book3, Book4, Book5]

plt.plot(Allbooks)
labels = 'Got', 'CoK', 'SoS', 'FfC', 'DwD'

sizes = [49, 73, 97, 27, 61]

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'red']

explode = (0, 0, 0.1, 0, 0)  # explode  slice

 

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=140)

 

plt.axis('equal')

plt.show()
Alldeaths = Book1 + Book2 + Book3 + Book4 + Book5

print(Alldeaths)