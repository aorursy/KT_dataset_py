# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Defining data frame

cdf = pd.read_csv('/kaggle/input/corona-virus-test-numbers-in-turkey/turkey_covid19_all.csv')

cdf.head(20) #Giving us first twenty values in data frame
cdf.info() #Informations about data frame

cdf.columns # Columns Objects
f,ax = plt.subplots(figsize=(13,13)) #Maps size

sns.heatmap(cdf.corr(),annot=True, linewidths=.5 , fmt='.1f',ax=ax) #Correlation map 





#annot true -> Shows numbers in each box

#linewidth -> Line widths

#fmt -> How many value will show after zero
#Line

cdf.Deaths.plot(kind = 'line' ,color = 'purple', label = 'Deaths', linewidth = 1 ,alpha = .5 ,  grid = False , linestyle = '-')

plt.title('Coronavirus Deaths in Turkey')

plt.xlabel('Days')

plt.ylabel('Numbers of Deaths')
#Scatter 

scatter = cdf.plot(kind='scatter' , x = 'Deaths' , y = 'Tests' , alpha = .5 , color = 'brown')

plt.xlabel('Deaths')

plt.ylabel('Tests')

plt.title('Relations Between Deaths and Tests')
#Histogram

#bins -> number of bars

cdf.Recovered.plot(kind='hist',bins = 10, figsize = (10,10))

plt.title('Histogram of Recovered People')

plt.show()
dic = {'Science Fiction':'Interstellar', 'Comedy': 'Hangover',

       'Horror':'The Conjuring','Drama':'The Pianist',

       'Action':'The Fast and Furios','Animation':'Cars'}

print(dic.keys())

print(dic.values())

dic['Drama']='The Green Mile' #Changing drama's value



dic['Western']='The Revenant' #Adding a new key and value



del dic['Comedy'] #Deleting Comedy in dictionary

print(dic)

print('Horror' in dic) #Checking wheather horror in dictionary or not

dic.clear() #Clear everything in the dictionary

print(dic)
#Comparison Operators

print(10 < 5)

print(1==1)

print(5 != 2)

print( 0> 1)

# Boolean operators

print(False & True)

print(True or False)
cdf = pd.read_csv('/kaggle/input/corona-virus-test-numbers-in-turkey/turkey_covid19_all.csv')
#Filtering Pandas Data Frame

filter_cdf = cdf['Deaths'] > 100

print(filter_cdf)

cdf[filter_cdf]
#Using logical_and for cdf(Coronavirus Data Frame)

cdf[np.logical_and(cdf['Deaths']>100, cdf['Recovered']<100)]
#Alternative way for filtering data frame

cdf[(cdf['Deaths']>100) & (cdf['Recovered']<100)]

#Counting the numbers between -5 and 10 (-5 and 10 included)



a = -5

while a <= 10:

    print('a is:',a)

    a +=1
list= [4,5,3,2,1,5,6,9]

for b in list:

    print('b is:',b)

print('')



#Enumerate index and value of list

#index : value = 0:4 , 1:5 , 2:3 , 3:2 , 4:1 , 5:5 , 6:6 , 7:9

for index, value in enumerate(list):

    print(index,':',value)

print('')



# For dictionaries

# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.

dic = {'Science Fiction':'Interstellar', 'Comedy': 'Hangover',

       'Horror':'The Conjuring','Drama':'The Pianist',

       'Action':'The Fast and Furios','Animation':'Cars'}

for key,value in dic.items():

    print(key,':',value)

print('')



    