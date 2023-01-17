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
#importing python libraries

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

#reading csv from the path came as output from first line of code

df=pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')#converting the csv into data frame

print(df.head(3))
#getting the basic info about the data set.

#We can get rough information about the whether null vaules are present or not

print(df.info())
#we will get information about count, mean, standard deviation & percentile stats

print(df.describe())
#crossverifying the null value count

df['species'].isnull().value_counts()
#getting the unique value counts

df['species'].value_counts()
#getting the mean(sum of values/no. of points) value according to group

df1=df.groupby(df['species']).mean()

df1['sepal_length']
#getting the unique values in species to map into simple plot

arr1=df['species'].unique()

arr1
#above code will reurn in array format

type(arr1)
#converting the array into series

df2=pd.Series(arr1)

type(df2)
#Matplotlib simple plot

plt.plot(df2,df1['sepal_length'],'b-H')

plt.xlabel('Species')

plt.ylabel('Average Sepal_length')

plt.title('Species vs. Average sepal_length')
#Object oriented plotting, with figure & axes



#df2 vs df1['sepal_length']

#df2 vs df1['petal_length']



#same axis only



fig=plt.figure()            #creating figure i.e. blank canvas

ax1=fig.add_axes([0,0,1,1]) #adding axes

ax1.plot(df2,df1['sepal_length'],'gD--')

ax1.plot(df2,df1['petal_length'],'ys--')

ax1.set_xlabel('Species')

ax1.set_ylabel('Length')

ax1.set_title('Species vs Sepal_length')
#multiplot

#species vs petal_length 

#species vs petal_width



plt.subplot(221)

plt.plot(df2,df1['sepal_width'],'ms--')

plt.title('species vs petal_length ')

plt.xticks(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])







plt.subplot(222)

plt.plot(df2,df1['petal_width'],'mD--')

plt.title('species vs petal_width ')

#if any graph overlaps each other due to improper row, col, index --- then it would remove previous graph
#multiplot with object oriented

fig1=plt.figure()



ax3=fig1.add_subplot(111) #larger graph

ax3.plot(df2,df1['petal_length'])

ax3.set_title('Pental_length')

                 

ax4=fig1.add_subplot(224,facecolor='y') #smaller graph

ax4.plot(df2,df1['petal_width'],'r')

ax4.set_title('Pental_width')



#barplot Species vs Avg. Petal_Width

fig2=plt.figure()

ax5=fig2.add_axes([0,0,1,1])

ax5.bar(df2,df1['petal_width'])

ax5.set_xlabel('Species')

ax5.set_ylabel('Avg.Petal_Width')

ax5.set_title('Species vs Avg. Petal_Width')
#histogram  Petal_length

fig3=plt.figure()

ax6=fig3.add_axes([0,0,1,1])

ax6.hist(df['petal_length'])

ax6.set_ylabel('Petal_length')

ax6.set_title('Distribution for Petal_Width')
#histogram sepal_length

fig3=plt.figure()

ax6=fig3.add_axes([0,0,1,1])

ax6.hist(df['sepal_length'])

ax6.set_ylabel('sepal_length')

ax6.set_title('Distribution for sepal_length')
#scatter sepal length

fig4=plt.figure()

ax7=fig4.add_axes([0,0,1,1])

ax7.scatter(df['species'],df['sepal_length'])

ax6.set_xlabel('Species')

ax6.set_ylabel('sepal_length')

ax6.set_title('Distribution for sepal_length')
#boxplot for sepal_length & petal_length 

fig5=plt.figure()

ax8=fig5.add_axes([0,0,1,1])

ax8.boxplot([df['sepal_length'],df['petal_length']])



#no outlier found
#violinplot for petal_length & petal_width

fig5=plt.figure()

ax8=fig5.add_axes([0,0,1,1])

ax8.violinplot([df['petal_length'],df['petal_width']])

ax8.set_title('Petal_length & Petal_width')