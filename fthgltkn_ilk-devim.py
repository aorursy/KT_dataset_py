# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#READ AND COLUMNS





data=pd.read_csv('../input/Iris.csv')

print(data.columns)
#GENEL VERİLER





data.info()
data.describe()
data.corr
#CORRELATİON MAP



f,ax = plt.subplots(figsize=(12, 12))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(15)
#LİNE PLOT





data.SepalWidthCm.plot(kind="line",color="red",linewidth=1,label="SepalWidthCm",alpha=0.8 ,grid="True",linestyle= ":")

data.PetalLengthCm.plot(kind="line",color="blue",linewidth=1,label="PetalLengthCm",alpha=0.8 ,grid="True",linestyle= "-.")

plt.legend(loc='upper right') 

plt.xlabel=("x axis")

plt.ylabel=("y axis")

plt.title=("line plot")

plt.show()

#SCATTER PLOT





data.plot(kind='scatter',x='SepalWidthCm',y='PetalLengthCm',grid=True,color='red',alpha=0.5)

plt.xlabel('SepalWidthCm')

plt.ylabel=('PetalLengthCm')

plt.title=('SepalWidthCm and PetalLengthCm ')

plt.show()
#HİSTOGRAM PLOT





data.SepalWidthCm.plot(kind='hist',bins=40,figsize=(8,8),color='red')

data.PetalLengthCm.plot(kind='hist',bins=40,figsize=(8,8),color='blue')

plt.show()
#DİCTİONARY





dicti = {'kalem':'pencil','masa':'table','telefon':'phone','bilgisayar':'computer'}

print(dicti.keys())

print(dicti.values())

dicti['kalem']='hair'    #update

print(dicti)



dicti['silgi']='eraser'  #add new entry

print(dicti)



del dicti['kalem']      #remove entry

print(dicti)



print('kalem' in dicti)  #check



dicti.clear()   #remove all







#PANDAS





series = data['SepalWidthCm']        # data['SepalWidthCm'] = series

print(type(series))

data_frame = data[['PetalLengthCm']]  # data[['PetalLengthCm']] = data frame

print(type(data_frame))

print(3==3)



print(3!=2)



print(True and False)

print(True or False)
a=data['SepalWidthCm']>3

data[a].head()
data[(data['SepalWidthCm']>3) & (data['PetalLengthCm']>1.5)].head()
#WHİLE AND FOR LOOPS







lis = [1,2,3,4,5,6]

for i in lis:

    print('i is: ',i)

print('')
for index, value in enumerate(lis):

    print(index," : ",value)

print('')   
dicti = {'kalem':'pencil','masa':'table','telefon':'phone','bilgisayar':'computer'}

for key,value in dicti.items():

    print(key," : ",value)

print('')
for index,value in data[['SepalWidthCm']][0:1].iterrows():

    print(index," : ",value)