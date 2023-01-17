# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/Iris.csv')

data.info()

#This data contains 150 rows and 6 columns.
data.corr ()
#There is not negative correlation between features.There is positive correlation  between PetalWidthCm and PetalLengthCm.
f,ax = plt.subplots(figsize=(16, 16))
sns.heatmap(data.corr(), annot=True, linewidths=.10, fmt= '.2f',ax=ax)
plt.show()

data.head(8)
data.columns
data.SepalLengthCm.plot(kind='line',color='orange',label='SepalLengthCm',linewidth=2,alpha=0.7,grid=True,linestyle=':')
data.PetalWidthCm.plot(kind='line',color='blue',label='PetalWidthCm',lw=4,alpha=0.5,grid=True,linestyle='-.')
plt.legend(loc='center right')
plt.xlabel('x axis')             
plt.ylabel('y axis')
plt.title('Line Plot')          
plt.show()
data.plot(kind='scatter',color='magenta',x='Id',y='PetalLengthCm',alpha=0.8,grid=True)
plt.xlabel('Id')
plt.ylabel('PetalLengthCm')
setosa=data[data.Species=="Iris-setosa"]
plt.hist(setosa.PetalLengthCm,bins=50)
plt.xlabel("PetalLengthCm Value")
plt.ylabel("Frequency")
plt.title("Histogram")
plt.show()
dictionary={'Ay':'Eylül','Gün':'Salı','Yıl':'2018'}
print(dictionary.keys())
print(dictionary.values())
dictionary['Ay ']='Ekim'
print(dictionary)
dictionary['Tarih']='25'
print(dictionary)

del dictionary['Tarih']
print(dictionary)
print('Mart' in dictionary)
print('2018' in dictionary)

print('Eylül' in dictionary)
print('Ay' in dictionary)
data2 = pd.read_csv('../input/Iris.csv')
series=data2['SepalLengthCm']
print(type(series))
data_frame=data2[['SepalLengthCm']]
print(type(data_frame))
print (7>8)
print (6!=4)

print(True or False)
print(True and False)
x=data2['SepalLengthCm'] >7
data2[x]
data[np.logical_and(data['SepalLengthCm']>7, data['PetalLengthCm']>5 )]
k = 0
while k != 10 :
    print('k = ',k)
    k +=1 
print(k,' = 10')
lis=[1,3,5,8,10]
for k in lis:
    print('k is: ',k)
print('')

for index, value in enumerate(lis):
    print(index," : ",value)
print('') 



dictionary={'meyve':'portakal','cicek':'ortanca'}
for key,value in dictionary.items():
    print(key," : ",value)
print('')
