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
dataframe = pd.read_csv("../input/Iris.csv")
dataframe.info()
dataframe.corr()
f, ax = plt.subplots(figsize=(12,12))
sns.heatmap(dataframe.corr(), annot=True, linewidths=.5,fmt='.1f', ax=ax)
plt.show()
dataframe.head(10)
dataframe.tail(10)
dataframe.columns
dataframe.Species.unique() # data frame columns 'Species' unique()
setosa = dataframe[dataframe.Species == "Iris-virginica"] # data frame setosa
virginica = dataframe[dataframe.Species == "Iris-virginica"] # data frame virginica
versicolor = dataframe[dataframe.Species == "Iris-versicolor"] # data frame versicolor
#setosa.describe()
#virginica.describe()
#versicolor.describe()
#data =  dataframe.drop(["Id"], axis=1)
dataframe['PetalLengthCm'].plot(kind= 'line', color='red', grid=True,label='PetalLengthCm', alpha= 0.7,linewidth=1)
dataframe['PetalWidthCm'].plot(kind= 'line', color= 'blue', grid=True, alpha=0.7, label='PetalWidthCm', linewidth=1)
dataframe['SepalWidthCm'].plot(kind= 'line', color= 'green', grid=True, alpha=0.7, label = 'SepalWidthCm', linewidth=1)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.show()
# Line Plot
plt.plot(setosa.Id,setosa.PetalLengthCm,color="green",linewidth=1, label= "setosa")
plt.plot(virginica.Id,virginica.PetalWidthCm, color="blue", label= "virginica",)
plt.plot(versicolor.Id,versicolor.SepalWidthCm,color="red", label= "versicolor")
plt.legend(loc="upper right")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Line Plot")
plt.show()
plt.plot(setosa.Id,setosa.PetalLengthCm,color="green",linewidth=1, label= "setosa")
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.show()
plt.plot(versicolor.Id,versicolor.SepalWidthCm,color="red", label= "versicolor")
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.show()
plt.plot(virginica.Id,virginica.PetalWidthCm, color="blue", label= "virginica",)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.show()
# Scatter Plot
dataframe.plot(kind= "scatter", color= "blue", alpha=0.5, x = "PetalLengthCm", y = "PetalWidthCm")
plt.xlabel("versicolor")
plt.ylabel("setosa")
plt.title("Setosa and Versicolor")
plt.show()
# Histogram Plot
dataframe['PetalLengthCm'].plot(kind= "hist", bins= 50, range= (0,10),figsize=(10,10))
plt.show()
# dictionary keys and values dictionary
dictionary = {'Turkey' : 'erzincan','ankara' : 'kemah'}
print(dictionary.keys())
print(dictionary.values())
# change dictionary content or clear
# list is change
dictionary['ankara'] = "turkey"
print(dictionary)
dictionary['erzincan'] = "kemah"
print(dictionary)
del dictionary["erzincan"]
print(dictionary)
print("ankara" in dictionary)
dictionary.clear()
print(dictionary)
print(dictionary)
dataframe = pd.read_csv('../input/Iris.csv')
series = dataframe['PetalLengthCm']
print(type(series))
data_frame = dataframe[['PetalLengthCm']]
print(type(data_frame))
# Comparison operator
print(3 > 2)
print(3 != 2)
# Boolean operator
print(True and False)
print(True or False)
dataframe.columns
# Filtring pandas data frame
a = dataframe['SepalLengthCm']>7.6 # are Length who have higher Iris-virginica ('SepalLengthCm')
dataframe[a]
# Filtering pandas with logical_and
dataframe[np.logical_and(dataframe['SepalLengthCm']>7, dataframe['PetalWidthCm']>1.9)] 
# Filtring Pandas
dataframe[(dataframe["SepalLengthCm"]>7.6) & (dataframe['PetalWidthCm']>2)] # Length and Width
# while loop
i = 0
while i != 5 :
    print('i is: ',i)
    i +=1
print(i, 'is equal to 5')
# for loop
lis = [1,2,3,4,5]
for i in lis:
    print('i is: ', i)
print('')

