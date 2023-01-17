# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/StudentsPerformance.csv")
data.info()
data.describe()
data.corr()
data.columns
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()

dfmale = data[data.gender == "male"] # New DataFrame for male
dffemale = data[data.gender == "female"] # New DataFrame for female


# Lineplot
dfmale["math score"].plot(kind = "line",color = "black",label = "Math Score (Male)", linewidth = 1, grid = True, alpha = 0.5, linestyle = ":",figsize = (12,12))
dffemale["math score"].plot(color = "r",label = "Math Score (Female)", linewidth = 1, grid = True, alpha = 0.5, linestyle = "-.")
plt.legend(loc="lower center")
plt.xlabel("number of students")
plt.ylabel("math exam score")
plt.title("Line Plot")
plt.show()
#ScatterPlot
data.plot(kind="scatter", x = "reading score",y = "writing score",alpha = 0.5,color = "orange",grid = True,figsize = (12,12))
plt.xlabel("Reading Score",color = "blue")
plt.ylabel("Writing score",color = "r")
plt.title("Writing vs Reading ")
plt.show()
#Histogram
data["math score"].plot(kind = "hist", bins = 250, grid = True, figsize = (24,12),color = "gray")
plt.ylabel("Frequency")
plt.xlabel("Math Score")
plt.show()
# clf() = cleans it up again you can start a fresh
data["math score"].plot(kind = "hist", bins = 250, grid = True, figsize = (24,12),color = "gray")
plt.clf()
dictionary = {"Turkey": "Ankara","England":"London","Usa":"'Washingon"}
print(dictionary.keys())
print(dictionary.values())


dictionary["Palestine"] = "Jarussalem" # Add new Entry
print(dictionary)
dictionary["Usa"] = "New York" # Update Entry
print(dictionary)
del dictionary["England"] # remove entry
print(dictionary)
print("Turkey" in dictionary) # include or not
dictionary.clear() # remove all
print(dictionary)
# In order to run all code you need to take comment this line
# del dictionary         # delete entire dictionary     
#print(dictionary)       # it gives error because dictionary is deleted

data = pd.read_csv("../input/StudentsPerformance.csv")
dfmale = data[data.gender == "male"] # New DataFrame for male
dffemale = data[data.gender == "female"] # New DataFrame for female
# Series
series = data["reading score"] # it's serie
print(type(series))
data_frame = data[["reading score"]] # it's dataFrame
print(type(data_frame))
# 1 - Filtering Pandas data frame
filter1 = dfmale["math score"] >95
dfmale[filter1] # only 12 female students

filt = data["reading score"]>95
filt2 = data["writing score"]>95
filt3 = data["math score"]>95
data[filt & filt2 & filt3]
malefilt = data.gender=="male"
femalefilt = data.gender =="female"
filt = data["reading score"]>95
filt2 = data["writing score"]>95
filt3 = data["math score"]>95
smartmale = data[filt & filt2 & filt3& malefilt].shape[0]
smartfemale = data[filt & filt2 & filt3& femalefilt].shape[0]

print("Number of hardworking male students: ", smartmale)
print("Number of hardworking female students",smartfemale)



#2 - Filtering pandas with logical_and
data[np.logical_and(data["reading score"]>95, data["writing score"]>95)] 
# Stay in loop if condition( i is not equal 5) is true

i = 0

while i!=5:
    print("i is: ", i)
    i +=1
print("i is equal to 5")    
# Stay in loop if condition( i is not equal 5) is true
liste = [1,2,3,4,5]

for i in liste:
    print(i)

print()

# Enumerate index and value of list
# index : value = 0:1, 1:2, 2:3, 3:4, 4:5

for index, value in enumerate(liste):
    print(index,":",value)
# For dictionaries
# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.
dict2 = {"Green":["Blue","Yellow"],"Orange":["Red","Yellow"]}
for key, value in dict2.items():
    print(key,":",value[0],value[1])
    
print()
# For pandas we can achieve index and value
for index,value in data[["math score"]][0:2].iterrows():
    print(index," : ",value)
    