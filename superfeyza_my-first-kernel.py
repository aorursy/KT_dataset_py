# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns #it's for visualization tool


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
house_sales = pd.read_csv("../input/HPI_master.csv")
house_sales.info()
house_sales.corr() #there is numeric features correlation :)
#if you want to visualize correlation :
f, ax = plt.subplots(figsize=(20,20))
sns.heatmap(house_sales.corr(),annot= True , linewidths = 0.2 , fmt = '.2f', ax=ax)
plt.show()
#heatmap is a visualization function in seaborn library 
house_sales.tail(7) #you can see last 7 rows here
house_sales.head() #you can see the first 5 rows it is useful for quickly testing
house_sales.columns 
house_sales.yr.plot(kind='hist',
                   bins=20,
                   figsize=(10,10))
plt.xlabel('sales')
plt.ylabel('years')
plt.show() #we compared house prices from January 1991 to August 2016
#If you want to filtering your datas ,lets continue :)
a = house_sales['yr'] > 2014 #from 2015 to 2016
house_sales[a] 

house_sales[np.logical_and(house_sales['yr']>2010, house_sales['period']>10 )] 
#house_sales[(house_sales['yr']>2010) & (house_sales['period']>10)] has same mean
# we used 2 filters with logical_and
#While statements example
temperature = 40
while temperature > 32 :
    print('phone is so hot')
    temperature = temperature - 1
print('phone temperature safe')
fav_fruits = ["apple", "banana", "cherry"]
for x in fav_fruits:
  print('my fav fruit:',x)
for index,value in house_sales[['yr']][10:15].iterrows():
    print(index," : ",value)
#10. to 15. datas from year feature , we achieve
students_class = {'Feyza':'11A','Ayşe':'11C','Ahmet':'10A','Veli':'10B'}
for key,value in students_class.items():
    print(key," : ",value)

#DATA SCIENCE TOOLBOX
#user defined function:
def fahr_to_celsius(temp):
    """
    (Fahrenheit-32)*5/9 = Celsius
    """
    return ((temp - 32)*5/9 )
fahr_to_celsius(276)
print('freezing point of water:', fahr_to_celsius(32), 'C')
print('boiling point of water:', fahr_to_celsius(212), 'C')

#Scope
f=22 #global scope
def g():
    f=34 #local scope
    return f
print(g()) #local scope
print(f) #global scope
print(house_sales.columns)
#exploratory data analysis
#value_counts()
print(house_sales['period'].value_counts(dropna=False))
#list comprehension
# lets classify homes to their years: old - new
homes = sum(house_sales.yr)/len(house_sales.yr)
house_sales["houses_newness"]= ["old" if i> homes else "new" for i in house_sales.yr ]
print(house_sales.tail()) #check datas 
house_sales.describe()