#Welcome

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/fifa19/data.csv') #read the data
data.head(20) #Let us see what is going on with the data, as a first glance
#data.columns # let us see the columns, to have a better understanding of the dataset
data.info()
#it seem to me that age is an important feature for a footballer, therefore I wanted to see the effect of age on potential and overall rating.
print(data['Age'].mean(),data['Age'].median(),data['Age'].std(),data['Age'].min(),data['Age'].max()) 
data.boxplot(column=['Age', 'Overall', 'Potential'])
plt.show()
conc_data_col = pd.concat([data['Age'],data['Overall'],data['Potential']],axis = 1)
data_new = conc_data_col.copy()
data_new
data_new.isnull().sum()
fig, axes = plt.subplots(nrows = 2, ncols =1)
data_new.plot(kind = 'hist',y= 'Age', bins = 50,normed = True, ax = axes[0])
data_new.plot(kind = 'hist',y= 'Overall', bins = 50,normed = True, ax = axes[1],cumulative = True)
data_new.plot(kind = 'hist',y= 'Potential', bins = 50,normed = True, ax = axes[1],cumulative = True)

plt.show()
data_new.plot(kind = 'scatter',x='Age',y='Overall')
data_new.plot(kind = 'scatter',x='Age',y='Potential')
data_new.plot(kind = 'scatter',x='Overall',y='Potential')

plt.show()
first_filter = data.Age >= 40
second_filter = data.Overall >= 85
third_filter = data.Potential >= 85
my_interest = data_new[first_filter & second_filter& third_filter]
my_interest