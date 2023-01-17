# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np                # linear algebra

import pandas as pd               # data frames

import seaborn as sns             # visualizations

import matplotlib.pyplot as plt   # visualizations

import scipy.stats                # statistics

from sklearn import preprocessing

import os

print(os.listdir("../input"))
Data=pd.read_csv("../input/Mall_Customers.csv")

Data.head()
Data.describe()
plt.hist(Data['Age'])

plt.xlabel('Age')

plt.show()

plt.hist(Data['Annual Income (k$)'])

plt.xlabel('Annual Income (k$)')

plt.show()

plt.hist(Data['Spending Score (1-100)'])

plt.xlabel('Spending Score (1-100)')

plt.show()

Muestra=Data['Gender'].value_counts()

print('%Female:',Muestra[0]*100/(Muestra[0]+Muestra[1]))

print('%Male:',Muestra[1]*100/(Muestra[0]+Muestra[1]))
plt.figure(figsize=(15,5))

sns.boxplot(Data['Gender'], Data['Spending Score (1-100)'])

plt.title('Gender vs Spending Score')

plt.show()
plt.figure(figsize=(15,5))

sns.boxplot(Data['Gender'], Data['Annual Income (k$)'])

plt.title('Gender vs Annual Income')

plt.show()
plt.scatter(Data['Annual Income (k$)'], Data['Age'],  c='red')

plt.ylabel('Age')

plt.xlabel('Annual Income (k$)')

plt.show()

plt.scatter(Data['Spending Score (1-100)'], Data['Age'],  c='red')

plt.ylabel('Age')

plt.xlabel('Spending Score (1-100)')

plt.show()

plt.scatter(Data['Spending Score (1-100)'], Data['Annual Income (k$)'],  c='red')

plt.ylabel('Annual Income (k$)')

plt.xlabel('Spending Score (1-100)')

plt.show()
Corr1=Data['Spending Score (1-100)'].corr(Data['Annual Income (k$)'])

print('CORRELACION gasto-ingreso :', Corr1*100,'%')

Corr2=Data['Spending Score (1-100)'].corr(Data['Age'])

print('CORRELACION gasto-edad :', Corr2*100,'%')

Corr3=Data['Annual Income (k$)'].corr(Data['Age'])

print('CORRELACION ingreso-edad :', Corr3*100,'%')