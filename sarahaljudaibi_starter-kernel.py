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
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
 # specify 'None' if want to read whole file

# Iherb categories.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('/kaggle/input/iherb-groceries-section-dataset/Iherb categories.csv')

df1.dataframeName = 'Iherb categories.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head()
df1.describe()
list_columns=list(df1.iloc[:,[8,9,10,11,12,13]])



fig, ax = plt.subplots(nrows=3, ncols=2,figsize=(15,13)) 

ax = ax.ravel() 

for i, column in enumerate(list_columns): 

    sns.distplot(df1[column].dropna(),color='skyblue',ax=ax[i])



fig.suptitle('DATA DISTRIBUATION FOR FOUR COLUMNS\n ', fontsize=16);





fig, ax = plt.subplots(nrows=1,figsize=(15,6)) 



x=sns.countplot(x="product_quality", hue="product_rate", data=df1,palette="rocket",saturation=0.6);

x.legend_.remove()

mask = np.zeros_like(df1.corr())

mask[np.triu_indices_from(mask)] = True



plt.subplots(figsize = (8,7))

sns.heatmap(df1.corr(), annot=True,mask = mask, cmap ='PRGn',linewidths=0.1,square=True)

plt.title("CORRELATIONS BETWEEN FEATURES\n", y = 1.03,fontsize = 20);





colors = ['yellowgreen', 'lightskyblue','gold']

#explode = (0,0,0,0,0,0,0.05)



# groupbed the data based on the gender and the class of the passenger

gender_class=df1.groupby('product_quality').agg('count')

gender_class

#plot the pie to display the percentage all the passengers [female,male] in all the classes

gender_class.iloc[0:3,[0]].plot.pie(subplots=True, autopct='%1.1f%%',

    shadow=True,legend=True, startangle=9)


