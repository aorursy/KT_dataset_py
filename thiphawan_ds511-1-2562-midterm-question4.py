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
import pandas as pd 

import numpy as np

import matplotlib as plt 

import seaborn as sns 
dataset1 = pd.read_csv("../input/agricuture-crops-production-in-india/datafile (1).csv")

dataset2 = pd.read_csv("../input/agricuture-crops-production-in-india/datafile (2).csv")

dataset3 = pd.read_csv("../input/agricuture-crops-production-in-india/datafile (3).csv")

dataset4 = pd.read_csv("../input/agricuture-crops-production-in-india/datafile.csv")

dataset5 = pd.read_csv("../input/agricuture-crops-production-in-india/produce.csv")
dataset1.columns = ['Crop','State','CostofCultivationA2',

                     'CostofCultivationC2','CostofProduction2','Yield']
dataset1.info()
dataset2.info()
dataset3.info()

dataset4.info()

dataset5.info()

print(dataset1.shape)

print(dataset2.shape)

print(dataset3.shape)

print(dataset4.shape)

print(dataset5.shape)
#check missing values

print('dataset 1 :')

print(dataset1.isnull().sum())

print(dataset1.shape)

print('--------------------------')

print('dataset 2 :')

print(dataset2.isnull().sum())

print(dataset2.shape)

print('--------------------------')

print('dataset 3 :')

print(dataset3.isnull().sum())

print(dataset3.shape)

print('--------------------------')

print('dataset 4 :')

print(dataset4.isnull().sum())

print(dataset4.shape)

print('--------------------------')

print('dataset 5 :')

print(dataset5.isnull().sum())

print(dataset5.shape)
print('dataset 1 :')



print(dataset1.nunique())

print('---------------------------------------')

print('dataset 2 :')

print(dataset3.nunique())

print('---------------------------------------')

print('dataset 3 :')

print(dataset3.nunique())



print('---------------------------------------')

print('dataset 4 :')

print(dataset4.nunique())

print('---------------------------------------')

print('dataset 5 :')

print(dataset5.nunique())

ax = sns.boxplot(x="Crop", y="CostofCultivationA2", data=dataset1, palette=sns.color_palette("Paired"))
dataset1.groupby('Crop').sum()

data = {'ARHAR': '1', 'COTTON':'2' ,'GRAM': '3', 'GROUNDNUT':'4','MAIZE': '5', 'MOONG':'6','PADDY': '7', 'RAPESEED AND MUSTARD':'8','SUGARCANE': '9', 'WHEAT':'10' }

dataset1['Crop'] = dataset1['Crop'].map(data)

dataset1.head()
sns.pairplot(dataset1 ,hue = 'Crop',diag_kind = 'kde',kind = "scatter",palette = sns.color_palette("Paired"))             
#check missing values

print('dataset 1 :')

print(dataset1.isnull().any())
print(dataset2.isnull().any())
print(dataset3.isnull().any())
print(dataset3.isnull().sum())

print(dataset3.shape)
print(dataset4.isnull().any())
print(dataset5.isnull().any())
print(dataset5.isnull().sum())

print(dataset5.shape)
sns.pairplot(dataset1 ,hue = 'Crop',diag_kind = 'kde',kind = "scatter",palette = sns.color_palette("RdBu_r", 7))             