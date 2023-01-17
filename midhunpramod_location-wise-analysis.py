# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

data = pd.read_csv('../input/zomato.csv')





#to delete all the values with null in them

data.dropna(how = 'any', inplace = True)

#to change the approx_cost into integer format 

data['approx_cost(for two people)'] = data['approx_cost(for two people)'].str.replace(',','')

data['approx_cost(for two people)'] = data['approx_cost(for two people)'].astype(int)

data['approx_cost(for two people)'] = data['approx_cost(for two people)']/2

data['appcost'] = data['approx_cost(for two people)']



data.drop(['url','address','approx_cost(for two people)'],axis = 1,inplace  =True)









n = data.groupby('location')['appcost'].mean()

n.sort_values(axis=0, ascending=False, inplace = True)

a = []

for i in n.index[:5]:

    a.append(i)

p2 = plt.bar(a,n[:5],width = 0.5,color = 'orange')

plt.xticks(a, rotation=20)

plt.title('Average Cost per Person Vs The Location (Highest)')

plt.xlabel('Location')

plt.ylabel('Average cost per person(in Rupees)')

    





    

    

    



n = data.groupby('location')['appcost'].mean()

n.sort_values(axis=0, ascending=True, inplace = True)

a = []

for i in n.index[:5]:

    a.append(i)

p2 = plt.bar(a,n[:5],width = 0.5,color = 'orange')

plt.xticks(a, rotation=20)

plt.title('Average Cost per Person Vs The Location (Lowest)')

plt.xlabel('Location')

plt.ylabel('Average cost per person(in Rupees)')

    