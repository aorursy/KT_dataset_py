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
data = pd.read_csv('../input/BlackFriday.csv')
data.info()
data.columns
data.head(10)
temp_data = data.drop_duplicates(['City_Category','Age'])

cityBased_sales = temp_data.pivot('City_Category','Age','Purchase')

ax = sns.heatmap(cityBased_sales,annot=False,linewidths=.5)
temp_data.Product_Category_1.plot(kind='line',color='blue',label='Product_Category_1',grid=True,linestyle=':')

temp_data.Occupation.plot(color='red',label='Occupation',linestyle='-.')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot ')

plt.show()
data.plot(kind='scatter',x='Product_Category_1',y='Purchase',color='red',alpha=0.5)

plt.title('Scatter Plot')

plt.show()
data.Purchase.plot(kind='hist',bins=50)

plt.xlabel('Purchase')

plt.title('Histogram Plot - Purchase Distribution')

plt.show()