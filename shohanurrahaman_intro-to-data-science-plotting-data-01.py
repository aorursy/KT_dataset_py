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
df = pd.read_csv('/kaggle/input/sample-sales-data/sales_data_sample.csv', encoding='unicode_escape')

display(df.head())
import matplotlib.pyplot as plt 



print('Minimum quantity ordered : ', df['QUANTITYORDERED'].min())

print('Maximum quantity ordered : ', df['QUANTITYORDERED'].max())

#print(df['QUANTITYORDERED'].value_counts())



#plot 

df['QUANTITYORDERED'].plot(kind='hist', bins=[0,5,10,15,20,25,30,35,40,45,50,55,60], rwidth = 0.8)

plt.show()
df['MONTH_ID'].plot(kind='density')

plt.show()
df.plot(kind='density', subplots=True, sharex=False, sharey=False, layout=(3,3),figsize=(15,15))

plt.show()