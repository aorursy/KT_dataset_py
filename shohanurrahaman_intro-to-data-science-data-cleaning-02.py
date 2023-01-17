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

display(df.head(5))
import matplotlib.pyplot as plt



df.plot(kind='box', subplots=True, sharex=False, sharey=False, figsize=(10,10), layout=(3,4))

plt.show()
print('original shape of dataset :',df.shape)



cols = ['SALES', 'MSRP','QUANTITYORDERED']

new_df = df[cols]



#calculation 

Q1 = new_df.quantile(0.25)

Q3 = new_df.quantile(0.75)

IQR = Q3-Q1

maximum = Q3+1.5*IQR

minimum = Q1-1.5*IQR

#print(minimum)



#filter outlier 

cond = (new_df <= maximum) & (new_df >= minimum)

'''

we specify that the condition should be true for all three columns by using the all function with axis=1 argument.

This gives us a list of True/False against each row. 

If a row has all three True values, then it gives a True value to that row

'''

cond = cond.all(axis=1)

df = df[cond]

print('filtered dataset shape : ',df.shape)



#plot again to check that if has any outlier

df.plot(kind='box', subplots=True, sharex=False, sharey=False, figsize=(10,10), layout=(3,4))

plt.show()
new_df = df = df[['SALES','QUANTITYORDERED','MSRP']]

pd.plotting.scatter_matrix(new_df, figsize = (10,10))
print('shape of original data :',df.shape)



mean = df['QUANTITYORDERED'].mean()

std_dev = df['QUANTITYORDERED'].std()



# find z scores

z_scores = (df['QUANTITYORDERED'] - mean) / std_dev

z_scores = np.abs(z_scores)



#print(z_scores.min())



#filter data

df = df[z_scores<3]

print('shape of filtered data : ',df.shape)



#plot data

df['QUANTITYORDERED'].plot(kind='box')

plt.show()