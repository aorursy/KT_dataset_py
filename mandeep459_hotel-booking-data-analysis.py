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
file='/kaggle/input/hotel-booking-demand/hotel_bookings.csv'
bd=pd.read_csv(file)
bd.info()
bd.isnull().sum()
bd.fillna(method ='bfill')
import matplotlib.pyplot as plt 

import seaborn as sns
fig,axes = plt.subplots(1,1,figsize=(7,7))



sns.heatmap(bd.corr(), cmap='coolwarm', linecolor='white')



plt.show()
bd['hotel'].value_counts()
sns.countplot(x='hotel',data=bd)
ct = pd.crosstab(bd.hotel,bd.is_canceled)



ct.plot.bar(stacked=True)

plt.legend(title='mark')



plt.show()
sns.countplot(x='arrival_date_year',data=bd)
g = sns.PairGrid(bd, vars=['arrival_date_month'],

                 hue='customer_type', palette='RdBu_r')

g.map(plt.scatter, alpha=0.8)

g.add_legend()
sns.boxplot(bd['adr'])
fig = plt.figure(figsize=(10,5))



x = bd['country'].value_counts().index[:10]

y = bd['country'].value_counts()[:10]



plt.bar(x,y, color='red')

plt.xlabel('Countries')

plt.ylabel('Customer count')

plt.title('Top 15 customer count')



plt.show()
ax = sns.scatterplot(x="adr", y="arrival_date_month", hue="hotel",

                     data=bd)
bd['required_car_parking_spaces'].value_counts()
ct=pd.crosstab(bd.required_car_parking_spaces,bd.is_canceled)

ct.plot.bar(stacked=True)

plt.legend(title='mark')



plt.show()