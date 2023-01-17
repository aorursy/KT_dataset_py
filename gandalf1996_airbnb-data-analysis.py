# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline

sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
air_df = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

air_df.head()
air_df.columns
def plotneighbourhood(data,cat):

    l=data.groupby(cat).size()

    l=np.log(l)

    l=l.sort_values()

    fig=plt.figure(figsize=(45,7))

    plt.yticks(fontsize=8)

    l.plot(kind='bar',fontsize=12,color='r')

    plt.xlabel('')

    plt.ylabel('Number of records',fontsize=10)



plotneighbourhood(air_df,'neighbourhood')
neighbourhood_df = air_df[['neighbourhood','price']]

neighbourhood_df.columns
plt.figure(figsize=(12,6))

sns.barplot(y="neighbourhood", x="price", data=neighbourhood_df.head(10))

plt.ioff()
rooms_df = air_df[['room_type','neighbourhood_group']]

rooms_df.columns
plt.figure(figsize=(10,6))

sns.countplot(x = 'room_type',hue = "neighbourhood_group",data = rooms_df)

plt.title("Room types occupied by the neighbourhood_group")

plt.show()