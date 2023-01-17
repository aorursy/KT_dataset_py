# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

data = pd.read_csv("../input/zomato-bangalore-restaurants/zomato.csv")
data.head()
data.columns
data.shape
import seaborn as sns

#checking if people like to order food online

sns.barplot(data.groupby('online_order').count().head()['url'].index,data.groupby('online_order').count().head()['url'])
plt.figure(figsize=(12,5))

#checking which type of restaurant was preferred the most

sns.barplot(data['rest_type'].value_counts().head(8).index,data['rest_type'].value_counts().head(8))
rating_data=data[np.logical_and(data['rate'].notnull(), np.logical_and(data['rate']!='-',data['rate']!='NEW' ))]

#print(rating_data.isnull().sum())

rating_data.index=range(rating_data.shape[0])



rating=[]

for i in range(rating_data.shape[0]):

    rating.append(rating_data['rate'][i][:3])

rating_data['rate']=rating

rating_data.sort_values('rate',ascending=False)[['name','location','rate']].head(100).drop_duplicates()
#checking the least rated restaurant

rating_data.sort_values('rate',ascending=True)[['name','location','rate']].head(50).drop_duplicates()