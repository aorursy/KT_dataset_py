# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Singapore_data=pd.read_csv("../input/singapore-airbnb/listings.csv")
# visualisation of the top 5 rows



Singapore_data.head()
Singapore_data.info()


# plt.figure(figsize=(20,30))

fig = plt.figure(figsize = (15,20))

ax = fig.gca()

Singapore_data.hist(ax = ax)
#Description of values in each of the numerical columns



Singapore_data.describe()
f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(Singapore_data.corr(),annot=True)#,linewidths=5,fmt='.1f',ax=ax)

#plt.show()
Singapore_data.columns
sns.pairplot(Singapore_data)
sns.regplot(x="minimum_nights", y="price", data=Singapore_data)
sns.barplot(x="minimum_nights", y="price",hue='room_type' ,data=Singapore_data)

fig = plt.figure(figsize = (15,20))

sns.scatterplot(x="minimum_nights", y="price",hue='room_type' , data=Singapore_data)
sns.scatterplot(x="number_of_reviews", y="price",hue='room_type' , data=Singapore_data)
sns.regplot(x="number_of_reviews", y="price", data=Singapore_data)
import numpy as np

def pearson_r(x, y):

    """Compute Pearson correlation coefficient between two arrays."""

    # Compute correlation matrix: corr_mat

    corr_mat = np.corrcoef(x, y)



    # Return entry [0,1]

    return corr_mat[0,1]



# Compute Pearson correlation coefficient for I. versicolor

r = pearson_r(Singapore_data.minimum_nights, Singapore_data.price)

r
r = pearson_r(Singapore_data.number_of_reviews, Singapore_data.availability_365)

r