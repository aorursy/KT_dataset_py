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
data = pd.read_csv("/kaggle/input/walmart-product-dataset-usa/walmart_com-ecommerce_product_details.csv")
data.head(10)
import seaborn as sns 

import matplotlib.pyplot as plt

%matplotlib inline
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
data.drop('Item Number',axis=1,inplace=True)
data['List Price'].hist(bins=50,color='darkred',alpha=0.7)
data.head()
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
data.drop(['Postal Code','Package Size'],axis=1,inplace=True)

sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
data['Category'].fillna('Health|Home Health Care|Daily Living Aids',inplace=True)
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
data['Gtin'].fillna('1.902137e+11',inplace=True)

data['Brand'].fillna('Upper Crust',inplace=True)

data['Description'].fillna('Stunning Looking Cat Eye Two Tone Reading Glas',inplace=True)
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
data.head(4)
sns.heatmap(data.corr())
sns.set_style('whitegrid')

sns.countplot(x='Available',data=data,palette='RdBu_r') 