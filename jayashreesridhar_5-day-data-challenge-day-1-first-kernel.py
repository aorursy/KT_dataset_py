# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
#Read the input file

data=pd.read_csv("../input/7210_1.csv")
print (data.shape)
#List top 5 records in the file

print ("Top 5 records\n",data.head())
#Describe numerical data

print ("Summary statistics of data\n",data.describe())
data.isnull().sum()
#Data cleaning-Remove columns containing mostly NaN's

data=data.drop('asins',axis=1)

data=data.drop('count',axis=1)

data=data.drop('dimension',axis=1)

data=data.drop('flavors',axis=1)

data=data.drop('isbn',axis=1)

data=data.drop('prices.availability',axis=1)

data=data.drop('prices.count',axis=1)

data=data.drop('prices.flavor',axis=1)

data=data.drop('prices.returnPolicy',axis=1)

data=data.drop('prices.shipping',axis=1)

data=data.drop('prices.source',axis=1)

data=data.drop('prices.warranty',axis=1)

data=data.drop('reviews',axis=1)

data=data.drop('vin',axis=1)

data=data.drop('websiteIDs',axis=1)

data=data.drop('weight',axis=1)

data=data.drop('Unnamed: 48',axis=1)

data=data.drop('Unnamed: 49',axis=1)

data=data.drop('Unnamed: 50',axis=1)

data=data.drop('Unnamed: 51',axis=1)
print (data.shape)
#Find number of unique brands present in the data

print ("Number of unique brands\n",len(data.brand.unique()))
#Average price of each brand sorted in descending order

brand_groupby=data.groupby(['brand'])['prices.amountMax'].mean().reset_index()

brand_groupby_descending=brand_groupby.sort_values('prices.amountMax', ascending=False)

print ("Average maximum price of each brand sorted in descending order\n",brand_groupby_descending)
#Distribution of average maximum price of brands

sns.kdeplot(brand_groupby_descending['prices.amountMax']);
#Price distribution of brand with maximum price

data[data['brand']=='JewelsObsession']['prices.amountMax'].plot(kind='kde')
brand_groupby_std=data.groupby(['brand'])['prices.amountMax'].std().reset_index()

brand_groupby_std_descending=brand_groupby_std.sort_values('prices.amountMax', ascending=False)

print ("Standard deviation of each brand sorted in descending order\n",brand_groupby_std_descending)
#Brand with widest price distribution

data[data['brand']=='Teva']['prices.amountMax'].plot(kind='kde')