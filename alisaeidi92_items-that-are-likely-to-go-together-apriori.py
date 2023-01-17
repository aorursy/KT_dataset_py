# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
### This analysis shows which items are likely to be bought together within the same transaction###
#Importing the needed libraries

import seaborn as sns
import matplotlib as plt
from matplotlib.pyplot import figure
#Importing the dataset
df = pd.read_csv('../input/BreadBasket_DMS.csv')
#Dropping the NONE values to avoid getting irrelevant baskets, example: [Coffee, NONE] bought together! so what?!
df = df.mask(df.eq('NONE')).dropna()
#looking at our final dataframe

df.head(10)
#Checking the number of item categories we have

len(df.Item.unique())
#Looking at the most and least sold items

fix, ax = plt.pyplot.subplots(figsize = (20,12))
g = sns.countplot(df['Item'])
for item in g.get_xticklabels():
    item.set_rotation(90)

#Looking at the number of total 'unique' transactions

len(df.Transaction.unique())
#Checking the datafrane total length

len(df)
# Apriori algorithm ONLY accepts list of lists. Therefore we need to convert our dataframe into a list of lists called bag.
# In other words, we have a big list of transactions where each transaction contains a list of items in that specific transaction.
# I chose 9466 because thats the maximum number of transactions. We cannot have a list longer than this.

bag = []
for i in range(1,9466):
        bag.append(df.Item[df.Transaction == i].tolist())
        
bag
#Importing the apriori algorithm

from apyori import apriori
#We need to create the rules that indicate the likelihood of items bought together.
#The 0.0022 support comes from: (3(min number of times an item was purchased in a day) * 7 (converted to a week)) / total transactions
#The higher the confidence, more obvious the rules are. The lower the confidence, less obvious they are. We wanna see the less obvious!
#lift, roughly, greater than 3 means strong likelihood of items being associated. less lift is bad!
#min_length is the min number of associated items we want to see in results. We don't wanna see just Coffee associated to itself!

rules = apriori(bag,min_support = 0.0022,min_confidence = 0.1, min_lift = 3, min_length = 2 )
#Visualizing the results
#We can see the items associated together and their support, confidence and lift values.

results = list(rules)
results

