# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv(os.path.join(dirname,filename))



set_items = list()

for row in df.fillna('').iterrows():

    item_row = row[1].values

    #exists 'asparagus' and ' asparagus'

    item_row = [item.strip() for item in item_row if item != ""]

    set_items.append(set(item_row))



transactions = list(set_items)



#showing the output

transactions[:3]
from mlxtend.preprocessing import TransactionEncoder



# Instantiate transaction encoder and fit in my list of sets data

encoder = TransactionEncoder().fit(transactions)



# Transform my actual data for a new representation

onehot = encoder.transform(transactions)



# Convert onehot encoded data to DataFrame

onehot = pd.DataFrame(onehot, columns = encoder.columns_)



onehot.iloc[:3]
#Most purchased items

support = onehot.mean()

support.sort_values()



(100*support.sort_values(ascending=False))[:20].plot(kind='bar',grid=True,figsize = (12,5))

plt.title("Probability of item in basket",fontsize = 16)

plt.xlabel('Item')

plt.ylabel('Probability')
toy_items = [{'coal', 'sirloin', 'beer','pork'},

             {'pork','beer', 'grounded beef'},

             {'chicken', 'pork','grounded beef'},

             {'sirloin', 'coal','pork'},

             {'pork','coal', 'sirloin', 'coke'},

             {'beer', 'coal','pork'}]



#Pre-processing steps

enc = TransactionEncoder().fit(toy_items)

onehot_toy = enc.transform(toy_items)



# Convert onehot encoded data to DataFrame

onehot_toy = pd.DataFrame(onehot_toy, columns = enc.columns_)



onehot_toy.astype(int)
print('grounded beef support:', onehot_toy['grounded beef'].mean())

print('grounded beef & beer support:', np.logical_and(onehot_toy['grounded beef'],onehot_toy['beer']).mean())
from itertools import permutations



def confidence(itemA,itemB,df):

    return float(np.logical_and(df[itemA],df[itemB]).mean() /(df[itemA].mean()))



def lift(itemA,itemB,df):

    return float(np.logical_and(df[itemA],df[itemB]).mean() /(df[itemA].mean() * df[itemB].mean()))



def leverage(itemA,itemB,df):

    return np.logical_and(df[itemA],df[itemB]).mean() - (df[itemA].mean()*df[itemB].mean())





item_pairs = list()

for itemA,itemB in permutations(onehot,2):

    item_pairs.append(list((itemA,itemB, #names

                            onehot[itemA].sum(),onehot[itemB].sum(), #individual count

                            np.logical_and(onehot[itemA],onehot[itemB]).sum(), #pair count

                            confidence(itemA,itemB,onehot), #confidence

                            lift(itemA,itemB,onehot), #lift

                            leverage(itemA,itemB,onehot), # leverage

                            ))) # 

    

item_pairs = pd.DataFrame(item_pairs,columns = ['itemA','itemB',

                                                'countItemA','countItemB',

                                                'countItemA&B',

                                                'Confidence',

                                                'Lift',

                                                'Leverage'])



item_pairs.sample(5)
THRESHOLD = 50

item_pairs[item_pairs['countItemA&B'] >= THRESHOLD].sort_values(by = 'Lift',ascending=False)[:10]