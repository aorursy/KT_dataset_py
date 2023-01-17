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
import numpy as np # linear algebra

import pandas as pd # for pre - processing

import matplotlib.pyplot as plt # for Data - Visualization

from scipy.special import comb # The number of combinations of N things taken K at a time; "N choose K"

from itertools import combinations, permutations # to form a "iterator algebra"
#! pip install apyori # Apriori algorith to find the Associations of the Grocery Data
print(os.listdir('../input'))
df = pd.read_csv('/kaggle/input/groceries/groceries - groceries.csv', delimiter=',')

df.head(20)
def apyori(df, minimum_support=0.1, confidence=0.22):

    df_values = df.values.astype(str)

    index, counts = np.unique(df_values,return_counts=True)

    df_item = pd.DataFrame(zip(index, counts), columns = ['product', 'frequency'])

    df_item.drop(df_item[(df_item['product'] == 'nan' )|(df_item['product'] == 'None' )].index, inplace=True)

    df_item.sort_values(by='frequency', ascending=False, inplace=True)

    df_item.reset_index(drop=True, inplace=True)

    df_item_frequent = df_item[df_item['frequency']>= minimum_support*len(df)]

    df_itemset_frequency = pd.DataFrame(columns=['itemset', 'frequency'])

    for i in range(1, len(df_item_frequent)+1):

        comb = list(combinations(df_item_frequent['product'].values, i) )

        for w in comb:

            count = 0 

            for instance in df_values:

                if all(elem in instance  for elem in w):

                    count = count +1

            if count >= (minimum_support*len(df)/2):#tirar /2

                df_itemset_frequency = df_itemset_frequency.append({'itemset':w, 'frequency':count}, ignore_index=True)

    df_itemset_frequency.sort_values(by='frequency', inplace=True, ascending=False)

    reliability = pd.DataFrame(columns=['rule', 'frequency', 'reliability'])

    for w in df_itemset_frequency['itemset'].values:

        w_p = list(permutations(w,len(w)))

        for j in w_p:

            #print (len(j[0]))



            p_uni = []

            for i in range(len(j)):



                count = 0 

                for instance in df_values:

                    if all(elem in instance  for elem in j[i:]):

                        count = count +1

                p_uni.append(count/len(df))



            if len(j) != 1:

                a = p_uni[-2]/p_uni[-1]



                for i in range(len(p_uni)-2):

                    a = p_uni[-i-3]/a

                j = list(j)

                j.reverse()

                reliability = reliability.append({'rule':j, 'frequency':p_uni[0], 'reliability':a}, ignore_index=True)

            else:

                reliability = reliability.append({'rule':j, 'frequency':p_uni[0], 'reliability':p_uni[0]}, ignore_index=True)

    reliability.sort_values(by='frequency', ascending=False)

    return reliability[reliability['reliability']>=confidence]

apyori(df.drop(columns='Item(s)'))