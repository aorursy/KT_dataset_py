# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

from scipy.special import comb

from itertools import combinations, permutations



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/groceries/groceries - groceries.csv', delimiter=',')
df.head(4)
def apyori(df, minimum_support=0.1, confidence=0.22):

    df_values = df.values.astype(str)

    index, counts = np.unique(df_values,return_counts=True)

    df_item = pd.DataFrame(zip(index, counts), columns = ['produto', 'frequencia'])

    df_item.drop(df_item[(df_item['produto'] == 'nan' )|(df_item['produto'] == 'None' )].index, inplace=True)

    df_item.sort_values(by='frequencia', ascending=False, inplace=True)

    df_item.reset_index(drop=True, inplace=True)

    df_item_frequent = df_item[df_item['frequencia']>= minimum_support*len(df)]

    df_itemset_frequencia = pd.DataFrame(columns=['itemset', 'frequencia'])

    for i in range(1, len(df_item_frequent)+1):

        comb = list(combinations(df_item_frequent['produto'].values, i) )

        for w in comb:

            count = 0 

            for instancia in df_values:

                if all(elem in instancia  for elem in w):

                    count = count +1

            if count >= (minimum_support*len(df)/2):#tirar /2

                df_itemset_frequencia = df_itemset_frequencia.append({'itemset':w, 'frequencia':count}, ignore_index=True)

    df_itemset_frequencia.sort_values(by='frequencia', inplace=True, ascending=False)

    confiabilidade = pd.DataFrame(columns=['regras', 'frequencia', 'confiabilidade'])

    for w in df_itemset_frequencia['itemset'].values:

        w_p = list(permutations(w,len(w)))

        for j in w_p:

            #print (len(j[0]))



            p_uniao = []

            for i in range(len(j)):



                count = 0 

                for instancia in df_values:

                    if all(elem in instancia  for elem in j[i:]):

                        count = count +1

                p_uniao.append(count/len(df))



            if len(j) != 1:

                a = p_uniao[-2]/p_uniao[-1]



                for i in range(len(p_uniao)-2):

                    a = p_uniao[-i-3]/a

                j = list(j)

                j.reverse()

                confiabilidade = confiabilidade.append({'regras':j, 'frequencia':p_uniao[0], 'confiabilidade':a}, ignore_index=True)

            else:

                confiabilidade = confiabilidade.append({'regras':j, 'frequencia':p_uniao[0], 'confiabilidade':p_uniao[0]}, ignore_index=True)

    confiabilidade.sort_values(by='frequencia', ascending=False)

    return confiabilidade[confiabilidade['confiabilidade']>=confidence]

apyori(df.drop(columns='Item(s)'))
