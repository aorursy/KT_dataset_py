# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

from pandas import DataFrame

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Pokemon.csv')



## removing unwanted columns from the DataFrame

df.drop(df.columns[[0,1,4,5,6,7,8,9,10,12]], axis=1, inplace=True)



## renaming the column names 'Type 1' and 'Type 2' to Type_1 and Type_2

df = df.rename(columns={'Type 1': 'Type_1', 'Type 2': 'Type_2'})

df.fillna(value='missing', axis=1, inplace=True)

print(df.head(3))

print(df.tail(3))
## we now create the dual dataframe

dual = df[~df['Type_2'].str.contains('missing')]



## Generation wise dataframes

gen1 = dual.loc[dual['Generation'] == 1]

gen2 = dual.loc[dual['Generation'] == 2]

gen3 = dual.loc[dual['Generation'] == 3]

gen4 = dual.loc[dual['Generation'] == 4]

gen5 = dual.loc[dual['Generation'] == 5]

gen6 = dual.loc[dual['Generation'] == 6]

## This function counts the number of rows returned by generation specific dataframes based on Type_1 and Type_2

def countFreqByGen(type1,type2,dataset):

    count = 0;

    count = dataset.loc[(dataset['Type_1'] == type1) & (dataset['Type_2'] == type2)]

    return count.shape[0]

# This function counts the number of rows returned by the dual dataframe based on Type_1 and Type_2

def countFreq(type1, type2):

    count = 0;

    count = dual.loc[(dual['Type_1'] == type1) & (dual['Type_2'] == type2)]

    return count.shape[0]



## Example

print(countFreqByGen('Grass','Poison',gen1))

print(countFreq('Grass','Poison'))
## Creating a types array, which will be used for creating a dataframe that references by type.

types = np.array(df['Type_1'].unique())

print(types)
## Creating a DataFrame that contains type-combos along with their count / generation, total count and generation

stats_dict = {}

k = 0;

for i in range(len(types)-1):

    for j in range(i+1, len(types)):

        type1 = types[i]

        type2 = types[j]

        ## Generation 1 stats

        count = countFreqByGen(type1, type2, gen1)

        cumulativeCount = countFreq(type1, type2)

        listtype = [(type1+" "+type2), count,cumulativeCount, 1]

        stats_dict.update({k:listtype})

        k = k+1

        ## Generation 2 stats

        count = countFreqByGen(type1, type2, gen2)

        cumulativeCount = countFreq(type1, type2)

        listtype = [(type1+" "+type2), count,cumulativeCount, 2]

        stats_dict.update({k:listtype})

        k = k+1

        ## Generation 3 stats

        count = countFreqByGen(type1, type2, gen3)

        cumulativeCount = countFreq(type1, type2)

        listtype = [(type1+" "+type2), count,cumulativeCount, 3]

        stats_dict.update({k:listtype})

        k = k+1

        ## Generation 4 stats

        count = countFreqByGen(type1, type2, gen4)

        cumulativeCount = countFreq(type1, type2)

        listtype = [(type1+" "+type2), count,cumulativeCount, 4]

        stats_dict.update({k:listtype})

        k = k+1

        ## Generation 5 stats

        count = countFreqByGen(type1, type2, gen5)

        cumulativeCount = countFreq(type1, type2)

        listtype = [(type1+" "+type2), count,cumulativeCount, 5]

        stats_dict.update({k:listtype})

        k = k+1

        ## Generation 6 stats

        count = countFreqByGen(type1, type2, gen6)

        cumulativeCount = countFreq(type1, type2)

        listtype = [(type1+" "+type2), count,cumulativeCount, 6]

        stats_dict.update({k:listtype})

        k = k+1



freqdf = pd.DataFrame(stats_dict).transpose()

freqdf = freqdf.rename(columns={0: 'Types',1:'Count',2:'Cumulative_Count',3:'Generation'})

freqdf = freqdf[freqdf['Count']>0]

print(freqdf.head(15))

new_index = []

for i in range(0,len(freqdf)):

    new_index.append(i)

freqdf['index'] = new_index

freqdf.set_index(['index'], inplace=True)

print(freqdf.head(15))
## Finding the most common Dual-Type Pokemon and their breakup across different Generations

most_common_types = freqdf[freqdf.Cumulative_Count == freqdf.Cumulative_Count.max()]

print(most_common_types)
## Finding the least common Dual-Type Pokemon and their breakup across different Generations

least_common_types = freqdf[freqdf.Cumulative_Count == freqdf.Cumulative_Count.min()]

print(least_common_types)
pd.options.mode.chained_assignment = None

## Generation 1 stats

gen1freq = freqdf[freqdf['Generation'] == 1]

gen1freq.drop(gen1freq.columns[[3]], axis=1, inplace=True)

## Most common combos in generation 1

mct_gen1 = gen1freq[gen1freq.Count == gen1freq.Count.max()]

print(mct_gen1)

## Least common combos in generation 1

lct_gen1 = gen1freq[gen1freq.Count == gen1freq.Count.min()]

print(lct_gen1)
## Generation 2 stats

gen2freq = freqdf[freqdf['Generation'] == 2]

gen2freq.drop(gen2freq.columns[[3]], axis=1, inplace=True)

## Most common combos in generation 2

mct_gen2 = gen2freq[gen2freq.Count == gen2freq.Count.max()]

print(mct_gen2)

## Least common combos in generation 2

lct_gen2 = gen2freq[gen2freq.Count == gen2freq.Count.min()]

print(lct_gen2)
## Generation 3 stats

gen3freq = freqdf[freqdf['Generation'] == 3]

gen3freq.drop(gen3freq.columns[[3]], axis=1, inplace=True)

## Most common combos in generation 3

mct_gen3 = gen3freq[gen3freq.Count == gen3freq.Count.max()]

print(mct_gen3)

## Least common combos in generation 3

lct_gen3 = gen3freq[gen3freq.Count == gen3freq.Count.min()]

print(lct_gen3)
## Generation 4 stats

gen4freq = freqdf[freqdf['Generation'] == 4]

gen4freq.drop(gen4freq.columns[[3]], axis=1, inplace=True)

## Most common combos in generation 4

mct_gen4 = gen4freq[gen4freq.Count == gen4freq.Count.max()]

print(mct_gen4)

## Least common combos in generation 4

lct_gen4 = gen4freq[gen4freq.Count == gen4freq.Count.min()]

print(lct_gen4)
## Generation 5 stats

gen5freq = freqdf[freqdf['Generation'] == 5]

gen5freq.drop(gen5freq.columns[[3]], axis=1, inplace=True)

## Most common combos in generation 5

mct_gen5 = gen5freq[gen5freq.Count == gen5freq.Count.max()]

print(mct_gen5)

## Least common combos in generation 5

lct_gen5 = gen5freq[gen5freq.Count == gen5freq.Count.min()]

print(lct_gen5)
## Generation 6 stats

gen6freq = freqdf[freqdf['Generation'] == 6]

gen6freq.drop(gen6freq.columns[[3]], axis=1, inplace=True)

## Most common combos in generation 6

mct_gen6 = gen6freq[gen6freq.Count == gen6freq.Count.max()]

print(mct_gen6)

## Least common combos in generation 6

lct_gen6 = gen6freq[gen6freq.Count == gen6freq.Count.min()]

print(lct_gen6)