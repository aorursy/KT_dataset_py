# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sbn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import matplotlib



matplotlib.rcParams['figure.subplot.wspace'] = 0.4

matplotlib.rcParams['figure.subplot.hspace'] = 0.4

matplotlib.rcParams['font.size'] = 8.0

#matplotlib.rcParams['figure.autolayout'] = True

matplotlib.rcParams['xtick.minor.size']= 1

matplotlib.rcParams['xtick.major.size']= 1

matplotlib.rcParams['xtick.minor.pad']=0

matplotlib.rcParams['xtick.major.pad']=0



% matplotlib inline
data_df = pd.read_csv("../input/Pokemon.csv")
# Removing Total-just the sum of other columns

data_df.drop(['Total'], axis=1, inplace=True)

data_df.head()
data_df.describe()
data_df['Type 2'].isnull().sum()
data_df.dtypes
data_df['Legendary'] = data_df['Legendary'].astype('int64')
var_int = data_df.dtypes[data_df.dtypes=='int64'].index

var_int = var_int[1:]

var_int
data_df['Type 1'].value_counts()
data_df['Type 2'].value_counts()
data_df['Type 2'].fillna('NoType', inplace=True)

data_df['Type 2'].value_counts()
# Seaborn Clustermap denoting the heat of different types

pok_pivot = data_df.pivot_table(values=["Attack","Defense","Speed"],index="Type 1",columns="Generation",aggfunc='mean',fill_value=0)

#print(pok_pivot)

sbn.clustermap(pok_pivot, figsize=(13,16))
l_int = len(var_int)

fig = plt.figure(figsize=(13,8))

#fig.add_subplot()

for i, val in enumerate(var_int):

    fig.add_subplot(3,3,i+1)

    plt.hist(data_df[val], bins=50)

    plt.title(val)



plt.show()
data_df.corr()
# Answering the simple questions

def_max = data_df['Defense'].max()

best_def_pok = data_df[data_df['Defense'] == def_max]



atck_max = data_df['Attack'].max()

best_atck_pok = data_df[data_df['Attack'] == atck_max]



spd_max = data_df['Speed'].max()

fastest_pok = data_df[data_df['Speed'] == spd_max]



hp_max = data_df['HP'].max()

long_stnd_pok = data_df[data_df['HP'] == hp_max]



print(" Best defending pokemon : \n{} \n\n\n Best attacking pokemon : \n{} \n\n\n Fastest pokemon : \n{} \n\n\n Long standing pokemon : \n{}".

     format(best_def_pok,best_atck_pok,fastest_pok,long_stnd_pok))
type_pok = data_df['Type 1'].unique()

type_pok
#for val in type_pok:

#    print("For {} type Pokemon : \n".format(val))

#    # Answering the simple questions

#    def1_max = data_df[data_df['Type 1']=='Grass']['Defense'].max()

#    best1_def_pok = data_df[data_df['Type 1']=='Grass'][data_df['Defense'] == def1_max]



#    atck1_max = data_df[data_df['Type 1']=='Grass']['Attack'].max()

#    best1_atck_pok = data_df[data_df['Type 1']=='Grass'][data_df['Attack'] == atck1_max]



#    spd1_max = data_df[data_df['Type 1']=='Grass']['Speed'].max()

#    fast1_pok = data_df[data_df['Type 1']=='Grass'][data_df['Speed'] == spd1_max]



#    hp1_max = data_df[data_df['Type 1']=='Grass']['HP'].max()

#    lng1_stnd_pok = data_df[data_df['Type 1']=='Grass'][data_df['HP'] == hp1_max]



#    print(" Best defending pokemon : \n{} \n\n\n Best attacking pokemon : \n{} \n\n\n Fastest pokemon : \n{} \n\n\n Long standing pokemon : \n{}".

#     format(best1_def_pok,best1_atck_pok,fast1_pok,lng1_stnd_pok))

    
var_grp = var_int[:-1]

#l_int = len(var_int)

fig = plt.figure(figsize=(13,15))

for j,typ in enumerate(type_pok):

    #fig.add_subplot()

    for i, val in enumerate(var_grp):

        fig.add_subplot(18,7,(j*7)+i+1)

        tmp_df = data_df[data_df['Type 1']==typ]

        plt.hist(tmp_df[val], bins=10)

        if (((j*7)+i) % 7 == 0):

            plt.ylabel(typ)

        if (j == 0):

            plt.title(val)

        



plt.show()
var_int[:-1]
typ = data_df['Type 1'].unique()
fig = plt.figure(figsize=(13,15))

ax1 = fig.add_subplot(311)

for i in var_int[:6]:

    avgs = []

    avgs.append(data_df.groupby(['Type 1'], as_index=None)[i].max())

    avgs_df = pd.DataFrame({'Type 1':avgs[0]['Type 1'],i:avgs[0][i]})

    #print(avgs[0]['Type 1'],avgs[0][i])

    #print(avgs_df)

    avgs_df.plot(x='Type 1',y=i, ax=ax1)

    #plt.plot(x=avgs[0]['Type 1'], y=avgs[0][i])

plt.xticks(range(18),typ)

plt.title("Maximum values of features across different types of Pokemon")

ax2 = fig.add_subplot(312)

for i in var_int[:6]:

    avgs = []

    avgs.append(data_df.groupby(['Type 1'], as_index=None)[i].mean())

    avgs_df = pd.DataFrame({'Type 1':avgs[0]['Type 1'],i:avgs[0][i]})

    #print(avgs[0]['Type 1'],avgs[0][i])

    #print(avgs_df)

    avgs_df.plot(x='Type 1',y=i, ax=ax2)

plt.xticks(range(18),typ)

plt.title("Mean values of features across different types of Pokemon")

ax3 = fig.add_subplot(313)

for i in var_int[:6]:

    avgs = []

    avgs.append(data_df.groupby(['Type 1'], as_index=None)[i].min())

    avgs_df = pd.DataFrame({'Type 1':avgs[0]['Type 1'],i:avgs[0][i]})

    #print(avgs[0]['Type 1'],avgs[0][i])

    #print(avgs_df)

    avgs_df.plot(x='Type 1',y=i, ax=ax3, legend=False)

plt.xticks(range(18),typ)

plt.title("Minimum values of features across different types of Pokemon")

plt.show()
# Visualizing them using box plot

fig = plt.figure(figsize=(13,24))

for i,col in enumerate(var_int[:6]):

    ax1 = fig.add_subplot(6,1,i+1)

    sbn.boxplot(x=data_df['Type 1'], y=data_df[col], ax=ax1)       



plt.show()
# Visualizing them using box plot

#fig = plt.figure(figsize=(13,12))

#ax1 = fig.add_subplot(211)

#sbn.boxplot(x=data_df['Type 1'], y=data_df['Attack'], ax=ax1)       

#ax2 = fig.add_subplot(212)

#sbn.boxplot(x=data_df['Type 1'], y=data_df['Sp. Atk'], ax=ax2)

#plt.show()
#redata_df['HP'] = 0

#redata_df.loc[data_df[data_df['HP']>=100].index,'HP'] = data_df[data_df['HP']>=100]['HP']

#redata_df.tail(20)
lim = 80

pokeH_df = data_df[(data_df['HP']>=lim)&((data_df['Attack']>=lim)&(data_df['Sp. Atk']>=data_df['Attack']))&

                          ((data_df['Defense']>=lim)|(data_df['Sp. Def']>=data_df['Defense']))]
pokeH_df.shape
pokeH_df['Type 1'].value_counts()
# Visualizing them using box plot

fig = plt.figure(figsize=(13,24))

for i,col in enumerate(var_int[:6]):

    ax1 = fig.add_subplot(6,1,i+1)

    sbn.boxplot(x=pokeH_df['Type 1'], y=pokeH_df[col], ax=ax1)       



plt.show()