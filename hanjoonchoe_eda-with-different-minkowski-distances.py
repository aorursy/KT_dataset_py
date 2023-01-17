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



import seaborn as sns

import matplotlib.pyplot as plt

import seaborn as sns

import math

from scipy.spatial import distance

import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/train.csv')
train.columns
groups = train.groupby(['City','Month','Hour','Weekend','Latitude','Longitude'])[['DistanceToFirstStop_p80','TotalTimeStopped_p80']].mean().reset_index()

groups.head()
#modified slightly

def add_distance(df,p):

    

    df_center = pd.DataFrame({"Atlanta":[33.753746, -84.386330],

                             "Boston":[42.361145, -71.057083],

                             "Chicago":[41.881832, -87.623177],

                             "Philadelphia":[39.952583, -75.165222]})

    

    df[f'Minkowski_{p}'] = df.apply(lambda row: distance.minkowski

                                   (np.column_stack((row.Latitude,row.Longitude)) ,df_center[row.City].values,p), axis=1)

    

    

    
p = np.linspace(1,10,10).astype(int)



for i in p:

    add_distance(groups,i)
groups.head()
types=['DistanceToFirstStop_p80','TotalTimeStopped_p80']



def box_cox(value,nlambda):

    bc_value = ((value**nlambda)-1)/nlambda

    return bc_value



def distplot(df,types=None,degree=None,nlambda=None):

    fig = plt.figure(figsize=(20,3))

    city_list = df.City.unique()

    if nlambda==None:

        nlambda = 1

    for i in range(len(city_list)):

        plt.subplot(1, 4, i+1)

        total_values = df[types]

        values = df.loc[df.City == city_list[i]][types]

        bc_value = box_cox(values,nlambda)

        bc_value_t = box_cox(total_values,nlambda)

        ax=sns.distplot(bc_value,kde=False)

        ax.set_xlim(min(bc_value_t),max(bc_value_t))

        ax.set_ylim(0,4*10**4)

        ax.set_title(city_list[i])

        ax.set_xlabel(types+' (box_cox// '+ 'lambda= ' + str(nlambda)+')')

        if i ==0:



            ax.set_ylabel('count')



    plt.show()
groups.groupby(['City']).count()
distplot(groups,types[0],nlambda = 0.15)

distplot(groups,types[1],nlambda = 0.3)
dist_index = [c for c in groups.columns if c not in train.columns]



def quantile_cut(df,dist,num):

    

    cities = df.City.unique()

    

    for city in cities:

        

        df.loc[df.City==city,dist+'_cat']=pd.qcut(df.loc[df.City==city][dist], q=num,labels=[1,2,3])

        

    return df



for dist in dist_index:

    

    groups = quantile_cut(groups,dist,3)

    
#Check it is splited equally

groups.Minkowski_1_cat.value_counts()
check1 = ['DistanceToFirstStop_p80','TotalTimeStopped_p80']

check2 = check2 = ['Hour','Month','Weekend']

check3 = [c for c in groups.columns if c not in train.columns]

check3 = check3[-10:]

def Plot(datasets,check1,check2,check3):

    

    count = datasets['City'].nunique()

    list = datasets['City'].unique()

    fig,ax =plt.subplots(1,4)

    fig.set_size_inches(15, 3)

    for i in range(count):

        dataset = datasets.loc[datasets['City']==list[i]]

        

        if check2 =='Weekend':

            dataset.groupby([check2,check3])[check1].mean().unstack().plot(ax=ax[i],kind='bar')

        else:

            dataset.groupby([check2,check3])[check1].mean().unstack().plot(ax=ax[i])

        if i==0:

            ax[i].set_ylabel(check1)

        if check2 == 'Month':

            ax[i].set(xlim=(6, 12))

        ax[i].set_title(list[i])

        

    plt.show()



for i in check3:

    

    Plot(groups,check1[0],check2[0],i)
for i in check3:

    Plot(groups,check1[1],check2[1],i)
types1 = ['DistanceToFirstStop_p80','TotalTimeStopped_p80']

types2 = [c for c in groups.columns if c not in train.columns]

types2 = types2[-10:]

def distplot2(df,city,types1,types2):

    print(city, types1)

    j = 0

    fig = plt.figure(figsize=(50,10))

    for type2 in types2:

        dataset = df.loc[df.City==city]

        list = dataset[type2].unique()

        count = dataset[type2].nunique()

        for i in range(count):

            plt.subplot(2, 5, j+1)

            values = box_cox(dataset[types1].loc[dataset[type2]==i+1],0.15)

            ax=sns.distplot(values,kde=False)

        j = j+1

    plt.show()
distplot2(groups,'Philadelphia',types1[1],types2)

distplot2(groups,'chicago',types1[1],types2)

distplot2(groups,'Boston',types1[1],types2)

distplot2(groups,'Atlanta',types1[1],types2)



distplot2(groups,'Philadelphia',types1[0],types2)

distplot2(groups,'chicago',types1[0],types2)

distplot2(groups,'Boston',types1[0],types2)

distplot2(groups,'Atlanta',types1[0],types2)