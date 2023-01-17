# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib 

import matplotlib.pyplot as plt



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

#--------------------------------------------------



#Unzip dataset

import zipfile

Dataset = "bob-ross"

with zipfile.ZipFile("../input/data/"+Dataset+".zip","r") as z:

    z.extractall(".")



#Read Data

df = pd.read_csv("bob-ross/elements-by-episode.csv")



#Make a single columns for trees

#  This is done because there is no way to differentiate between similarly named elements

df['TREES_ALL'] = df[['TREE','TREES','DECIDUOUS','CONIFER','PALM_TREES']].max(axis=1)

df = df.drop(['TREE','TREES','DECIDUOUS','CONIFER','PALM_TREES'],axis=1)



#Make a single columns for Mountains

df['MOUNTAINS_ALL'] = df[['MOUNTAIN','MOUNTAINS','SNOWY_MOUNTAIN']].max(axis=1)

df = df.drop(['MOUNTAIN','MOUNTAINS','SNOWY_MOUNTAIN'],axis=1)



#Make a single columns for Clouds

df['CLOUDS_ALL'] = df[['CLOUDS','CUMULUS']].max(axis=1)

df = df.drop(['CLOUDS','CUMULUS'],axis=1)



#Prepare data for plot

item_nb = 15

df_sum = df.sum()

df_sum = df_sum.drop(['EPISODE','TITLE'])

df_sum = df_sum.sort_values(ascending=False)[:item_nb]





#Plot Line Chart including most popular items

matplotlib.style.use('bmh')

fig, ax = plt.subplots(figsize=(12,8))

plt.subplots_adjust(right=0.8)



for idx, value in df_sum.iteritems():

    line_val = df[idx].cumsum()

    plt.plot(line_val, label=idx)



#Plot Visuals

plt.title('Bob Ross Paintings\' Elements over All Episodes')

plt.ylabel('Element Cumulative Count')

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax = plt.gca()

ax.set_xlim(0,403)

plt.tick_params(

    axis='x',          

    which='both',     

    bottom='off',     

    top='off',        

    labelbottom='off') 

plt.show()