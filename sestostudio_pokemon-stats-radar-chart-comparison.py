# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pokemon_df = pd.read_csv('../input/Pokemon.csv')
pokemon_df.head()
pokemon_df.info()
type_list = pokemon_df['Type 1'].unique()

type_count = pokemon_df[['Type 1']].groupby('Type 1')['Type 1'].count()



plt.figure(figsize=(16, 6))

ax = sns.barplot(sorted(type_count),type_list,palette="Reds_d")

ax.set_title('Number of type comparison')
type_index = pokemon_df[['Type 1','HP','Attack','Defense','Sp. Atk','Sp. Def']].groupby(['Type 1'])['HP','Attack','Defense','Sp. Atk','Sp. Def'].mean()

type_index.head()
type_index.max().max()
from math import ceil



def roundup(x):

    return int(ceil(x / 10.0)) * 10



def get_criterias(maximum_attribute):

    criteria_attribute = []

    for i in range(3):

        criteria_attribute.append(roundup(maximum_attribute * i * 1.25 /3))

    return criteria_attribute



maximum_attribute = roundup(type_index.max().max())

criteria_attribute = get_criterias(maximum_attribute)

criteria_attribute
from math import pi



def plot_radar_comparison(type_name_1, type_name_2):

    attributes = type_index.columns

    attributes_count = len(attributes)



    # We will get the stats and the 1st and 2nd type

    type_1 = type_index.loc[type_name_1].values.tolist()

    type_1 += type_1[:1]



    # In addition we will concatinate the list with the first number of the list, to draw line connection

    type_2 = type_index.loc[type_name_2].tolist()

    type_2 += type_2[:1]

    

    # Calculate angles for filling purpose

    angles = [n / float(attributes_count) * 2 * pi for n in range(attributes_count)]

    angles += angles [:1]



    angles2 = [n / float(attributes_count) * 2 * pi for n in range(attributes_count)]

    angles2 += angles2 [:1]



    ax = plt.subplot(111, polar=True)

    

    #Add the attribute labels to our axes

    plt.xticks(angles[:-1],attributes)

    plt.yticks([criteria_attribute[0],criteria_attribute[1],criteria_attribute[2]], [str(criteria_attribute[0]),str(criteria_attribute[1]),str(criteria_attribute[2])], color="grey", size=7)

    plt.ylim(0, type_index.max().max()*1.25)

    #Plot the line around the outside of the filled area, using the angles and values calculated before

    ax.plot(angles,type_1,'o-', linewidth=2)

    ax.fill(angles, type_1, alpha=0.25)



    ax.plot(angles2,type_2)

    ax.fill(angles2, type_2, 'red', alpha=0.15)

    

    #Use figure text:

    plt.figtext(0.2,0.9,type_name_1,color="blue")

    plt.figtext(0.3,0.9,'Versus')

    plt.figtext(0.4,0.9,type_name_2,color="red")

    plt.show()

    plt.show()
plot_radar_comparison('Dark','Rock')
for key, value in type_index.idxmax().items():

    if (value != 'Dragon'):

        plot_radar_comparison(value, "Dragon")