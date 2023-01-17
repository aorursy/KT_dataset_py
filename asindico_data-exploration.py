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
colors = pd.read_csv('../input/colors.csv')

colors.head()
len(colors['rgb'].unique())
import matplotlib.pyplot as plt

import matplotlib.patches as patches



fig1 = plt.figure()

sq = 0.2

width = 5+sq*len(colors['rgb'])



fig1, axarr = plt.subplots(3, sharex=True)

axarr[0].set_xlim((0, 11))

axarr[0].set_ylim((0,0.2))

axarr[0].axis('off')

fig1.set_figheight(15)

fig1.set_figwidth(30)

for k in range(1,50):

    axarr[0].add_patch(

        patches.Rectangle(

            (0.1+k*0.2, 0.1),   # (x,y)

            0.2,          # width

            0.2,          # height

            facecolor="#"+colors['rgb'][k],

    )

)

    



axarr[1].set_xlim((0, 11))

axarr[1].set_ylim((0,0.2))

axarr[1].axis('off')

for k in range(50,100):

    axarr[1].add_patch(

        patches.Rectangle(

            (0.1+(k-50)*0.2, 0.1),   # (x,y)

            0.2,          # width

            0.2,          # height

            facecolor="#"+colors['rgb'][k],

    )

)

    

axarr[2].set_xlim((0, 11))

axarr[2].set_ylim((0,0.2))

axarr[2].axis('off')

for k in range(100,124):

    axarr[2].add_patch(

        patches.Rectangle(

            (0.1+(k-100)*0.2, 0.1),   # (x,y)

            0.2,          # width

            0.2,          # height

            facecolor="#"+colors['rgb'][k],

    )

)

#fig1.savefig('rect1.png', dpi=90, bbox_inches='tight')
colors['name'].unique()
inv_parts = pd.read_csv('../input/inventory_parts.csv')

inv_parts.head()
parts = pd.read_csv('../input/parts.csv')

parts.head()
inv_set = pd.read_csv('../input/inventory_sets.csv')

inv_set.head()
sets = pd.read_csv('../input/sets.csv')

sets.head()
import seaborn as sns

fig, axs = plt.subplots()

sd = sets.groupby('year')['year'].count().sort_values(ascending=False)

ybp = sns.barplot(sd[0:50].index,sd[0:50].values,  color="g", ax=axs)

fig.set_size_inches(12,10)

for item in ybp.get_xticklabels():

    item.set_rotation(45)