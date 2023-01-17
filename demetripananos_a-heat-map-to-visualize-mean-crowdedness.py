# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/data.csv')
#analyze time in hours instead of seconds

df['Hour'] = df.timestamp.apply( lambda x: int(np.floor(x/3600))) 



g = df[['Hour','number_people','day_of_week']]



#Group by tme and day

F = g.groupby(['Hour','day_of_week'], as_index = False).number_people.mean().pivot('day_of_week','Hour', 'number_people').fillna(0)





grid_kws = {"height_ratios": (.9, .05), "hspace": .3}



dow= 'Monday Tuesday Wednesday Thursday Friday Saturday Sunday'.split()

dow.reverse()



ax = sns.heatmap(F, cmap='RdBu_r',cbar_kws={"orientation": "horizontal"})

ax.set_yticklabels(dow, rotation = 0)

ax.set_ylabel('')

ax.set_xlabel('Hour')



cbar = ax.collections[0].colorbar

cbar.set_label('Average Number of People')
lwise = np.gradient(F, edge_order = 2)[1]

Fp = pd.DataFrame(lwise, columns=F.columns, index = F.index)





ax = sns.heatmap(Fp, cmap='RdBu_r',cbar_kws={"orientation": "horizontal"})

ax.set_yticklabels(dow, rotation = 0)

ax.set_ylabel('')

ax.set_xlabel('Hour')



cbar = ax.collections[0].colorbar

cbar.set_label('Rate of Change')