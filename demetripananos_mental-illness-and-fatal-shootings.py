# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set_style('white')

colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]

sns.set_palette(sns.xkcd_palette(colors))

import matplotlib.pyplot as plt

from scipy.stats import mode





%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/database.csv')

df.head()
#Let's see how many mentally ill related shootings exist in the data.



mi= df.groupby(['state','signs_of_mental_illness'], as_index = False).id.count()



mi.sort_values(by = ['signs_of_mental_illness','id'], ascending=[False,False], inplace = True)



fig, ax = plt.subplots(figsize = (6,15))

sns.stripplot(ax = ax,data = mi, x = 'id',y = 'state', hue = 'signs_of_mental_illness',size = 10,clip_on = False)



ax.yaxis.grid(True)

ax.set_xlim(0,mi.id.max())



ax.set_xlabel('Shootings')

ax.set_ylabel('State')



# Shrink current axis by 20%

box = ax.get_position()

ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])



# Put a legend to the right of the current axis

ax.legend(loc='center left', bbox_to_anchor=(1, 1))
mi2 = mi.pivot('state','signs_of_mental_illness','id').apply(lambda x: x/x.sum(), axis = 1).sort_values(by = True)







fig, ax = plt.subplots(figsize = (6,15))

mi2.plot(kind = 'barh', stacked = True,ax = ax)



ax.set_xlabel('% of Shootings')

ax.set_ylabel('State')



# Shrink current axis by 20%

box = ax.get_position()

ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])



# Put a legend to the right of the current axis

ax.legend(loc='center left', bbox_to_anchor=(1, 1))
arms = df.groupby(['state','signs_of_mental_illness','armed']).id.count()



arms.loc['WA',:,:].reset_index(level = 0, drop = True).reset_index().pivot('armed','signs_of_mental_illness','id').fillna(0).sort_values(True,ascending = False)
mi_gun = arms[:,:,'gun'].reset_index().pivot('state','signs_of_mental_illness','id').sort_values(True,ascending= False)
fig,ax = plt.subplots(figsize = (6,15))

mi_gun.apply(lambda x: x/x.sum(), axis = 1).sort_values(True,ascending = True).plot(kind = 'barh', stacked = True, ax = ax)





# Shrink current axis by 20%

box = ax.get_position()

ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])



# Put a legend to the right of the current axis

ax.legend(loc='center left', bbox_to_anchor=(1, 0.95), title='Mentall Ill')



ax.set_xlabel('% of Shootings')

ax.set_ylabel('State')

ax.set_title('Gun Weilding Related Shootings and Signs of Mental Illness')

ß