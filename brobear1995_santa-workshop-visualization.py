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
import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt



sns.set_style('whitegrid')



df = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv')

print('Santa must attend {} families!'.format(df.shape[0]))

df.head()
plt.figure(figsize=(20,12))

sns.countplot(x='n_people', data=df, order=df['n_people'].value_counts().index)
people_prob_density = df['n_people'].value_counts(normalize=True).to_dict()
demand = [0 for _ in range(101)]



def add_demand(x):

    for i in range(0,10):

        y = x.at['choice_'+str(i)]

        demand[y] = demand[y] + ((10-i)*x.at['n_people'])

    return x

df = df.apply(lambda x: add_demand(x), axis=1)

demand = demand[1:]

demand = [float(x)/sum(demand) * 100 for x in demand]

sns.scatterplot(y=demand, x=list(range(1,101)))
import numpy as np

demand_arr = np.array(demand)

np.where(demand_arr > 2)[0]+1
import numpy as np

np.where(np.logical_and(demand_arr<2, demand_arr > 1))[0]+1
import numpy as np

np.where(demand_arr<1)[0]+1
demand_dict = dict(enumerate(demand, 1))

def calculate_eagerness(x):

    eagerness = 0

    for i in range(0,9):

        eagerness = eagerness + (9-i)*demand_dict[x.at['choice_'+str(i)]]

    return round(eagerness,0)

    

df['eagerness'] = df.apply(lambda x: calculate_eagerness(x), axis=1)
sns.distplot(df['eagerness'])
eager_df = df.sort_values(by='eagerness', ascending=False)[['family_id','eagerness','n_people']]

display(eager_df.head())

display(eager_df.tail())
df[df.family_id==4851]
df[df.family_id==4904]