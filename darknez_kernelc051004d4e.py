# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv')

df[df['threat_level'] == 'undetermined']['id'].count()

attacked = df[df['threat_level'] == 'attack']
def make_autopct(values):

    def my_autopct(pct):

        total = sum(values)

        val = int(round(pct*total/100.0))

        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)

    return my_autopct
statecount = attacked.groupby('state')['id'].count()
statecount.plot.pie(y ='values',figsize =(15,15),legend = True, pctdistance=0.8,  autopct=make_autopct(statecount.values), labeldistance=1.2)
df.head()
df.groupby(['manner_of_death','flee'])['flee'].describe()
vals = make_autopct(df.groupby(['signs_of_mental_illness','threat_level'])['id'].count())

df.groupby(['signs_of_mental_illness','threat_level'])['id'].count().plot.pie(legend = True, figsize = (10,10), autopct=vals)
vals = make_autopct(df.groupby(['body_camera','threat_level'])['id'].count())

df.groupby(['body_camera','threat_level'])['threat_level'].count().plot.pie(legend = True, autopct = vals, figsize = (10, 10))
vals = make_autopct(df.groupby(['body_camera','manner_of_death'])['id'].count())

df.groupby(['body_camera','manner_of_death'])['threat_level'].count().plot.pie(legend = True, autopct = vals, figsize = (10, 10))
df['Age-group'] = pd.cut(df['age'],bins = [0,20,40,60,80,100])
df.groupby(['Age-group','manner_of_death'])['manner_of_death'].count().plot.pie(legend = True, figsize = (10,10), 

                                                            autopct = make_autopct(df.groupby(['Age-group','manner_of_death'])['manner_of_death'].count()))
vals = df.groupby(['Age-group','manner_of_death','state'])['id'].count().iloc[0*len(statecount.index):0*len(statecount.index) + 10]

df.groupby(['Age-group','manner_of_death','state'])['id'].count().iloc[0*len(statecount.index):0*len(statecount.index)+10].plot.pie(figsize = (10,10),

                                                                                                                                        autopct = make_autopct(vals))
vals = df.groupby(['Age-group','manner_of_death','state'])['id'].count().iloc[2*len(statecount.index):2*len(statecount.index) + 10]

df.groupby(['Age-group','manner_of_death','state'])['id'].count().iloc[2*len(statecount.index):2*len(statecount.index)+10].plot.pie(figsize = (10,10),

                                                                                                                                        autopct = make_autopct(vals))

vals = df.groupby(['Age-group','manner_of_death','state'])['id'].count().iloc[4*len(statecount.index):4*len(statecount.index) + 10]

df.groupby(['Age-group','manner_of_death','state'])['id'].count().iloc[4*len(statecount.index):4*len(statecount.index)+10].plot.pie(figsize = (10,10),

                                                                                                                                        autopct = make_autopct(vals))
vals = df.groupby(['Age-group','manner_of_death','state'])['id'].count().iloc[3*2*len(statecount.index):3*2*len(statecount.index) + 10]

df.groupby(['Age-group','manner_of_death','state'])['id'].count().iloc[3*2*len(statecount.index):3*2*len(statecount.index)+10].plot.pie(figsize = (10,10),

                                                                                                                                        autopct = make_autopct(vals))
vals = df.groupby(['Age-group','manner_of_death','state'])['id'].count().iloc[8*len(statecount.index):8*len(statecount.index) + 10]

df.groupby(['Age-group','manner_of_death','state'])['id'].count().iloc[8*len(statecount.index):8*len(statecount.index)+10].plot.pie(figsize = (10,10),

                                                                                                                                        autopct = make_autopct(vals))