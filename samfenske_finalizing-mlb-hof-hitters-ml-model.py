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
#python won't show a long unnecessary error message that comes up a lot

pd.options.mode.chained_assignment = None
hitters=pd.read_csv('/kaggle/input/hitters/hitters_filtered').drop(columns='Unnamed: 0')

hitters.head(10)
names=pd.read_csv('/kaggle/input/the-history-of-baseball/player.csv')

names['name']=names['name_first']+' '+names['name_last']

names=names[['player_id','name']]

names=names[names['player_id'].isin(hitters['player_id'].tolist())]

hitters=hitters.join(names.set_index('player_id'),on='player_id')

hitters.head(10)
hitters2=hitters[-hitters['percent'].isnull()].reset_index().drop(columns='index')
y=hitters2['percent']

features=['g','ab','r','h','rbi','bb','hr']

X=hitters2[features]
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1, test_size=0.4)

basic_model = DecisionTreeRegressor(random_state=1)

basic_model.fit(train_X, train_y)

predictions=basic_model.predict(val_X)
df=pd.DataFrame(val_X)

df=df.join(hitters2[['player_id','name','inducted','percent','threshold','year']])

df['prediction']=predictions



df['guess']=''

for index in df.reset_index()['index']:

    if df['prediction'][index]>=df['threshold'][index]:

        df['guess'][index]='Y'

    else:

        df['guess'][index]='N'

df['correct?']=df['guess']==df['inducted']



#change the order of the columns

df=df[['name','player_id','g','ab','r','h','hr','rbi','bb','percent','threshold','year','inducted','prediction','guess',

      'correct?']]
df['correct?'].value_counts()
pd.set_option('display.max_rows', None)

hof=df[df['inducted']=='Y']

hof
hof['correct?'].value_counts()
import matplotlib.pyplot as plt
s30=range(1930,1940)

s40=range(1940,1950)

s50=range(1950,1960)

s60=range(1960,1970)

s70=range(1970,1980)

s80=range(1980,1990)

s90=range(1990,2000)

s2000 = range(2000,2016)



decades=[s30,s40,s50,s60,s70,s80,s90,s2000]



fig, axes = plt.subplots(nrows=4, ncols=2,figsize=(40, 20))

fig.subplots_adjust(hspace=1)

plt.suptitle('MLB HOF Voting results and predictions \n green: incorrect- should be HOF \n blue: incorrect- should not be HOF',fontsize=30)

for decade,ax in zip(decades,axes.flatten()):

    frame=df[df['year'].isin(decade)]

    

    ax.plot(frame['name'],frame['percent'],'o',color='red',label = 'Actual Values')



    ax.plot(frame['name'],frame['prediction'],'X',color='yellow',label = 'Predicted Values')

  

    incorrect=frame[frame['correct?'].isin([False])]

    circle_rad = 10 

    

    overshoot=incorrect[incorrect['prediction']>incorrect['percent']]

    ax.plot(overshoot['name'], overshoot['percent'], 'o',ms=circle_rad * 2, mec='b', mfc='none', mew=2)

    ax.plot(overshoot['name'], overshoot['prediction'], 'o',ms=circle_rad * 2, mec='b', mfc='none', mew=2)

    

    undershoot=incorrect[incorrect['percent']>incorrect['prediction']]

    ax.plot(undershoot['name'], undershoot['percent'], 'o',ms=circle_rad * 2, mec='g', mfc='none', mew=2)

    ax.plot(undershoot['name'], undershoot['prediction'], 'o',ms=circle_rad * 2, mec='g', mfc='none', mew=2)

    

    ax.set_xlabel('Player')

    ax.set_ylabel('Percent of HOF Votes')

    ax.set_title(str(decade[0])+'-'+str(decade[-1]))

    ax.legend(loc = 'upper right')

    ax.set_xticklabels(labels=frame['name'],rotation=90)
incorrect=df[df['correct?'].isin([False])]

incorrect
overshoot=incorrect[incorrect['prediction']>incorrect['percent']]

len(overshoot)
undershoot=incorrect[incorrect['percent']>incorrect['prediction']]

len(undershoot)