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
cols

# Restrict to just offensive plays by play type Run or pass

OffPlay = PlaybyPlay[PlaybyPlay['PlayType'].isin(['Run', 'Pass'])]

# yrdline100 ydstogo Touchdown FirstDown down PlayType GoalToGo





#It's much easier if the amount of yards to go is in one column. The dataset provides it

# in two columns depending on if its first down (ydstogo) or near the goal line (GoalToGo) 

# we will create one 'CompositeToGo' to use. 

def coalesce_togo(x):

    if x['ydstogo'] == 0:

        return x['GoalToGo']

    elif x['GoalToGo'] == 0:

        return x['ydstogo']





OffPlay['compositeTogo'] = OffPlay[['ydstogo', 'GoalToGo']].apply(coalesce_togo, axis=1)

cols = ['yrdline100', 'ydstogo', 'Touchdown' ,'FirstDown', 'down', 'PlayType' ,'GoalToGo', 'compositeTogo']

yrds = {}







OffPlay[cols].loc[(OffPlay['down'] ==1) & (OffPlay['yrdline100'] == 40)].groupby(['Touchdown', 'compositeTogo']).count()
yards = range(100)



PlaybyPlay = pd.read_csv('../input/NFLPlaybyPlay2015.csv', index_col=[0,1])
