# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
ps=pd.read_csv('../input/googleplaystore.csv')

ps.head(2)
dup=ps.duplicated(subset='App').sum()

print('it has {} duplicates of {} original data'.format(dup,len(ps)))



ps=ps.drop(ps.index[ps.duplicated(subset='App')])

print('now data set size is {} '.format(len(ps)))




ps=ps[np.isfinite(ps['Rating'])]

print('removed NaN entries, dataset length is  {}'.format(len(ps)))
def freeapp(): #free app entries listed

    apps=ps.loc[ps.Type =="Free"]['App']

    print('Free apps available are {}'.format(len(apps)))

    return apps

freeapp()