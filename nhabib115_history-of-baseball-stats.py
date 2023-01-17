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
#Location of Lahman's dataset

prefix = '../input/'



master = pd.read_csv(prefix + 'player.csv')

batting = pd.read_csv(prefix + 'batting.csv')

pitching = pd.read_csv(prefix + 'pitching.csv')
master.loc[master['name_last'] == 'Hamels']
df = pitching[['player_id','year','ipouts','so','w','er','h','bb','hbp','cg','sho']]

df = df.loc[df['year'].isin([2014,2015])]

master.loc[master['name_last'].isin(['Verlander', 'Peacock', 'Hendricks'])]
df.loc[df['player_id'].isin(['hendrky01','peacobr01','verlaju01'])]