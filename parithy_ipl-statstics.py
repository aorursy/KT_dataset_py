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
f=pd.read_csv("../input/matches.csv")

f['Winning Team']=np.where((f['toss_winner'] == f['winner']) & (f['toss_decision'] == 'bat' ),f['toss_winner'],np.nan)

f=f.drop(['date','city',

                'dl_applied','win_by_wickets',

                'venue','result',

                'umpire1','player_of_match',

                'umpire2',

                'umpire3'], axis=1)

f.groupby('winner').size()

f.plot(kind='bar')

#print('Team Won Most Matches = ' +  f.idxmax(axis=1))

#print('Number of matches Won = ' , f.max())
#    print(results)

#    htm=htm.groupby(['winner','win_by_runs','win_by_wickets','season']).size()

result = pd.pivot_table(f, index='winner', columns='season', values='win_by_runs', aggfunc=np.size)

sns.heatmap(result, annot=False, fmt="g" ,cbar_kws={"orientation": "horizontal"})

   