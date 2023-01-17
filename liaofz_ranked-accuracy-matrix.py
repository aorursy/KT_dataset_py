# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output



# Any results you write to the current directory are saved as output.
m1 = pd.read_csv('../input/accuracy_matrix.csv')

defense = pd.read_csv('../input/defense_results.csv')

nont = pd.read_csv('../input/non_targeted_attack_results.csv')
defense_rank = list(np.array(defense.KaggleTeamId))

nont_rank = list(np.array(nont.KaggleTeamId))
data = np.array(m1)[:,1:]

defenseid = list(np.array(m1)[:,0])

attackid = list(m1.columns[1:])

sort_defense = np.array([defenseid.index(d) for d in defense_rank if d in defenseid])

sort_nont = np.array([attackid.index(d) for d in nont_rank if d in attackid])

defense_name =  np.array([n for d,n in zip(defense_rank,defense.TeamName) if d in defenseid])

attack_name =  np.array([n for d,n in zip(nont_rank,nont.TeamName) if d in attackid])
data_sort = []

for id in sort_defense:

    data_sort.append(data[id])

data_sort = np.array(data_sort)



data_sort2 = []

for id in sort_nont:

    data_sort2.append(data_sort[:,id])

data_sort2 = np.array(data_sort2).T
df = pd.DataFrame(data_sort2)

df.columns = attack_name

df.insert(0,'TeamName',defense_name)

df
df = pd.DataFrame(data_sort2.T)

df.columns = defense_name

df.insert(0,'TeamName',attack_name)

df