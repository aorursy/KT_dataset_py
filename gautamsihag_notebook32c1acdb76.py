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
temp = pd.read_csv("../input/deliveries.csv")
temp.columns[[1]]
t = temp.drop(temp.columns[[3, 9, 15, 17, 18, 19, 20]], axis = 1)
t.info()
#t.ix[:,8:14].head(2)

#collist = list(t)

#collist.remove('match_id', 'inning', 'batting_team', 'over', 'ball', 'batsman', 'non_striker', 'bowler')



t['extra_sum'] = t[['wide_runs','bye_runs','legbye_runs','noball_runs','penalty_runs','extra_runs']].sum(axis=1)

 
t
df1 = t.groupby(['bowler', 'batting_team', 'over' ,'match_id', 'inning'])
df2 = t.groupby(['bowler', 'non_striker']) #, 'over' ,'match_id', 'inning'])
df2[['noball_runs']].count()
df1['bowler'].count()