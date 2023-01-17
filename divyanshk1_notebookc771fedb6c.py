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
data = pd.read_csv('../input/deliveries.csv')

data.columns
cat = ['caught', 'bowled', 'lbw', 'stumped', 'caught and bowled']

bowler_wicket = data['player_dismissed'][(data.player_dismissed==data.batsman) &

                                         ((data.dismissal_kind == cat[0]) |

                                         (data.dismissal_kind == cat[1]) |

                                         (data.dismissal_kind == cat[2]) |

                                         (data.dismissal_kind == cat[3]) |

                                         (data.dismissal_kind == cat[4]))]

data1 = data.merge(pd.DataFrame({'wkt':bowler_wicket}),how='left',left_index=True,right_index=True)

data1 = data1.merge(pd.DataFrame({'fr':data.batsman_runs==4}),how='left',left_index=True,right_index=True)

data1 = data1.merge(pd.DataFrame({'sx':data.batsman_runs==6}),how='left',left_index=True,right_index=True)

data1 = data1.merge(pd.DataFrame({'bowl':(data.extra_runs==0)|(data.bye_runs!=0)|(data.legbye_runs!=0)}),how='left',left_index=True,right_index=True)
col = ["batsman","batsman_runs"]

bats = data[col]

fours = bats[bats.batsman_runs ==4]

fours.columns = ["batsman","fours"]

fours = fours.groupby('batsman').agg({'fours':'count'}).reset_index()

sixes = bats[bats.batsman_runs ==6]

sixes.columns = ["batsman","sixes"]

sixes = sixes.groupby('batsman').agg({'sixes':'count'}).reset_index()
bowl = data.where((data.extra_runs==0)|(data.bye_runs!=0)|(data.legbye_runs!=0))

h = bowl.batsman.value_counts()

co = {'bowls': h}

c = pd.DataFrame(co).reset_index()

c.columns = ['batsman','bowls']
grouped = bats.groupby('batsman').agg({'batsman_runs':'sum'}).reset_index()

bats = grouped.merge(fours,how ='left',on='batsman')

bats = bats.merge(sixes,how ='left',on='batsman')

bats = bats.merge(c,how ='left',on='batsman')

bats = bats.fillna(0)
sr_bound = ((bats.fours+bats.sixes)/(bats.bowls))*100

sr = (bats.batsman_runs/bats.bowls)*100

sr = pd.DataFrame({'sr_bnd':sr_bound,

     'sr':sr})

bats = bats.merge(sr,how ='left',left_index=True,right_index=True)

bats.describe()
matches = data.groupby('batsman').agg({'match_id':'nunique'}).reset_index()

matches = matches['match_id']

bats = bats.merge(pd.DataFrame({'matches':matches}),how ='left',left_index=True,right_index=True)
bats.sample()
gr = data1.groupby(['batsman','bowler']).agg({'total_runs':'sum','extra_runs':'sum','wkt':'count','match_id':'nunique','fr':'sum','sx':'sum'})

gr