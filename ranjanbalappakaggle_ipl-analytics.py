



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import pprint
deliveries=pd.read_csv('../input/deliveries.csv')

matches=pd.read_csv('../input/matches.csv')
deliveries.head()
matches.head()
wickets_group=deliveries[deliveries['player_dismissed'].isnull()==False].groupby('over')

wickets_over=wickets_group.size()



wc=wickets_over.plot(

kind='bar',

rot=0,

title='Number Of Wickets Taken In Each Over',

color='#00304e',

)





wc.set_ylabel('Number Of Wickets',fontsize=8)
#Partnership/Wk

wicket=0

partnerships=0

first=0

count=0

max_partnership=[0,0,0,0,0,0,0,0,0,0,0]

batsmens=[{"batsman1":"","batsman2":"","Partnership":0} for i in range(11)]

for key,g in deliveries.groupby(['match_id','inning']):

    #print(g['player_dismissed'].isnull()==True)\

    index_list=g[g['player_dismissed'].isnull()==False].index.tolist()

    last=0

    wicket=0

    for i in index_list:

        last=i  

        partnerships=deliveries[first:last]['total_runs'].sum()

        #print(partnerships)

        if partnerships> max_partnership[wicket]:

            batsmens[wicket].update({"batsman1":deliveries.loc[last]['batsman'],"batsman2":deliveries.loc[last]['non_striker']

                                    ,"Partnership":partnerships})

            max_partnership[wicket]=partnerships

        first=last+1

        wicket+=1  

        

    partnerships=deliveries[first:g.index[-1]+1]['total_runs'].sum()

   

    if partnerships> max_partnership[wicket]:

        batsmens[wicket]={"batsman1":deliveries.loc[first]['batsman'],"batsman2":deliveries.loc[first]['non_striker']

                                ,"Partnership":partnerships}

        max_partnership[wicket]=partnerships

    first=g.index[-1]+1     

batsmens=batsmens[0:10]

wicket=1

for p in batsmens:

    print('{3} {0}-{1} {2}'.format(p['batsman1'],p['batsman2'],p['Partnership'],wicket))

    wicket+=1