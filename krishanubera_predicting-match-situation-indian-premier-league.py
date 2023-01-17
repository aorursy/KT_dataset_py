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
pd.set_option('display.max_rows', 30)
matches = pd.read_csv("/kaggle/input/ipldata/matches.csv")

deliveries = pd.read_csv("/kaggle/input/ipldata/deliveries.csv")
matches.head()
deliveries.head()
deliveries.info();
deliveries.loc[deliveries['player_dismissed'].isnull() & deliveries['fielder'].notnull() ]
## get randome sample

def get_random_sample(df, block_size=6):

    start = np.random.randint(low=0, high=len(df)-block_size)

    return df.iloc[start:start+block_size:]

deliveries['team_score'] = deliveries.groupby(['match_id','inning'])['total_runs'].cumsum()
get_random_sample(deliveries).head(10)[['match_id', 'batsman','over', 'ball' ,'team_score']]
deliveries[(deliveries['match_id']==1) & (deliveries['inning']==1) & (deliveries['over']==20) & (deliveries['ball']==6)]['team_score']
deliveries['out_ind'] = deliveries['player_dismissed'].notnull()

deliveries['wicket_fallen'] = deliveries.groupby(['match_id','inning'])['out_ind'].cumsum()
get_random_sample(deliveries).head(10)[['match_id', 'batsman','over', 'ball' ,'team_score','player_dismissed', 'wicket_fallen']]
deliveries[(deliveries['match_id']==1) & (deliveries['inning']==1) & (deliveries['over']==20) & (deliveries['ball']==6)]['wicket_fallen']