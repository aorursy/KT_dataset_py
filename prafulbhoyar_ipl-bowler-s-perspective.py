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
dfbase = pd.read_csv("../input/deliveries.csv")



df = dfbase.groupby(['match_id','inning','bowler','over']).agg({'total_runs':sum})

df.reset_index(inplace= True)

#df.groupby(['over']).agg({'total_runs':['min','max','mean']})

#df = df[df['over'] >16]

df = df.groupby('bowler').agg({'total_runs':['max','min','count','mean']})

df. reset_index(inplace= True)

df.columns = df.columns.droplevel()

df= df[df['count']>50].sort_values(by = 'mean',ascending= True)

df.head()

# top 5 bowlers who have bowled more than 50 overs and have best economy 

df = dfbase.groupby(['match_id','inning','bowler','over']).agg({'total_runs':sum})

df.reset_index(inplace= True)

#df.groupby(['over']).agg({'total_runs':['min','max','mean']})

df = df[df['over'] >16]

df = df.groupby('bowler').agg({'total_runs':['max','min','count','mean']})

df. reset_index(inplace= True)

df.columns = df.columns.droplevel()

df= df[df['count']>50].sort_values(by = 'mean',ascending= True)

df.head()

# top 5 bowlers who have bowled more than 50 overs and have best economy in death overs
df = dfbase.groupby(['bowler']).agg({'total_runs':['sum','count']})

df.reset_index(inplace= True)

df.columns = ['bowler','runs','deliveries']

df['economy'] = (df['runs'] /df['deliveries'])*6.

df = df[df['deliveries'] > 300]

df = df.sort_values(by='economy',ascending=True)

df.head()

#top 6 bowlers who have bowled more than 50 overs and have the best economy