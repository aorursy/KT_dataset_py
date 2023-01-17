import numpy as np 

import pandas as pd 



data = pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/players_20.csv')

pd.set_option('display.max_columns', None)
data.drop(['sofifa_id', 'player_url', 'dob'], axis=1, inplace=True)

data.drop(data[data['value_eur']<1].index, inplace= True)



data.loc[:,['short_name', 'age', 'overall' ,'potential','value_eur']].sort_values(by='overall', ascending= False)
for p in range (48,95):

    oall = data[data['overall']== p]

    pot = data[data['potential']== p]

    mean= round(oall.loc[:,'value_eur'].mean())

    data.at[pot.index,'potential_value']= mean
data['value_change']= data['potential_value']-data['value_eur']

data.loc[:,['short_name', 'overall' ,'potential','value_eur','potential_value','value_change']].sort_values(by='value_change', ascending= False).head(20)

potential = data.loc[(data['potential'])-(data['overall'])>1]

potential.loc[:,['short_name', 'age', 'overall' ,'potential','value_eur','potential_value','value_change']].sort_values(by='potential', ascending= False).head(1)