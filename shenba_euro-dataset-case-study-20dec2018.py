import pandas as pd
import numpy as np
import os
os.listdir('../input/')

euro = pd.read_csv("../input/Euro_dataset (1).csv")
euro.head(3)
euro.shape
euro.tail(3)
euro.dtypes
euro.describe()
euro.columns
len(euro['Team'].unique())

discipline_Df = euro[['Team', 'Yellow Cards', 'Red Cards']]
discipline_Df
discipline_Df.sort_values(by=['Red Cards', 'Yellow Cards'], inplace=True)
discipline_Df
euro['Goals conceded']
euro[(euro['Goals'] >= 6) & (euro['Goals conceded'] <= 6)]['Team'].values

euro.iloc[ : , :7]

euro[(euro['Team'] == 'England') | (euro['Team'] == 'Italy') | (euro['Team'] =='Russia')][['Team', 'Shooting Accuracy']]
teams = ['England', 'Italy', 'Russia']
euro[euro['Team'].isin(teams)]


euro.groupby(by=['Team']).agg({'Goals':'sum'})

euro[euro['Yellow Cards'] ==euro['Yellow Cards'].max()]
euro.sort_values(by=['Yellow Cards'], ascending=False).head(1)
euro[euro['Headed goals'] ==euro['Headed goals'].max()]
euro.sort_values(by=['Headed goals'], ascending=False).head(1)
euro.groupby(['Team']).agg({'Yellow Cards': 'max'}).sort_values(by=['Yellow Cards'], ascending = False).head(1)

