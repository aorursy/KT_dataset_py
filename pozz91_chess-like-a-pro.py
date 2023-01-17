import pandas as pd

import numpy as np

df=pd.read_csv('../input/chess/games.csv')

chess=df[['opening_ply', 'black_rating', 'white_rating','victory_status', 'turns',

      'opening_name', 'winner']]

chess.head()
#splitting

black= chess[chess['winner']=='black']

white= chess[chess['winner']=='white']
w_group=white.groupby(['victory_status','opening_name'])

w_group=w_group.agg({'black_rating' :'mean', 'white_rating':'mean', 'turns': 'mean', 'winner': 'count'})

ninety_percentile_white=np.percentile(white['white_rating'], q=90)



#all players

w_group_all=w_group.sort_values(by='winner', ascending=False).reset_index()



#all-percentage

w_group_all_count=w_group_all.groupby('victory_status').sum()['winner']

w_group_all_percentage=np.round(w_group_all_count/w_group_all_count.sum(), 2)



#pro players

w_group_pro= w_group[w_group['white_rating']>= ninety_percentile_white].sort_values(by='winner', ascending=False).reset_index()



#pro percentage

w_pro_count=w_group_pro.groupby('victory_status').sum()['winner']

w_pro_percentage= np.round(w_pro_count/w_pro_count.sum(), 2)
#analyze black

b_group=black.groupby(['opening_name','victory_status'])

b_group=b_group.agg({'black_rating' :'mean', 'white_rating':'mean', 'turns': 'mean', 'winner': 'count'})

b_ninety_percentile_black=np.percentile(black['black_rating'], q=90)



#all-players

b_all=b_group.sort_values(by='winner',ascending=False).reset_index()



#percentage all

b_percentage_all=np.round(b_all.groupby('victory_status').sum()['winner']/b_all.groupby('victory_status').sum()['winner'].sum(), 2)



#top players

b_group_pro=b_group[b_group['black_rating']>= b_ninety_percentile_black].sort_values(by='winner', ascending=False).reset_index()



#percentage-top

b_pro_count=b_group_pro.groupby('victory_status').sum()['winner']

b_pro_percentage= np.round(b_pro_count/b_pro_count.sum(), 2)

archetypes=df['opening_name'].str.extract(r"([A-Za-z'-]*\s[A-Za-z'-]*[\s[A-Za-z'-]*]?)")

len(df.opening_name)

len(df['opening_name'])== len(archetypes)
chess['archetypes']=archetypes[0].str.strip()

archetypes=chess.groupby(['archetypes', 'victory_status']).count()[['opening_name']].sort_values(by='opening_name', ascending=False).reset_index()



#top

overall_white_percentile=np.percentile(chess['white_rating'], q=90)

overall_black_percentile=np.percentile(chess['black_rating'], q=90)

white_perc=chess[chess['white_rating']>= overall_white_percentile] 

black_perc= chess[chess['black_rating']>= overall_black_percentile]



#put togheder overall top players

overall_top=(pd.concat([white_perc, black_perc])).drop_duplicates()

top=overall_top.groupby(['archetypes', 'victory_status']).count()[['opening_name']].reset_index().sort_values(by='opening_name',ascending=False)