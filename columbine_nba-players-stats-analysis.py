# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import altair as alt



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os, warnings

warnings.filterwarnings("ignore")



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
alt.renderers.enable('kaggle')
%%time

PlayerSet = pd.read_csv('/kaggle/input/nba-players-stats-20142015/players_stats.csv')
print(PlayerSet.columns)
height_weight_info = PlayerSet[['Name', 'Height', 'Weight', 'Pos']]



height_weight_info.head(5)
alt.Chart(height_weight_info).mark_circle(size=20).encode(

    x = 'Height',

    y = 'Weight',

    color = 'Pos',

    tooltip = ['Name', 'Height', 'Weight', 'Pos']

).properties(

    width=600, 

    height=600

).interactive()
three_points_rate = PlayerSet[['Team', '3P%']]



team_three_points_rate = three_points_rate.groupby('Team').mean()



team_three_points_rate['Team']  = team_three_points_rate.index



team_three_points_rate.index = [i for i in range(30)]



team_three_points_rate.head(3)
team_data = alt.Chart(team_three_points_rate).mark_bar(

    color='lightblue'

).encode(

    x = 'Team',

    y = '3P%'

)

mean_rate = alt.Chart(team_three_points_rate).mark_rule(

    color='green'

).encode(

    y = 'mean(3P%)'

)

(team_data + mean_rate).properties(width=600)
LAL_PLAYER = PlayerSet.loc[PlayerSet['Team']=='LAL']



LAL_PLAYER = LAL_PLAYER[['Name', 'Team', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']]



LAL_PLAYER.index = range(LAL_PLAYER.shape[0])



t_min, t_pts, t_reb, t_ast, t_stl, t_blk, t_tov =  LAL_PLAYER[['MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']].sum()



attr_list = []



for i in LAL_PLAYER.index:

    

    p_name, p_team, p_min, p_pts, p_reb, p_ast, p_stl, p_blk, p_tov = LAL_PLAYER.iloc[i]

    

    attr_list.append([p_name, 'MIN', p_min/t_min])

    attr_list.append([p_name, 'PTS', p_pts/t_pts])

    attr_list.append([p_name, 'REB', p_reb/t_reb])

    attr_list.append([p_name, 'AST', p_ast/t_ast])

    attr_list.append([p_name, 'STL', p_stl/t_stl])

    attr_list.append([p_name, 'BLK', p_blk/t_blk])

    attr_list.append([p_name, 'TOV', p_tov/t_tov])

    

LAL_DATA = pd.DataFrame(attr_list, columns=['Name', 'Contribute_name', 'Values'])
import altair as alt



bars = alt.Chart(LAL_DATA).mark_bar().encode(

    x=alt.X('sum(Values)', stack='zero'),

    y=alt.Y('Contribute_name'),

    color=alt.Color('Name')

)



text = alt.Chart(LAL_DATA).mark_text(dx=-15, dy=3, color='white').encode(

    x=alt.X('sum(Values):Q', stack='zero'),

    y=alt.Y('Contribute_name'),

    detail='Name',

    text=alt.Text('sum(Values):Q', format='.4f')

)



(bars + text).properties(width=800, height=300)