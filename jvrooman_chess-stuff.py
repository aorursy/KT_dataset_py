# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#import the data

df_games = pd.read_csv('../input/games.csv')
#show basic data of the data

print(df_games.columns)

print(df_games.describe())

print(df_games.info())
#strip off unwanted columns

keep = ['id','rated','turns','victory_status','winner','increment_code','white_rating','black_rating','moves','opening_eco','opening_name','opening_ply']



df_games = df_games[keep]

print(df_games.columns)
df_minis = df_games[df_games['turns'] <= 30]

games = len(df_games.index)

minis = len(df_minis.index)

percent_minis = round(minis/games*100,2)



print("{mi}/{gmes} = {percent}% games are minis.".format(mi=minis,gmes=games,percent = percent_minis))

#pie chart of minis vs non-minis

colors = ['blue','orange']

plt.pie([minis,games-minis], labels=["Minis","Not Minis"], colors=colors, startangle=90, autopct='%.1f%%')

plt.show()
time_control_types = {'classic':{'start':60,'end':999},'rapid':{'start':11,'end':59},'blitz':{'start':5,'end':10}, 'bullet':{'start':0,'end':4}}

game_types = ['classic','rapid','blitz']

def get_time_control(minutes):

    if minutes >= time_control_types['classic']['start']:

        return 'classic'

    elif minutes >= time_control_types['rapid']['start']:

        return 'rapid'

    elif minutes >= time_control_types['blitz']['start']:

        return 'blitz'

    else:

        return 'bullet'

    

temp_df = df_games['increment_code'].str.split("+",n=1,expand=True)

temp_df[0]=temp_df[0].apply(lambda x: get_time_control(int(x)))

print(temp_df.head(10))

df_games=pd.concat([df_games,temp_df[0]], axis=1)
df_games.rename(columns={0:'time_control_type'},inplace=True)

print(df_games.columns)
ends = df_games['winner'].unique()

finish_by_control={}

control_totals={}

for control in game_types:

    finish_by_control[control]={}

    control_totals[control]=0

    for end in ends:

        finish_by_control[control][end]=(df_games[(df_games['winner']==end) & (df_games['time_control_type']==control)].count()['winner'])

        control_totals[control]=finish_by_control[control][end]+control_totals[control]

print(finish_by_control)

print(control_totals)

end_percents={} 

for control in game_types:

    end_percents[control]={}

    for end in ends:

        end_percents[control][end]=(round((finish_by_control[control][end]/control_totals[control])*100,2))

        

end_percents
#graphing the win rates grouped by time control



#bar widiths and postitions

bar_width=0.25

bpos=np.arange(len(end_percents.keys()))



#create the bars!

fig,ax=plt.subplots()

white = ax.bar(bpos-bar_width,[end_percents[control]['white'] for control in game_types],bar_width,label='White')

black = ax.bar(bpos,[end_percents[control]['black'] for control in game_types],bar_width,label='Black')

draws = ax.bar(bpos+bar_width,[end_percents[control]['draw'] for control in game_types],bar_width,label='Draw')



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Win Percentage')

ax.set_title('Win Percentage by Control Type')

ax.set_xticks(bpos)

ax.set_xticklabels([x.capitalize() for x in game_types])

ax.legend()



fig.tight_layout()

plt.show()
list_opening_moves = ['e4','d4','Nf','other']

opening_moves={}

for move in list_opening_moves:

    if (move != 'other'):

        opening_moves[move]=df_games[df_games['moves'].str.slice(0,2,1)==move].count()['moves']

    else:

        opening_moves[move]=df_games[~df_games['moves'].str.slice(0,2,1).isin(list_opening_moves)].count()['moves']



print(opening_moves)

result_by_opening_move={}

for opening in list_opening_moves:

    result_by_opening_move[opening]={}

    

    for end in ends:

        if (opening != 'other'):

            result_by_opening_move[opening][end]=(df_games[(df_games['winner']==end) & (df_games['moves'].str.slice(0,2,1)==opening)].count()['moves'])

        else:

            result_by_opening_move[opening][end]=(df_games[(df_games['winner']==end) & (~df_games['moves'].str.slice(0,2,1).isin(list_opening_moves))].count()['moves'])

result_by_opening_move
move_percentage={}

for move in list_opening_moves:

    move_percentage[move]={}

    for end in ends:

        move_percentage[move][end]=(round((result_by_opening_move[move][end]/opening_moves[move])*100,2))

move_percentage
#graphing the win rates grouped by opening move



#bar widiths and postitions

bar_width=.25

bpos=np.arange(len(move_percentage.keys()))



#create the bars!

fig,ax=plt.subplots()

white = ax.bar(bpos-bar_width,[move_percentage[move]['white'] for move in list_opening_moves],bar_width,label='White')

black = ax.bar(bpos,[move_percentage[move]['black'] for move in list_opening_moves],bar_width,label='Black')

draws = ax.bar(bpos+bar_width,[move_percentage[move]['draw'] for move in list_opening_moves],bar_width,label='Draw')



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Win Percentage')

ax.set_title('Win Percentage by Move')

ax.set_xticks(bpos)

ax.set_xticklabels(['King Pawn','Queen Pawn','Reti','Other'])

ax.legend()



fig.tight_layout()

plt.show()