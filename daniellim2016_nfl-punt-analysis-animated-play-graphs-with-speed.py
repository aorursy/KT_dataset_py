import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from shapely.geometry import LineString

video_review = pd.read_csv('..//input//video_review.csv')

for col in ['Player_Activity_Derived', 'Turnover_Related', 'Primary_Impact_Type', 'Primary_Partner_Activity_Derived', 'Friendly_Fire']:
    video_review[col]=video_review[col].astype('category')

video_review['Primary_Partner_GSISID'] = pd.to_numeric(video_review['Primary_Partner_GSISID'],
                                                       errors = 'coerce', 
                                                       downcast='integer')
    
#video_review.describe(include='all')
def label_ngs (row):
    year = str(row['Season_Year'])
    stype = row['Season_Type']
    week = row['Week']
    if stype != 'Reg':
        return '-'.join(['NGS', year, stype.lower()])
    elif week < 7 :
        return '-'.join(['NGS', year, stype.lower(),'wk1-6'])
    elif week < 13 :
        return '-'.join(['NGS', year, stype.lower(),'wk7-12'])
    else:
        return '-'.join(['NGS', year, stype.lower(),'wk13-17'])
game_data = pd.read_csv('..//input//game_data.csv')
review_game_data = pd.merge(video_review[['GameKey']],game_data, on='GameKey')
review_game_data['NGS_File'] = review_game_data.apply(label_ngs, axis=1)

ngs_file_list = list(review_game_data['NGS_File'].unique())
#review_game_data
play_information = pd.read_csv('..//input//play_information.csv')
video_footage_control = pd.read_csv('..//input//video_footage-control.csv')
video_footage_injury = pd.read_csv('..//input//video_footage-injury.csv')
play_player_role = pd.read_csv('..//input//play_player_role_data.csv')
player_punt_data = pd.read_csv('..//input//player_punt_data.csv')

player_punt_data = player_punt_data.groupby('GSISID').agg({'Number':lambda x: list(x), 'Position': 'last'})
#PPL in punt coverage?
punt_coverage = ['GL','PLW','PLT','PLG','PLS','PRG','PRT','PRW','PC','PPR','P','GR','GRo','GRi','GLo','GLi','PPRo',
                 'PPRi','PPL', 'PPLi', 'PPLo']
#PDM in punt return?
punt_return = ['VL','PDL1','PDL2','PDL3','PDR3','PDR2','PDR1','VR','VR','PLL','PLM','PLR','PFB','PR','PLM1','PDR4',
               'PDR5','PDR6','VRi','VRo','VLo','VLi','PDL4','PDL5','PDL6','PLR1','PLR2','PLR3','PLL3','PLL2','PLL1','PDM']

play_player_role['Role'].unique()
GameKey_list = list(video_review['GameKey'].unique())
PlayID_list = list(video_review['PlayID'].unique())
play_player_role_final = play_player_role[(play_player_role['GameKey'].isin(GameKey_list)) & 
                                          (play_player_role['PlayID'].isin(PlayID_list))].copy()
#Designate based on which side of play each player is on, 'Punt' or 'Return' Team
def action_me(row):
    if row.Role in punt_coverage:
        return 'Punt'
    elif row.Role in punt_return:
        return 'Return'
    else:
        return 'np.nan'

play_player_role_final.loc[:,'Team_Action'] = play_player_role_final.apply(action_me, axis=1)
temp_players = list(video_review['GSISID'].unique())
injured_players = pd.merge(pd.merge(play_player_role_final, video_review, on = ['Season_Year','GameKey','PlayID','GSISID']),
                           player_punt_data, on = 'GSISID')

action_count = injured_players.groupby('Team_Action').count().sort_values('Season_Year', ascending=False).iloc[:,0]
activity_count = injured_players.groupby('Player_Activity_Derived').count().sort_values('Season_Year', ascending=False).iloc[:,0]
role_count = injured_players.groupby('Role').count().sort_values('Season_Year', ascending=False).iloc[:,0]
impact_count = injured_players.groupby('Primary_Impact_Type').count().sort_values('Season_Year', ascending=False).iloc[:,0]
position_count = injured_players.groupby('Position').count().sort_values('Season_Year', ascending=False).iloc[:,0]
position_count2 = injured_players.groupby(['Role','Position']).count().iloc[:,0].unstack()

fig, axes = plt.subplots(3, 2, figsize = (14,10))
fig.tight_layout()

action_count.plot.bar(rot=0,ax = axes[0][0])
activity_count.plot.bar(rot=0,ax = axes[0][1])
role_count.plot.bar(rot=90,ax = axes[1][0])
impact_count.plot.bar(rot=0,ax = axes[1][1])
position_count.plot.bar(rot=0,ax = axes[2][0])

axes[2][1].imshow(position_count2.transpose().values, cmap=cm.gray)

plt.setp(axes[2][1], xticklabels=position_count2.index.tolist(),
         yticklabels=position_count2.columns.tolist(),
         xticks=list(range(position_count2.shape[0])), 
         yticks=list(range(position_count2.shape[1])))
plt.setp(axes[2][1].get_xticklabels(), rotation=90)

plt.subplots_adjust(hspace=0.3)
plt.show()
action_activity_count = injured_players.groupby(['Team_Action','Player_Activity_Derived'], sort=False).count().iloc[:,0].unstack()
action_impact_count = injured_players.groupby(['Team_Action','Primary_Impact_Type'], sort=False).count().iloc[:,0].unstack()
activity_impact_count = injured_players.groupby(['Player_Activity_Derived','Primary_Impact_Type'], sort=False).count().iloc[:,0].unstack()
role_impact_count = injured_players.groupby(['Role','Primary_Impact_Type'], sort=True).count().iloc[:,0].unstack()
role_action_count = injured_players.groupby(['Role','Team_Action'], sort=True).count().iloc[:,0].unstack()
role_activity_count = injured_players.groupby(['Role','Player_Activity_Derived'], sort=True).count().iloc[:,0].unstack()

fig2, axes2 = plt.subplots(3, 2, figsize = (12,8))
fig2.tight_layout()
plt.subplots_adjust(wspace=0.5, hspace=0.3)

action_activity_count.plot.bar(rot=0,ax = axes2[0][0], stacked=True)
action_impact_count.plot.bar(rot=0,ax = axes2[0][1], stacked=True)
activity_impact_count.plot.bar(rot=0,ax = axes2[1][0], stacked=True)
role_action_count.plot.bar(rot=0,ax = axes2[1][1], stacked=True)
role_impact_count.plot.bar(rot=0,ax = axes2[2][1], stacked=True)
role_activity_count.plot.bar(rot=0,ax = axes2[2][0], stacked=True)

plt.setp(axes2[2][1].get_xticklabels(), rotation=90)
plt.setp(axes2[1][1].get_xticklabels(), rotation=90)
plt.setp(axes2[2][0].get_xticklabels(), rotation=90)
axes2[0][0].legend(loc='center left',bbox_to_anchor=(1, 0.5))
axes2[0][1].legend(loc='center left',bbox_to_anchor=(1, 0.5))
axes2[1][0].legend(loc='center left',bbox_to_anchor=(1, 0.5))
axes2[1][1].legend(loc='center left',bbox_to_anchor=(1, 0.5))
axes2[2][0].legend(loc='center left',bbox_to_anchor=(1, 0.5))
axes2[2][1].legend(loc='center left',bbox_to_anchor=(1, 0.5))

plt.show()
ngs_file = {}
def dataprep(game_info=None, filename=None):
    myGameKey, myPlayID, GSISID1, GSISID2 = game_info
    if (filename not in ngs_file.keys()): 
        ngs_file[filename] =  pd.read_csv('..//input//' + filename + '.csv')
        ngs_file[filename].loc[:,'Speed'] = ngs_file[filename]['dis']/0.1 *3600/1760

    ngs_file[filename]['Time'] = pd.to_datetime(ngs_file[filename]['Time'])
    temp_pd = ngs_file[filename].sort_values('Time').copy()
    temp_pd = temp_pd[(temp_pd.GameKey == myGameKey) & (temp_pd.PlayID == myPlayID)]
    if pd.isnull(GSISID2):
        other_player_df = temp_pd[~(temp_pd.GSISID == GSISID1)].copy()
    else:
        other_player_df = temp_pd[~((temp_pd.GSISID == GSISID1) | (temp_pd.GSISID == GSISID2))].copy()

    #speed should be the distance(will be in yards) divided by time (which is 0.1 sec intervals) corrected to MPH
    p1_speed = temp_pd[temp_pd.GSISID == GSISID1]
#    Alternative method to calculate speed
#    p1_speed[['time_diff','x_diff', 'y_diff']] = p1_speed[['Time','x','y']].diff().fillna(0.)
#    p1_speed['Speed']= np.sqrt(p1_speed.x_diff**2+ p1_speed.y_diff**2)/0.1 *3600/1760
#    p1_speed = p1_speed.drop(['time_diff','x_diff', 'y_diff'], 1)
    
    #if GSISID is "Unclear" i.e. player fell on his own
    if pd.isnull(GSISID2):
        p1_speed.rename(columns={'x': 'x1', 'y': 'y1', 'Speed': 'Speed1'}, inplace=True)
        return p1_speed, other_player_df
    
    p2_speed = temp_pd[temp_pd.GSISID == GSISID2]
#    p2_speed[['time_diff','x_diff', 'y_diff']] = p2_speed[['Time','x','y']].diff().fillna(0.)
#    p2_speed['Speed']= np.sqrt(p2_speed.x_diff**2 + p2_speed.y_diff**2)/0.1 *3600/17603
#    p2_speed = p2_speed.drop(['time_diff','x_diff', 'y_diff'], axis=1)

    relev_cols = ['Time','GSISID', 'x', 'y', 'dis','o', 'dir', 'Speed']
    player_speed = pd.merge(p1_speed, p2_speed.loc[:,relev_cols], on='Time', validate='one_to_one',suffixes = ['1','2'])
    return player_speed, other_player_df

def speed_plots(vr_row=None, gd_row=None, filename=None):
    #%matplotlib notebook
    
    if vr_row is None:
        print("Provide a pandas row from video_review.csv dataframe")
        return None
    if gd_row is None:
        print("Provide a pandas row from game_data.csv dataframe")
        return None
    if filename is None:
        print('Specify a Filename to save final mp4')
        return None    
    
    tempGameKey, tempPlayID, tempGSISID1, tempGSISID2 =list(vr_row[['GameKey', 'PlayID', 'GSISID', 'Primary_Partner_GSISID']])
    player1_team = play_player_role_final[((play_player_role_final.GameKey == tempGameKey) & 
                                           (play_player_role_final.PlayID == tempPlayID) & 
                                           (play_player_role_final.GSISID == tempGSISID1))].iloc[-1,-1]
    player2_team = play_player_role_final[((play_player_role_final.GameKey == tempGameKey) & 
                                           (play_player_role_final.PlayID == tempPlayID) & 
                                           (play_player_role_final.GSISID == tempGSISID2))].iloc[-1,-1] if not pd.isnull(tempGSISID2) else ''
    
    player_speed_final, other_players_final = dataprep([tempGameKey, tempPlayID, tempGSISID1, tempGSISID2],gd_row[-1])
    
    text_col = ['HomeTeamCode','VisitTeamCode','Season_Year','Season_Type','StadiumType','Turf','GameWeather','Temperature']
    game_text = review_game_data[review_game_data.GameKey == vr_row['GameKey']][text_col].iloc[0].tolist()
    
    
    fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(14,8))
    ax1.set_xlabel('Time',fontsize=10)
    ax1.set_ylabel('Speed (MPH)',fontsize=10)
    ax1.set_title('Players\' Speed',fontsize=20, y=1.1)
    ax1.set_xlim(np.min(player_speed_final['Time']), np.max(player_speed_final['Time'] + np.timedelta64(2, 's')))
    ax1.set_ylim(0,40)
    
    ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax3.set_ylabel('Combine Speed', color='red')  # we already handled the x-label with ax1
    ax3.tick_params(axis='y', labelcolor='red')
    ax3.set_ylim(0,55)
    ax3.set_xlim(np.min(player_speed_final['Time']), np.max(player_speed_final['Time'] + np.timedelta64(2, 's')))

    
    ax2.grid(which='major', axis='x', linestyle='--')
    ax2.set_xlim(0, 120)
    ax2.set_ylim(0,53.3)
    ax2.set_title('Players\' Route - '+ vr_row['Primary_Impact_Type'],fontsize=20)
    ax2.set_xlabel(' - '.join([' vs. '.join(game_text[:2])]+
                              [str(x) for x in game_text[2:-1]] +
                              [str(game_text[-1])+' Degree F']),fontsize=10)
    plt.setp(ax2, xticks=range(10,120,10), xticklabels=['Goal', 10, 20, 30, 40, 50, 40, 30, 20, 10, 'Goal'],yticks=[])

    sc1 = ax2.scatter([], [], c = 'g', marker = 'o')
    sc2 = ax2.scatter([], [], c = 'm', marker = 'x')
    
    plt.subplots_adjust(hspace=0.3)
    
    def animate(i):
        new_df = player_speed_final.iloc[:int(i+1)]
        game_time = new_df.Time.iloc[-1]
        test = (player_speed_final.shape[1] > 12)
        graphs = []
        temp = ax1.plot(new_df['Time'], new_df['Speed1'], color = 'blue')
        graphs.append(temp[0])
        if test:
            temp = ax1.plot(new_df['Time'], new_df['Speed2'], color = 'orange')
            temp2 = ax3.plot(new_df['Time'],(new_df['Speed1'] + new_df['Speed2']), color = 'red')
            graphs = graphs + [temp[0], temp2[0]]
        ax2.plot(new_df['x1'], new_df['y1'], color = 'blue')
        if test:
            ax2.plot(new_df['x2'], new_df['y2'], color = 'orange')
        event = new_df.Event.iloc[-1]
        if not(pd.isnull(event)):
            ax1.annotate(event, xy=(new_df.Time.iloc[-1], new_df.Speed1.iloc[-1]),
                   xycoords='data', xytext=(0, 70), textcoords='offset points',
                   arrowprops=dict(arrowstyle="->"))
        
        legend_text = ['Player1 - '+ vr_row['Player_Activity_Derived'] + ' - ' + player1_team]
        if test:
            legend_text = legend_text + [('Player2 - '+ vr_row['Primary_Partner_Activity_Derived'] + ' - ' + player2_team), 
                                         'Combined Speed']
            
        plt.legend([sc1, sc2] + graphs, ['Punting Team', 'Return Team'] + legend_text, 
                   loc='lower center', bbox_to_anchor=(0.5, 0.98), fancybox=True, ncol=len(legend_text)+2)
        
        #position of other players
        other_player_pos = other_players_final[other_players_final.Time == game_time]
        other_player_pos = pd.merge(play_player_role_final, other_player_pos, on = ['Season_Year','GameKey','PlayID','GSISID'])
        
        new_array1 = other_player_pos.loc[other_player_pos['Team_Action'] == 'Punt'].loc[:,['x','y']].as_matrix()
        new_array2 = other_player_pos.loc[other_player_pos['Team_Action'] == 'Return'].loc[:,['x','y']].as_matrix()
        n = other_player_pos['GSISID'].tolist()
        
        sc1.set_offsets(new_array1)
        sc2.set_offsets(new_array2)
        return([sc1, sc2] + graphs)
        
    ani = animation.FuncAnimation(fig, animate, frames=len(player_speed_final), repeat=True, interval=1, blit=True)
    ani.save((filename + '.gif'), writer='imagemagick', fps=10)
    plt.show()
for x in range(len(review_game_data)):
    speed_plots(video_review.iloc[x,:],review_game_data.iloc[x,:],'Play'+str(x))
    print('Plot ' + str(x) + ' done')