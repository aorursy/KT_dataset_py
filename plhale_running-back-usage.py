import pandas as pd  # data manipultion librabry
import numpy as np # numerical cmputation library

# Display up to 120 columns of a dataframe
pd.set_option('display.max_columns', 120)

import matplotlib.pyplot as plt  # plotting library
%matplotlib inline

# Internal ipython tool for setting figure size
from IPython.core.pylabtools import figsize

# Seaborn for visualization
import seaborn as sns
# sns.set(font_scale = 2)
def match(roster):   
    rb_roster = nfl_roster[nfl_roster['Pos']=='RB'][['GSIS_ID'][0]].values
    return roster in rb_roster
import glob

path =r'../input/roster'
allFiles = glob.glob(path + '/*.csv')
frame = pd.DataFrame()
list_ = []

for file_ in allFiles:
    df_r = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df_r)
nfl_roster = pd.concat(list_)

nfl_roster[(nfl_roster['Pos'] == 'RB') &
          (nfl_roster['Team'] == 'DAL')].head()
rec_df = pd.read_csv("../input/football/game_receiving_df.csv")
rec_df = rec_df[rec_df.Receiver_ID.map(match)]
rec_df[(rec_df['Team']=='DAL') & (rec_df['Player_Name']=='E.Elliott')]
rushing = pd.read_csv("../input/football/game_rushing_df.csv")
rushing = rushing[rushing.Rusher_ID.map(match)]
rushing[(rushing['Team']=='DAL') & (rushing['Player_Name']=='E.Elliott')]
df = (rushing.
            rename(columns = {'Total_Yards':'Total_Rushing_Yards',
                       'Yards_per_Drive':'Rushing_Yards_per_Drive',
                       'Fumbles':'Rushing_Fumbles',
                        'TD_to_Fumbles':'Rush_TD_to_Fumbles',
                        'TDs':'Rush_TDs',
                        'Total_EPA':'Rush_Total_EPA',
                        'Success_Rate':'Rush_Success_Rate',
                        'Total_WPA':'Rush_Total_WPA',
                        'WPA_per_Drive':'Rush_WPA_per_Drive',
                        'Win_Success_Rate':'Rush_Win_Success_Rate',
                        'Total_Clutch_EPA':'Rush_Total_Clutch_EPA',
                        'Clutch_EPA_per_Drive':'Rush_Clutch_EPA_per_Drive'
                       }).
      drop(['Fumbles_per_Drive','Drives','Opponent'], axis=1).
      merge(rec_df.
            drop(['Team','Player_Name','Fumbles_per_Drive','Drives','Opponent'],axis=1).
      rename(columns = {'Total_Yards':'Total_Rec_Yards',
                       'Yards_per_Drive':'Rec_Yards_per_Drive',
                       'Fumbles':'Rec_Fumbles',
                       'TDs':'Rec_TDs',
                        'TD_to_Fumbles':'Rec_TD_to_Fumbles',
                        'Total_EPA':'Rec_Total_EPA',
                        'Success_Rate':'Rec_Success_Rate',
                        'Total_WPA':'Rec_Total_WPA',
                        'WPA_per_Drive':'Rec_WPA_per_Drive',
                        'Win_Success_Rate':'Rec_Win_Success_Rate',
                        'Total_Clutch_EPA':'Rec_Total_Clutch_EPA',
                        'Clutch_EPA_per_Drive':'Rec_Clutch_EPA_per_Drive'
                       }), 
            left_on=(['Rusher_ID','GameID']),right_on=(['Receiver_ID','GameID']), how='outer'))
#df = functools.reduce(lambda left,right: pd.merge(left, right, on='GSIS_ID'), dfs)

df[(df['GameID']==2017123108)]
import math

def first_n_digits(num, n):
    return num // 10 ** (int(math.log(num, 10)) - n + 1)
df['Season'] = df.GameID.apply(lambda x: first_n_digits(x,4))
list(df.columns)
# E.Elliott Stats: http://www.nfl.com/player/ezekielelliott/2555224/careerstats
# Model Stats: http://insidethepylon.com/football-science/football-statistics/2016/05/26/nfl-running-back-usage/

combined = (df.groupby(['Season','Team','Player_Name']).
            agg({'Car_per_Drive':'mean',
                 'Targets_per_Drive':'mean',
                 'Carries':'sum',
                  'Total_Rushing_Yards':'sum',
                  'Yards_per_Car':'mean',
                  'Targets':'sum',
                  'Receptions':'sum',
                  'Total_Rec_Yards':'sum',
                  'Yards_per_Target':'mean',
                  'Rush_Total_Clutch_EPA':'mean',
                  'Rush_Win_Success_Rate':'mean',
                  'Rush_Clutch_EPA_per_Drive':'mean',
                  'Rec_Success_Rate':'mean',
                  'Rec_Win_Success_Rate':'mean',
                  'Rec_Clutch_EPA_per_Drive':'mean',
                 'GameID':'count'    
     }).
 reset_index().
 rename(columns = {'GameID':'Games',
                   'Car_per_Drive':'% Rush Att',
                   'Targets_per_Drive':'% Tgts',
                   'Yards_per_Car':'YPC',
                   'Yards_per_Target':'YPT'
                  })
)

(combined[combined['Games']>=10].sort_values(['Season','% Rush Att','Total_Rushing_Yards','% Tgts','Total_Rec_Yards'], ascending=False)
)

combined['workhorse'] = np.where(combined['Games'] < 8, 'backup',
                          (np.where(combined['% Rush Att'] > 2.2, 'workhorse',
                                    (np.where(combined['% Rush Att'] > 1.7 , 'time_share','backup')))))

#combined['team_rb_duo'] = np.where(new_df['workhorse'] == 'workhorse','1 RB', 'Multi RB')
combined
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(combined['% Rush Att'].describe())
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(combined.describe())
combined[combined['Team']=='DAL'].sort_values('workhorse', ascending=False)
combined[(combined.Team == 'DAL')].groupby('workhorse').agg({'Season':len}).reset_index()
def fig6():
    rs_fg = combined[(combined.Team == 'DAL')].groupby('workhorse').agg({'Season':len}).reset_index()
    rs_fg.columns=['workhorse', 'Count']
    rs_fg['Percent Total'] = rs_fg.Count.apply(lambda x: 100 * x / float(rs_fg.Count.sum()))

    po_fg = combined[combined.Team =='NO'].groupby('workhorse').agg({'Season':len}).reset_index()
    po_fg.columns=['workhorse', 'Count']
    po_fg['Percent Total'] = po_fg.Count.apply(lambda x: 100 * x / float(po_fg.Count.sum()))

    sns.set_palette(['green', 'orange', 'red'])


    fig, axes = plt.subplots(2, 2,sharey=True,figsize=(14,7))
    order = ['workhorse','time_share','backup']

    sns.violinplot(ax=axes[0][0], data=combined[(combined.Team == 'DAL')], x='YPC', y='workhorse',order=order, scale='width', bw=0.05)
    sns.violinplot(ax=axes[1][0], data=combined[combined.Team =='NO'], x='YPC', y='workhorse',order=order, scale='width', bw=0.05)
    axes[0][0].set_xlim(0,10)
    axes[1][0].set_xlim(0,10)

    sns.barplot(ax=axes[0][1], data=rs_fg,y='workhorse', x='Percent Total',order=order)
    sns.barplot(ax=axes[1][1], data=po_fg,y='workhorse', x='Percent Total',order=order)
    axes[0][1].set_xlim(0,100)
    axes[1][1].set_xlim(0,100)

    axes[0][1].set_xticklabels(['0%','20%','40%','60%','80%','100%'])
    axes[1][1].set_xticklabels(['0%','20%','40%','60%','80%','100%'])


    axes[0][0].set_title('Workhorse by Yards Per Carried Gained')
    axes[0][0].set_xlabel('')
    axes[0][0].set_ylabel('Dallas Cowboys')

    axes[0][1].set_title('Workhorse Distribution')
    axes[0][1].set_xlabel('')
    axes[0][1].set_ylabel('')

    axes[1][0].set_ylabel('NO')
    axes[1][0].set_xlabel('Yards Gained Distance (yds)')
    axes[1][0].figure

    axes[1][1].set_ylabel('')
    axes[1][1].set_xlabel('Percent Total')
    return fig

fig6()
teams = [['ARI', 'Arizona', 'Cardinals', 'Arizona Cardinals'],
 ['ATL', 'Atlanta', 'Falcons', 'Atlanta Falcons'],
 ['BAL', 'Baltimore', 'Ravens', 'Baltimore Ravens'],
 ['BUF', 'Buffalo', 'Bills', 'Buffalo Bills'],
 ['CAR', 'Carolina', 'Panthers', 'Carolina Panthers'],
 ['CHI', 'Chicago', 'Bears', 'Chicago Bears'],
 ['CIN', 'Cincinnati', 'Bengals', 'Cincinnati Bengals'],
 ['CLE', 'Cleveland', 'Browns', 'Cleveland Browns'],
 ['DAL', 'Dallas', 'Cowboys', 'Dallas Cowboys'],
 ['DEN', 'Denver', 'Broncos', 'Denver Broncos'],
 ['DET', 'Detroit', 'Lions', 'Detroit Lions'],
 ['GB', 'Green Bay', 'Packers', 'Green Bay Packers', 'G.B.', 'GNB'],
 ['HOU', 'Houston', 'Texans', 'Houston Texans'],
 ['IND', 'Indianapolis', 'Colts', 'Indianapolis Colts'],
 ['JAC', 'Jacksonville', 'Jaguars', 'Jacksonville Jaguars', 'JAX'],
 ['KC', 'Kansas City', 'Chiefs', 'Kansas City Chiefs', 'K.C.', 'KAN'],
 ['LA', 'Los Angeles', 'Rams', 'Los Angeles Rams', 'L.A.'],
 ['MIA', 'Miami', 'Dolphins', 'Miami Dolphins'],
 ['MIN', 'Minnesota', 'Vikings', 'Minnesota Vikings'],
 ['NE', 'New England', 'Patriots', 'New England Patriots', 'N.E.', 'NWE'],
 ['NO', 'New Orleans', 'Saints', 'New Orleans Saints', 'N.O.', 'NOR'],
 ['NYG', 'Giants', 'New York Giants', 'N.Y.G.'],
 ['NYJ', 'Jets', 'New York Jets', 'N.Y.J.'],
 ['OAK', 'Oakland', 'Raiders', 'Oakland Raiders'],
 ['PHI', 'Philadelphia', 'Eagles', 'Philadelphia Eagles'],
 ['PIT', 'Pittsburgh', 'Steelers', 'Pittsburgh Steelers'],
 ['SD', 'San Diego', 'Chargers', 'San Diego Chargers', 'S.D.', 'SDG'],
 ['SEA', 'Seattle', 'Seahawks', 'Seattle Seahawks'],
 ['SF', 'San Francisco', '49ers', 'San Francisco 49ers', 'S.F.', 'SFO'],
 ['STL', 'St. Louis', 'Rams', 'St. Louis Rams', 'S.T.L.'],
 ['TB', 'Tampa Bay', 'Buccaneers', 'Tampa Bay Buccaneers', 'T.B.', 'TAM'],
 ['TEN', 'Tennessee', 'Titans', 'Tennessee Titans'],
 ['WAS', 'Washington', 'Redskins', 'Washington Redskins', 'WSH']]

teams_dict = {x[3]:x[0] for x in teams}


pass_rush_attempts_by_team = df.groupby(['Team','Season']).agg(sum)[['Rec_Win_Success_Rate','Rush_Win_Success_Rate']]
pass_rush_attempts_by_team['PassRushRatio'] = pass_rush_attempts_by_team.apply(lambda x: (x.Rec_Win_Success_Rate * 1.0) / x.Rush_Win_Success_Rate, axis=1)

sns.set_palette('muted')
plot_df = pass_rush_attempts_by_team
plot_teams = teams_dict


def plotPassRushByTeam(team_focus_1, team_focus_2):
    fig,ax = plt.subplots(1,1,figsize=(15,8))
    for team in plot_teams:
        if (plot_teams[team] != team_focus_1) or (plot_teams[team] != team_focus_1):
            plt.plot(plot_df.loc[plot_teams[team]]['PassRushRatio'], color='0.91')
    plt.plot(plot_df.loc[team_focus_1]['PassRushRatio'], color='Blue', axes=ax)
    plt.plot(plot_df.loc[team_focus_2]['PassRushRatio'], color='Red', axes=ax)
    return fig


def fig7():
    sns.set_style('white')
    return plotPassRushByTeam(team_focus_1 = 'DAL', team_focus_2 = 'NO')
figure_7 = fig7()
c_wpa = (df.groupby(['Season','Team','Player_Name']).
            agg({'EPA_per_Car':'sum',
                'Rush_Total_EPA':'sum',
                 'Rush_Success_Rate':'mean',
                 'Rush_Total_Clutch_EPA':'sum',
                 
                 
                 'Car_per_Drive':'mean',
                 
                 'Clutch_EPA_per_Car':'mean',
                 'Rush_Total_WPA':'sum',
                 'Rush_WPA_per_Drive':'mean',
                 'Rush_Clutch_EPA_per_Drive':'mean',
                'Carries':'sum',
                 'Rush_TDs':'sum',
                 'Total_Rushing_Yards':'sum',
                 
                 
                 'Yards_per_Car':'mean',
                 'Rushing_Yards_per_Drive':'mean',
                 'Rushing_Fumbles':'mean',
                 
                 
                 
                 
                 'TD_per_Car':'mean',
                 'Fumbles_per_Car':'mean',
                 'TD_Drive':'sum',
                 'EPA_per_Drive':'mean',
                 
                 'Rush_Win_Success_Rate':'mean',
                 'WPA_per_Car':'mean',
                 'WPA_Ratio':'mean',             
 
     }).
 reset_index().
 rename(columns = {
                   'Car_per_Drive':'% Rush Att'                   
                  })
)


c_wpa['workhorse'] = np.where(c_wpa['Carries'] < 100, 'backup',
                          (np.where(c_wpa['% Rush Att'] > 2.2, 'workhorse',
                                    (np.where(c_wpa['% Rush Att'] > 1.7 , 'time_share','backup')))))


c_wpa[(c_wpa['Season']==2017) & (c_wpa['workhorse'] != 'backup')].sort_values(['EPA_per_Car',
                                                                               '% Rush Att',
                                                                               'Rush_Success_Rate',
                                                                               'Rush_Total_Clutch_EPA',  
                                                                               'Rush_Total_EPA'
                                                                                            
                                                                               ], ascending=False).reset_index()
c_wpa.isnull().any()
c_wpa.describe()
c_wpa.hist(bins=50, figsize=(20,15))
plt.show()
potentialFeatures=['EPA_per_Car',
 '% Rush Att',
 'Rush_Success_Rate',
 'Clutch_EPA_per_Car',
 'Rush_Total_WPA',
 'Rush_WPA_per_Drive',
 'Rush_Clutch_EPA_per_Drive',
 'Carries',
 'Total_Rushing_Yards',
 'Yards_per_Car',
 'Rushing_Yards_per_Drive',
 'Rushing_Fumbles',
 'Rush_TDs',
 'Rush_Total_EPA',
 'TD_per_Car',
 'Fumbles_per_Car',
 'TD_Drive',
 'EPA_per_Drive',
 'Rush_Win_Success_Rate',
 'WPA_per_Car',
 'WPA_Ratio']
from pandas.plotting import scatter_matrix
scatter_matrix(c_wpa[potentialFeatures], figsize=(12, 8))
plt.show()
for f in potentialFeatures:
    related_1 = c_wpa['Rush_Total_Clutch_EPA'].corr(c_wpa[f])
    related_2 = c_wpa['EPA_per_Car'].corr(c_wpa[f])
    print("%s: %f %f" % (f, related_1, related_2))
print("Rush Total Clutch EPA Overall Corrolation: %f" % (related_1))
print("EPA per Car: %f" % (related_2))
related_1, related_2
list(c_wpa.columns)
corr_matrix = c_wpa[(c_wpa['workhorse'] != 'backup')].corr()
corr_matrix['EPA_per_Car'].sort_values(ascending=False)
cols = [
 'Rush_Total_EPA',
 'Rush_Success_Rate',
 'Rush_Total_Clutch_EPA',
 '% Rush Att',
 'Clutch_EPA_per_Car',
 'Rush_Total_WPA',
 'Rush_WPA_per_Drive',
 'Rush_Clutch_EPA_per_Drive',
 'Carries',
 'Rush_TDs',
 'Total_Rushing_Yards',
 'Yards_per_Car',
 'Rushing_Yards_per_Drive',
 'Rushing_Fumbles',
 'TD_per_Car',
 'Fumbles_per_Car',
 'TD_Drive',
 'EPA_per_Drive',
 'Rush_Win_Success_Rate',
 'WPA_per_Car',
 'WPA_Ratio']
# create a list containing Pearson's correlation between 'EPA_per_Car' with each column in cols

correlations = [c_wpa['EPA_per_Car'].corr(c_wpa[f]) for f in cols]
len(cols), len(correlations)
def plot_dataframe(df, y_label):
    color='coral'
    fig = plt.gcf()
    fig.set_size_inches(20, 12)
    plt.ylabel(y_label)
    
    ax = df.correlation.plot(linewidth=3.3, color=color)
    ax.set_xticks(df.index)
    ax.set_xticklabels(df.attributes, rotation=75); # Notice the ; (remove it and see what happns!)
    plt.show()
# create a dataframe using cols and correlations

df2 = pd.DataFrame({'attributes':cols, 'correlation':correlations})
# Plot the plot_dataframe using the function
plot_dataframe(df2, 'Player\'s Overall Rating')
# Define the features you want to use for grouping players
select5features= ['EPA_per_Drive', 'Rush_Success_Rate', 'Rushing_Yards_per_Drive', '% Rush Att','TD_per_Car']
select5features
# generate a new dataframe by selecting the features you just defined

df_select = c_wpa[select5features].copy(deep=True)
df_select.head()
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

# Perform scaling on the dataframe containing the features
data = scale(df_select)

# Define number of clusters
cluster_num = 4

# Train model
model = KMeans(init='k-means++', n_clusters=cluster_num, n_init=20).fit(data)
print(90*'_')
print("\nCount of Running Backs in each cluster")
print(90*'_')

pd.value_counts(model.labels_, sort=False)
def pd_centers(featuresUsed, centers):
    from itertools import cycle, islice
    from pandas.tools.plotting import parallel_coordinates
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    colNames = list(featuresUsed)
    colNames.append('prediction')
    # Zip with a column called 'prediction' (index)
    Z = [np.append(A, index) for index, A in enumerate(centers)]
    
    # Convert to pandas for plotting
    P = pd.DataFrame(Z, columns=colNames)
    P['prediction'] = P['prediction'].astype(int)
    return P

def parallel_plot(data):
    from itertools import cycle, islice
    from pandas.tools.plotting import parallel_coordinates
    import matplotlib.pyplot as plt
    
    my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(data)))
    plt.figure(figsize=(15,8)).gca().axes.set_ylim([-2.5,+2.5])
    parallel_coordinates(data, 'prediction', color = my_colors, marker='o')
# Create a composite dataframe for plotting
# .... Use custom function declared in customplot.py (see requirements section, should be imported in your notebook)
P = pd_centers(featuresUsed=select5features, centers=model.cluster_centers_)
P
# use matplotlib for graphical functions inside the notebook

%matplotlib inline

parallel_plot(P)
