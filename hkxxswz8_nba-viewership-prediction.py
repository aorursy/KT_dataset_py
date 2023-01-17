import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import warnings
import seaborn as sns 
from datetime import datetime
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings('ignore')
game_data=pd.read_csv('../input/game_data.csv')
player_data=pd.read_csv('../input/player_data.csv')
test_set = pd.read_csv('../input/test_set.csv')
training_set = pd.read_csv('../input/training_set.csv')
salary_1617 = pd.read_csv('../input/salary_1617.csv')
salary_1718 = pd.read_csv('../input/salary_1718.csv')
player_data[player_data.isnull().any(axis=1)].head()
game_data[game_data.isnull().any(axis=1)].head()
training_set[training_set.isnull().any(axis=1)].head()

print (len(test_set[test_set['Season'] == '2016-17']),'games are from Season 2016-17.')
print (len(test_set[test_set['Season'] == '2017-18']),'games are from Season 2017-18.')
def get_home_team_id (game_id):
    return training_set[training_set['Game_ID']==game_id]['Home_Team'].unique()
def get_away_team_id (game_id):
    return training_set[training_set['Game_ID']==game_id]['Away_Team'].unique()
def get_season(game_id):
    return training_set[training_set['Game_ID']==game_id]['Season'].unique()

new_df = training_set.groupby('Game_ID').agg('sum').reset_index()

new_df['Home_Team']= new_df['Game_ID'].map(lambda x: get_home_team_id(x)[0])
new_df['Away_Team']= new_df['Game_ID'].map(lambda x: get_away_team_id(x)[0])
new_df['Season']= new_df['Game_ID'].map(lambda x: get_season(x)[0])

Team_Ranking_1617 = ['GSW','SAS','HOU','BOS','CLE','LAC','TOR','UTA','WAS','OKC','ATL',
               'MEM','MIL','IND','POR','CHI','MIA','DEN','DET','CHA','NOP',
               'DAL','SAC','NYK','MIN','ORL','PHI','LAL','PHX','BKN'] 

Team_Ranking_1516 = ['GSW','SAS','CLE','TOR','OKC','LAC','MIA','ATL','BOS','CHA',
               'IND','POR','DET','CHI','DAL','MEM','WAS','HOU','UTA','ORL',
               'MIL','SAC','DEN','NYK','NOP','MIN','PHX','BKN','LAL','PHI'] 

new_df.head()
all_team_viewers = new_df[new_df['Season']=='2016-17'].pivot_table(index='Away_Team',columns='Home_Team',values='Rounded Viewers')
all_team_viewers = all_team_viewers.reindex(Team_Ranking_1516)
all_team_viewers = all_team_viewers[Team_Ranking_1516]
fig = plt.figure(1, figsize=(13, 12))
ax = fig.add_subplot(111)
sns.heatmap(all_team_viewers)
all_team_viewers = new_df[new_df['Season']=='2017-18'].pivot_table(index='Away_Team',columns='Home_Team',values='Rounded Viewers')
all_team_viewers = all_team_viewers.reindex(Team_Ranking_1617)
all_team_viewers = all_team_viewers[Team_Ranking_1617]
fig = plt.figure(1, figsize=(13, 12))
ax = fig.add_subplot(111)
sns.heatmap(all_team_viewers)
Price_1617 = {'GSW':215, 'NYK':177,'LAL':139,'CLE':114,'SAS':105,'CHI':95,
             'TOR':95,'DEN':93,'BOS':90,'BKN':89,'HOU':88,'OKC':85,'SAC':78,
             'PHX':75,'DAL':74,'MIN':73,'POR':70,'PHI':70,'WAS':65,'ATL':57,
             'MIA':55,'UTA':55,'MIL':52,'LAC':50,'ORL':49,'CHA':45,'DET':44,
             'MEM':41,'IND':38,'NOP':35}

view_price_1617 =new_df[new_df['Season']=='2016-17'].groupby('Home_Team').agg('sum')
view_price_1617['Price'] = view_price_1617.index.map(lambda x: Price_1617.get(x))

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111)
plt.scatter(view_price_1617['Price'],view_price_1617['Rounded Viewers'], s=70,c ='red', edgecolors='black', alpha=0.7)
plt.xlabel('Price', fontsize=14)
plt.ylabel('Total Viewership',fontsize=14)
plt.title('Home Team Ticket Price vs. Total Viewership in 2016-17 Season',fontsize=18)

def get_date(game_id):
    return training_set[training_set['Game_ID'] == game_id]['Game_Date'].unique()[0]
lead_change_view = game_data.loc[:,['Game_ID','Lead_Changes']].dropna().groupby('Game_ID').agg('sum')
lead_change_view['Rounded Viewers'] = lead_change_view.index.map(lambda x: new_df[new_df['Game_ID']==x] \
                                        ['Rounded Viewers'].unique()[0])


fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111)
plt.hist(lead_change_view['Lead_Changes'],bins=range(0,35,1),edgecolor='black')
plt.xlabel('Lead Changes', fontsize=14)
plt.ylabel('Frequency',fontsize=14)
plt.title('Histogram of Numbers of Lead Changes',fontsize=18)
fig = plt.figure(figsize=(16,11))
ax = fig.add_subplot(111)
plt.scatter(lead_change_view['Lead_Changes'],lead_change_view['Rounded Viewers'], s=50,c ='blue', edgecolors='black', alpha=0.7)
plt.xlabel('Lead Changes', fontsize=14)
plt.ylabel('Total Viewership',fontsize=14)
plt.title('Scatterplot of Lead Changes vs. Total  Viewership',fontsize=18)

score_diff = game_data.dropna().loc[:,['Game_ID','Location','Final_Score']]
# convert the score of one team to minus 
for i in score_diff.index:
    if score_diff.loc[i,'Location'] == 'A':
        score_diff.loc[i,'Final_Score'] = score_diff.loc[i,'Final_Score']*-1

# calculate the score difference

score_diff = score_diff.groupby('Game_ID').agg('sum')
score_diff['Final_Score'] = score_diff['Final_Score'].map(lambda x: abs(x))
score_diff['Rounded Viewers'] = score_diff.index.map(lambda x: new_df[new_df['Game_ID']==x] \
                                        ['Rounded Viewers'].unique()[0])

fig = plt.figure(figsize=(16,11))
ax = fig.add_subplot(111)
plt.scatter(score_diff['Final_Score'],score_diff['Rounded Viewers'], s=50,c ='orange', edgecolors='black', alpha=0.7)
plt.xlabel('Score Differences', fontsize=14)
plt.ylabel('Total Viewership',fontsize=14)
plt.title('Scatterplot of Score Differences vs. Total  Viewership',fontsize=18)
from datetime import datetime
game_data_viewers = game_data.merge(new_df, left_on = 'Game_ID', right_on = 'Game_ID',how='inner').dropna()
game_data_viewers = game_data_viewers[game_data_viewers['Location']=='H']
game_data_viewers['Game_ID'] = game_data_viewers['Game_ID'].astype('category')
game_data_viewers['Game_Date'] = pd.to_datetime(game_data_viewers['Game_Date'])
game_data_viewers['Weekday'] = game_data_viewers['Game_Date'].map(lambda x: x.isoweekday())
game_data_viewers['Month'] = game_data_viewers['Game_Date'].map(lambda x:x.month)
# average number of viewers of each game by date 
game_timeline_viewers = game_data_viewers.groupby('Game_Date').agg('mean')['Rounded Viewers'].reset_index()
game_timeline_viewers['Weekday'] = game_timeline_viewers['Game_Date'].map(lambda x: x.isoweekday())

# Visualize by different seasons 
game_timeline1617_viewers = game_timeline_viewers[(game_timeline_viewers['Game_Date'] < '2017-05-01') & 
                                                 (game_timeline_viewers['Game_Date'] > '2016-10-01')]
game_timeline1718_viewers = game_timeline_viewers[(game_timeline_viewers['Game_Date'] > '2017-10-01') & 
                                                 (game_timeline_viewers['Game_Date'] < '2018-05-01')]

# fist line:
fig = plt.figure(figsize=(16,11))

plt.subplot(211)
plt.plot(game_timeline1617_viewers['Game_Date'],game_timeline1617_viewers['Rounded Viewers'] ,marker='o',
         alpha=0.7)
plt.title("2016-17 Season")

 
# second line
plt.subplot(212)
plt.plot(game_timeline1718_viewers['Game_Date'],game_timeline1718_viewers['Rounded Viewers'], marker='o',
         color='red',alpha=0.7)
plt.title("2017-18 Season")


import plotly
import plotly.plotly as py
import plotly.graph_objs as go


trace = go.Scatter(
    x =game_timeline1617_viewers['Game_Date'] ,
    y = game_timeline1617_viewers['Rounded Viewers'] ,
    mode = 'lines+markers',
    marker = dict(
    colorscale= 'Weekday',
    color = game_timeline1617_viewers['Weekday'],
    showscale=True)
)


data = [trace]


py.iplot(data, filename='2016-17 season')
trace = go.Scatter(
    x =game_timeline1718_viewers['Game_Date'] ,
    y = game_timeline1718_viewers['Rounded Viewers'] ,
    mode = 'lines+markers',
    marker = dict(
    colorscale= 'Weekday',
    color = game_timeline1718_viewers['Weekday'],
    showscale=True)
)

data = [trace]


py.iplot(data, filename='2017-18 season')
Big_dates = ['10/25/2016','10/27/2016','12/25/2016','10/17/2017','10/19/2017',
            '10/22/2017','11/2/2017','11/16/2017','12/25/2017','1/4/2018','1/23/2018']
big_date_team_list = game_data[game_data['Game_Date'].isin(Big_dates)]['Team'].tolist()

count_team = dict()

for team in big_date_team_list:
    count_team[team] = count_team.get(team,0)+1

count_team
team_strength_1516 = {'GSW':89,'SAS':81.7,'CLE':69.5,'TOR':68.3,'OKC':67.1,'LAC':64.6,
                      'MIA':58.5,'ATL':58.5,'BOS':58.5,'CHA':58.5,
               'IND':54.9,'POR':53.7,'DET':53.7,'CHI':51.2,'DAL':51.2,'MEM':51.2,
                      'WAS':50,'HOU':50,'UTA':48.8,'ORL':42.7,
               'MIL':40.2,'SAC':40.2,'DEN':40.2,'NYK':39.0,'NOP':36.6,'MIN':35.4,
                      'PHX':28,'BKN':25.6,'LAL':20.7,'PHI':12.2}

team_strength_1617 = {'GSW':81.7,'SAS':74.4,'HOU':67.1,'BOS':64.6,'CLE':62.2,'LAC':62.2,
                      'TOR':62.2,'UTA':62.2,'WAS':59.8,'OKC':57.3,'ATL':52.4,
               'MEM':52.4,'MIL':51.2,'IND':51.2,'POR':50,'CHI':50,'MIA':50,'DEN':48.8,
                      'DET':45.1,'CHA':43.9,'NOP':41.5,
               'DAL':40.2,'SAC':39,'NYK':37.8,'MIN':37.8,'ORL':35.4,'PHI':34.1,
                      'LAL':31.7,'PHX':29.3,'BKN':24.4}
player_followers = {'Kevin Love':7,'Manu Ginobili':7.1,'Damian Lillard':7.2,
                   'John Wall':7.4,'Chris Bosh':8.2,'Paul Pierce':9.1,
                   'James Harden':9.4,'Pau Gasol':10.6,'Blake Griffin':12.1,
                   'Jeremy Lin':12.6,'Derrick Rose':13.2,'Kyrie Irving':14.1,
                   'Dwight Howard':15.5,'Russell Westbrook':15.6,'Chris Paul':21.2,
                   'Carmelo Anthony':22,'Kevin Durant':26.6,'Stephen Curry':32.2,
                   'Dwyane Wade':34.1,'LeBron James':86.6}
Price_1617 = {'GSW':215, 'NYK':177,'LAL':139,'CLE':114,'SAS':105,'CHI':95,
             'TOR':95,'DEN':93,'BOS':90,'BKN':89,'HOU':88,'OKC':85,'SAC':78,
             'PHX':75,'DAL':74,'MIN':73,'POR':70,'PHI':70,'WAS':65,'ATL':57,
             'MIA':55,'UTA':55,'MIL':52,'LAC':50,'ORL':49,'CHA':45,'DET':44,
             'MEM':41,'IND':38,'NOP':35}
lm1617 = training_set[training_set['Season']=='2016-17'].groupby('Game_ID').agg('sum')

def get_home_team(gameid):
    return game_data[(game_data['Game_ID']==gameid) & (game_data['Location']=='H')]['Team'].values[0]

def get_away_team(gameid):
    return game_data[(game_data['Game_ID']==gameid) & (game_data['Location']=='A')]['Team'].values[0]

    
# add team price & team strength factors
for i in lm1617.index:
    lm1617.loc[i,'Team_Price'] = Price_1617.get(get_home_team(i)) + Price_1617.get(get_away_team(i))
    lm1617.loc[i,'Team_Strength'] = team_strength_1516.get(get_home_team(i)) + team_strength_1516.get(get_away_team(i))
    
# add social influence factor. If the player is missing from certain game, we won't add the factor into account. 
player_data_wo_nan = player_data.dropna()
social_list = list(player_followers.keys())

def add_home_social_influence(game_id):
    social_score = 0
    team_id = get_home_team(game_id)
    player_data_temp = player_data_wo_nan[(player_data_wo_nan['Game_ID'] == game_id) & (player_data_wo_nan['Team'] == team_id)]
    on_court_player = player_data_temp[player_data_temp['Active_Status'] == 'Active']['Name'].tolist()
    for player in on_court_player:
        if player in social_list:
            social_score = social_score + player_followers.get(player)
    return social_score

def add_away_social_influence(game_id):
    social_score = 0
    team_id = get_away_team(game_id)
    player_data_temp = player_data_wo_nan[(player_data_wo_nan['Game_ID'] == game_id) & (player_data_wo_nan['Team'] == team_id)]
    on_court_player = player_data_temp[player_data_temp['Active_Status'] == 'Active']['Name'].tolist()
    for player in on_court_player:
        if player in social_list:
            social_score = social_score + player_followers.get(player)
    return social_score

for i in lm1617.index:
    lm1617.loc[i,'Player_Influence'] = add_home_social_influence(i) + add_away_social_influence(i)
# add salary factor. If the player is missing from certain game, we won't add the factor into account.
player_salary_dict_1617 = salary_1617.set_index('Name').drop(['Unnamed: 0'],axis=1).to_dict().get('Salary')
player_salary_list = list(player_salary_dict_1617.keys())

def add_home_salary_factor(game_id):
    total_salary= 0
    team_id = get_home_team(game_id)
    player_data_temp = player_data_wo_nan[(player_data_wo_nan['Game_ID'] == game_id) & (player_data_wo_nan['Team'] == team_id)]
    on_court_player = player_data_temp[player_data_temp['Active_Status'] == 'Active']['Name'].tolist()
    for player in on_court_player:
        if player in player_salary_list:
            total_salary = total_salary + player_salary_dict_1617.get(player)
    return total_salary

def add_away_salary_factor(game_id):
    total_salary= 0
    team_id = get_away_team(game_id)
    player_data_temp = player_data_wo_nan[(player_data_wo_nan['Game_ID'] == game_id) & (player_data_wo_nan['Team'] == team_id)]
    on_court_player = player_data_temp[player_data_temp['Active_Status'] == 'Active']['Name'].tolist()
    for player in on_court_player:
        if player in player_salary_list:
            total_salary = total_salary + player_salary_dict_1617.get(player)
    return total_salary


for i in lm1617.index:
    lm1617.loc[i,'Team_Salary'] = add_home_salary_factor(i) + add_away_salary_factor(i)
X_1617 = lm1617.iloc[:,1:]
y_1617 = lm1617['Rounded Viewers']
X_1617.corr()
X_train_1617, X_test_1617, y_train_1617, y_test_1617 = train_test_split(X_1617, y_1617, test_size=0.3,random_state=56)

linear_reg_1617 = LinearRegression()
linear_reg_1617.fit(X_train_1617,y_train_1617)
linear_reg_1617.predict(X_test_1617);
linear_reg_1617.score(X_test_1617,y_test_1617)
poly = PolynomialFeatures(degree=2)
X_1617_ = poly.fit_transform(X_train_1617)
X_1617_test_ = poly.fit_transform(X_test_1617)

lg = LinearRegression()
lg.fit(X_1617_,y_train_1617)
lg.score(X_1617_test_,y_test_1617)
lm1718 = training_set[training_set['Season']=='2017-18'].groupby('Game_ID').agg('sum')

# add team price & team strength factors
# Since we are unable to find price info for 17-18 Season, we will use the ticket price in 16-17 Season. 
for i in lm1718.index:
    lm1718.loc[i,'Team_Price'] = Price_1617.get(get_home_team(i)) + Price_1617.get(get_away_team(i))
    lm1718.loc[i,'Team_Strength'] = team_strength_1617.get(get_home_team(i)) + team_strength_1617.get(get_away_team(i))
    
# add social influence factor. If the player is missing from certain game, we won't add the factor into account. 

for i in lm1718.index:
    lm1718.loc[i,'Player_Influence'] = add_home_social_influence(i) + add_away_social_influence(i)
# add salary factor. If the player is missing from certain game, we won't add the factor into account.
player_salary_dict_1718 = salary_1718.set_index('Name').drop(['Unnamed: 0'],axis=1).to_dict().get('Salary')
player_salary_list = list(player_salary_dict_1718.keys())

def add_home_salary_factor(game_id):
    total_salary= 0
    team_id = get_home_team(game_id)
    player_data_temp = player_data_wo_nan[(player_data_wo_nan['Game_ID'] == game_id) & (player_data_wo_nan['Team'] == team_id)]
    on_court_player = player_data_temp[player_data_temp['Active_Status'] == 'Active']['Name'].tolist()
    for player in on_court_player:
        if player in player_salary_list:
            total_salary = total_salary + player_salary_dict_1718.get(player)
    return total_salary

def add_away_salary_factor(game_id):
    total_salary= 0
    team_id = get_away_team(game_id)
    player_data_temp = player_data_wo_nan[(player_data_wo_nan['Game_ID'] == game_id) & (player_data_wo_nan['Team'] == team_id)]
    on_court_player = player_data_temp[player_data_temp['Active_Status'] == 'Active']['Name'].tolist()
    for player in on_court_player:
        if player in player_salary_list:
            total_salary = total_salary + player_salary_dict_1718.get(player)
    return total_salary


for i in lm1718.index:
    lm1718.loc[i,'Team_Salary'] = add_home_salary_factor(i) + add_away_salary_factor(i)
X_1718 = lm1718.iloc[:,1:]
y_1718 = lm1718['Rounded Viewers']
X_1718.corr()
X_train_1718, X_test_1718, y_train_1718, y_test_1718 = train_test_split(X_1718, y_1718, test_size=0.3,random_state=56)
linear_reg_1718 = LinearRegression()
linear_reg_1718.fit(X_train_1718,y_train_1718)
linear_reg_1718.predict(X_test_1718);
linear_reg_1718.score(X_test_1718,y_test_1718)
poly = PolynomialFeatures(degree=2)
X_1718_ = poly.fit_transform(X_train_1718)
X_1718_test_ = poly.fit_transform(X_test_1718)

lg = LinearRegression()
lg.fit(X_1718_,y_train_1718)
lg.score(X_1718_test_,y_test_1718)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_1617.as_matrix(), i) for i in range(X_1617.shape[1])]
vif["features"] = X_1617.columns
vif
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_1718.as_matrix(), i) for i in range(X_1718.shape[1])]
vif["features"] = X_1718.columns
vif