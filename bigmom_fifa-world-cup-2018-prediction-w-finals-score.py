import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

rankings = pd.read_csv('../input/fifa-international-soccer-mens-ranking-1993now/fifa_ranking.csv')
rankings = rankings.loc[:,['rank', 'country_full', 'country_abrv', 'cur_year_avg_weighted', 'rank_date', 
                           'two_year_ago_weighted', 'three_year_ago_weighted']]
rankings = rankings.replace({"IR Iran": "Iran"})
rankings['weighted_points'] =  rankings['cur_year_avg_weighted'] + rankings['two_year_ago_weighted'] + rankings['three_year_ago_weighted']
rankings['rank_date'] = pd.to_datetime(rankings['rank_date'])

matches = pd.read_csv('../input/international-football-results-from-1872-to-2017/results.csv')
matches =  matches.replace({'Germany DR': 'Germany', 'China': 'China PR'})
matches['date'] = pd.to_datetime(matches['date'])

world_cup = pd.read_csv('../input/fifa-worldcup-2018-dataset/World Cup 2018 Dataset.csv')
world_cup = world_cup.loc[:, ['Team', 'Group', 'First match \nagainst', 'Second match\n against', 'Third match\n against']]
world_cup = world_cup.dropna(how='all')
world_cup = world_cup.replace({"IRAN": "Iran", 
                               "Costarica": "Costa Rica", 
                               "Porugal": "Portugal", 
                               "Columbia": "Colombia", 
                               "Korea" : "Korea Republic"})
world_cup = world_cup.set_index('Team')
# I want to have the ranks for every day 
rankings = rankings.set_index(['rank_date'])\
            .groupby(['country_full'], group_keys=False)\
            .resample('D').first()\
            .fillna(method='ffill')\
            .reset_index()

# join the ranks
matches = matches.merge(rankings, 
                        left_on=['date', 'home_team'], 
                        right_on=['rank_date', 'country_full'])
matches = matches.merge(rankings, 
                        left_on=['date', 'away_team'], 
                        right_on=['rank_date', 'country_full'], 
                        suffixes=('_home', '_away'))
# feature generation
matches['rank_difference'] = matches['rank_home'] - matches['rank_away']
matches['point_difference'] = matches['weighted_points_home'] - matches['weighted_points_away']
matches['score_difference'] = matches['home_score'] - matches['away_score']
matches['is_stake'] = matches['tournament'] != 'Friendly'
matches['is_won'] = matches['score_difference'] > 0 # take draw as lost

# I tried earlier rest days but it did not turn to be useful
max_rest = 30
matches['rest_days'] = matches.groupby('home_team').diff()['date'].dt.days.clip(0,max_rest).fillna(max_rest)

# I tried earlier the team as well but that did not make a difference either
matches['wc_participant'] = matches['home_team'] * matches['home_team'].isin(world_cup.index.tolist())
matches['wc_participant'] = matches['wc_participant'].replace({'':'Other'})
matches = matches.join(pd.get_dummies(matches['wc_participant']))
from sklearn import linear_model
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

X, y = matches.loc[:,['rank_home', 'rank_difference', 'point_difference', 'is_stake']], matches['is_won']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

logreg = linear_model.LogisticRegression(C=1e-5)
features = PolynomialFeatures(degree=2)
model = Pipeline([
    ('polynomial_features', features),
    ('logistic_regression', logreg)
])
model = model.fit(X_train, y_train)

# figures 
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
plt.figure(figsize=(15,5))
ax = plt.subplot(1,3,1)
ax.plot([0, 1], [0, 1], 'k--')
ax.plot(fpr, tpr)
ax.set_title('AUC score is {0:0.2}'.format(roc_auc_score(y_test, model.predict_proba(X_test)[:,1])))
ax.set_aspect(1)

ax = plt.subplot(1,3,2)
cm = confusion_matrix(y_test, model.predict(X_test))
ax.imshow(cm, cmap='Blues', clim = (0, cm.max())) 

ax.set_xlabel('Predicted label')
ax.set_title('Performance on the Test set')

ax = plt.subplot(1,3,3)
cm = confusion_matrix(y_train, model.predict(X_train))
ax.imshow(cm, cmap='Blues', clim = (0, cm.max())) 
ax.set_xlabel('Predicted label')
ax.set_title('Performance on the Training set')
pass
# let's define a small margin when we safer to predict draw then win
margin = 0.05

# let's define the rankings at the time of the World Cup
world_cup_rankings = rankings.loc[(rankings['rank_date'] == rankings['rank_date'].max()) & 
                                    rankings['country_full'].isin(world_cup.index.unique())]
world_cup_rankings = world_cup_rankings.set_index(['country_full'])
from itertools import combinations

world_cup['points'] = 0
world_cup['total_prob'] = 0

for group in set(world_cup['Group']):
    print('___Starting group {}:___'.format(group))
    for home, away in combinations(world_cup.query('Group == "{}"'.format(group)).index, 2):
        print("{} vs. {}: ".format(home, away), end='')
        row = pd.DataFrame(np.array([[np.nan, np.nan, np.nan, True]]), columns=X_test.columns)
        # entering the rank_home value using loc to search ranking
        row['rank_home'] = world_cup_rankings.loc[home, 'rank']
        home_points = world_cup_rankings.loc[home, 'weighted_points']
        opp_rank = world_cup_rankings.loc[away, 'rank']
        opp_points = world_cup_rankings.loc[away, 'weighted_points']
        # Computing difference in ranking
        row['rank_difference'] = row['rank_home'] - opp_rank
        row['point_difference'] = home_points - opp_points
        
        home_win_prob = model.predict_proba(row)[:,1][0]
        world_cup.loc[home, 'total_prob'] += home_win_prob
        world_cup.loc[away, 'total_prob'] += 1-home_win_prob
        
        
        points = 0
        if home_win_prob <= 0.5 - margin:
            print("{} wins with {:.2f}".format(away, 1-home_win_prob))
            # Establishing the 'points' column here
            world_cup.loc[away, 'points'] += 3
        if home_win_prob > 0.5 - margin:
            points = 1
        if home_win_prob >= 0.5 + margin:
            points = 3
            world_cup.loc[home, 'points'] += 3
            print("{} wins with {:.2f}".format(home, home_win_prob))
        if points == 1:
            print("Draw")
            world_cup.loc[home, 'points'] += 1
            world_cup.loc[away, 'points'] += 1
pairing = [0,3,4,7,8,11,12,15,1,2,5,6,9,10,13,14]

labels = []
odds = []

world_cup = world_cup.sort_values(by=['Group', 'points', 'total_prob'], ascending=False).reset_index()
next_round_wc = world_cup.groupby('Group').nth([0, 1]) # select the top 2
next_round_wc = next_round_wc.reset_index()
next_round_wc = next_round_wc.loc[pairing]
next_round_wc = next_round_wc.set_index('Team')

finals = ['round_of_16', 'quarterfinal', 'semifinal', 'final']

for f in finals:
    print("___Starting of the {}___".format(f))
    iterations = int(len(next_round_wc) / 2)
    winners = []

    for i in range(iterations):
        home = next_round_wc.index[i*2]
        away = next_round_wc.index[i*2+1]
        print("{} vs. {}: ".format(home,
                                   away), 
                                   end='')
        row = pd.DataFrame(np.array([[np.nan, np.nan, np.nan, True]]), columns=X_test.columns)
        row['rank_home'] = world_cup_rankings.loc[home, 'rank']
        home_points = world_cup_rankings.loc[home, 'weighted_points']
        opp_rank = world_cup_rankings.loc[away, 'rank']
        opp_points = world_cup_rankings.loc[away, 'weighted_points']
        row['rank_difference'] = row['rank_home'] - opp_rank
        row['point_difference'] = home_points - opp_points

        home_win_prob = model.predict_proba(row)[:,1][0]
        if model.predict_proba(row)[:,1] <= 0.5:
            print("{0} wins with probability {1:.2f}".format(away, 1-home_win_prob))
            winners.append(away)
        else:
            print("{0} wins with probability {1:.2f}".format(home, home_win_prob))
            winners.append(home)

        labels.append("{}({:.2f}) vs. {}({:.2f})".format(world_cup_rankings.loc[home, 'country_abrv'], 
                                                        1/home_win_prob, 
                                                        world_cup_rankings.loc[away, 'country_abrv'], 
                                                        1/(1-home_win_prob)))
        odds.append([home_win_prob, 1-home_win_prob])
                
    next_round_wc = next_round_wc.loc[winners]
    print("\n")
row['rank_home'] = world_cup_rankings.loc['Denmark', 'rank']
home_points = world_cup_rankings.loc['Denmark', 'weighted_points']
opp_rank = world_cup_rankings.loc['Portugal', 'rank']
opp_points = world_cup_rankings.loc['Portugal', 'weighted_points']
# Computing difference in ranking
row['rank_difference'] = row['rank_home'] - opp_rank
row['point_difference'] = home_points - opp_points
Denmark_winning_prob = model.predict_proba(row)[:,1][0]
Portugal_winning_prob=1 -Denmark_winning_prob
Portugal_winning_prob
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
df = pd.read_csv('../input/complete-player-info/CompleteDataset.csv')
df.head()
df2 = df.sort_values(['Nationality','Overall'],ascending=False).groupby('Nationality').head(50)
def str2number(amount):
    if amount[-1] == 'M':
        return float(amount[1:-1])*1000000
    elif amount[-1] == 'K':
        return float(amount[1:-1])*1000
    else:
        return float(amount[1:])
    
df2['MaxPotential'] = df2['Potential'] - df2['Overall']
df2['ValueNum'] = df2['Value'].apply(lambda x: str2number(x))
top_teams = df2.groupby("Nationality").mean().sort_values("ValueNum", ascending=False).head(16).reset_index()[["Nationality", "Overall", "ValueNum",'MaxPotential']]
trace1 = go.Bar(
    x = top_teams["Nationality"].tolist(),
    y = top_teams["Overall"].tolist(),
    name='Country Overall',
    marker=dict(
        color='rgba(55, 128, 191, 0.7)',
        line=dict(
            color='rgba(55, 128, 191, 1.0)',
            width=2,
        )
    )
)

trace2 = go.Bar(
    x = top_teams["Nationality"].tolist(),
    y = top_teams["MaxPotential"].tolist(),
    name='Country Potential',
    marker=dict(
        color='rgba(219, 64, 82, 0.7)',
        line=dict(
            color='rgba(219, 64, 82, 1.0)',
            width=2,
        )
    )
)

trace3 = go.Scatter(
    x = top_teams["Nationality"].tolist(),
    y = (top_teams["ValueNum"] / 1000000).tolist(),
    name='Country Value [M€]',
    mode = 'lines+markers',
    yaxis='y2'
)

data = [trace1, trace2,trace3]

layout = go.Layout(
    barmode='stack',
    title = 'Level of talent across countries',
    titlefont=dict(size=25),
    width=850,
    height=500,
    paper_bgcolor='rgb(244, 238, 225)',
    plot_bgcolor='rgb(244, 238, 225)',
    yaxis = dict(
        title= 'Country Overall/Potential',
        anchor = 'x',
        rangemode='tozero'
    ),
    xaxis = dict(title= 'Country Name'),
    yaxis2=dict(
        title='Country Value [M€]',
        titlefont=dict(
            color='rgb(148, 103, 189)'
        ),
        tickfont=dict(
            color='rgb(148, 103, 189)'
        ),
        overlaying='y',
        side='right',
        anchor = 'x',
        rangemode = 'tozero',
        dtick = 200
    ),
    #legend=dict(x=-.1, y=1.2)
    legend=dict(x=0.05, y=0.05)
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
# Countries identified by logreg to make it past group stages.
next_round_wc = world_cup.groupby('Group').nth([0, 1]) 
next_round_wc['Team']
# Overlaps
next_round_wc['Team'].isin(top_teams['Nationality']).agg('sum')
# Extracting latest ranking for the top 4 countries in semifinals
finals = ['Germany','Portugal','Belgium','Denmark']
Lastest_rank_check = rankings.sort_values('rank_date', ascending = False).groupby('country_full').head(1)
Lastest_rank_check[Lastest_rank_check['country_full'].isin(finals)]
# predictive model of away_score
list = ['rank_difference','is_stake','is_won','away_score']
df2 = matches[list]
# Importing the dataset
X2 = df2.iloc[:, 0:3]
y2 = df2.iloc[:, -1]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = 0.2, random_state = 0)

#Logistic regression accuracy score (predicting away) = 0.389
logreg = linear_model.LogisticRegression(C=1e-5)
features = PolynomialFeatures(degree=2)
model = Pipeline([('polynomial_features', features),('logistic_regression', logreg)])
model2 = model.fit(X2_train, y2_train)
score2 = model2.score(X2_test,y2_test)
cm2 = confusion_matrix(y2_test, model2.predict(X2_test))

#Decision tree accuracy score (predicting home) = 0.2335
#from sklearn.tree import DecisionTreeRegressor
#model2 = DecisionTreeRegressor(random_state = 0)
#model2 = model2.fit(X2_train, y2_train)
#score2 = model2.score(X2_test,y2_test)

cm2 = confusion_matrix(y2_test, np.round(model2.predict(X2_test)))

import seaborn as sns
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score2)
plt.title(all_sample_title, size = 15);
# creating dataset with index[0] > Denmark (Home) vs  Portugal (Away) & index[1] > Belgium (Home) vs Germany (Away)
df3 = pd.DataFrame(columns=["rank_difference", "is_stake",'is_won'], data=[[8,True,False],[2,True,False]])
df3
predicted_away_score = model2.predict(df3)
predicted_away_score
list = ['rank_difference','is_stake','is_won','home_score']
df6 = matches[list]

X3 = df6.iloc[:, 0:3]
y3 = df6.iloc[:, -1]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size = 0.2, random_state = 0)

#Logistic regression accuracy score (predicting home) =
logreg = linear_model.LogisticRegression(C=1e-5)
features = PolynomialFeatures(degree=2)
model = Pipeline([('polynomial_features', features),('logistic_regression', logreg)])
model3 = model.fit(X3_train, y3_train)
score3 = model3.score(X3_test,y3_test)
cm3 = confusion_matrix(y3_test, model3.predict(X3_test))

#Decision tree accuracy score (predicting home) = 0.2335
#from sklearn.tree import DecisionTreeRegressor
#model2 = DecisionTreeRegressor(random_state = 0)
#model2 = model2.fit(X2_train, y2_train)
#score2 = model2.score(X2_test,y2_test)

import seaborn as sns
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score3)
plt.title(all_sample_title, size = 15);
df4 = pd.DataFrame(columns=["rank_difference", "is_stake",'is_won'], data=[[8,True,False],[2,True,False]])
predicted_home_score = model3.predict(df4)
predicted_home_score
away_score= pd.DataFrame(predicted_away_score, columns=['away_score']) 
home_score= pd.DataFrame(predicted_home_score, columns=['home_score'])
df7 = pd.DataFrame(columns=["Home","Away","rank_difference", "is_stake",'is_won'], data=[['Denmark', 'Portugal',8,True,False],['Belgium','Germany', 2,True,False]])
result = pd.concat([df7,away_score,home_score], axis=1, sort=False)
result