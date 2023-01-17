
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.ticker as plticker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
results = pd.read_csv('../input/results.csv')
world_cup = pd.read_csv('../input/World Cup 2018 Dataset.csv')
world_cup.head()
results.head()
winner = []
for i in range (len(results['home_team'])):
    if results ['home_score'][i] > results['away_score'][i]:
        winner.append(results['home_team'][i])
    elif results['home_score'][i] < results ['away_score'][i]:
        winner.append(results['away_team'][i])
    else:
        winner.append('Draw')
results['winning_team'] = winner

#adding goal difference column
results['goal_difference'] = np.absolute(results['home_score'] - results['away_score'])

results.head()
teams = [t for t in world_cup['Team']]
teams = teams[:-1]
points_table = pd.DataFrame(teams,columns=['Team'])
points_table['Points'] = 0
groups = [g for g in world_cup['Group']]
groups = groups[:-1]
points_table['Group'] = [g for g in groups]
points_table = points_table.reset_index(drop=True)
points_table['Team'][30] = "Colombia"
points_table['Team'][4] = "Portugal"
points_table['Team'][7] = "Iran"
points_table.head(10)
df_home = results[results['home_team'].isin(teams)]
df_away = results[results['away_team'].isin(teams)]
df_teams = pd.concat((df_home, df_away))
df_teams.drop_duplicates()
df_teams.count()
year = []
for row in df_teams['date']:
    year.append(int(row[:4]))
df_teams['match_year'] = year
df_teams_1930 = df_teams[df_teams.match_year >= 1930]
df_teams_1930 = df_teams.drop(['date', 'home_score', 'away_score', 'tournament', 'city', 'country', 'goal_difference', 'match_year'], axis=1)
df_teams_1930.head()


df_teams_1930 = df_teams_1930.reset_index(drop=True)
df_teams_1930.loc[df_teams_1930.winning_team == df_teams_1930.home_team,'winning_team']=2
df_teams_1930.loc[df_teams_1930.winning_team == 'Draw', 'winning_team']=1
df_teams_1930.loc[df_teams_1930.winning_team == df_teams_1930.away_team, 'winning_team']=0

df_teams_1930.tail()
final = pd.get_dummies(df_teams_1930, prefix=['home_team', 'away_team'], columns=['home_team', 'away_team'])

# Separate X and y sets
X = final.drop(['winning_team'], axis=1)
y = final["winning_team"]
y = y.astype('int')

# Separate train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
score = logreg.score(X_train, y_train)
score2 = logreg.score(X_test, y_test)

print("Training set accuracy: ", '%.3f'%(score))
print("Test set accuracy: ", '%.3f'%(score2))
ranking = pd.read_csv('../input/fifa_rankings.csv')
fixtures = pd.read_csv('../input/fixtures.csv')
fixtures.insert(1, 'first_position', fixtures['Home Team'].map(ranking.set_index('Team')['Position']))
fixtures.insert(2, 'second_position', fixtures['Away Team'].map(ranking.set_index('Team')['Position']))

# We only need the group stage games, so we have to slice the dataset
fixtures = fixtures.iloc[:48, :]
fixtures.head()
pred_set = []
for index, row in fixtures.iterrows():
    if row['first_position'] < row['second_position']:
        pred_set.append({'home_team': row['Home Team'], 'away_team': row['Away Team'], 'winning_team': None})
    else:
        pred_set.append({'home_team': row['Away Team'], 'away_team': row['Home Team'], 'winning_team': None})
        
pred_set = pd.DataFrame(pred_set)
backup_pred_set = pred_set

pred_set.head()
pred_set = pd.get_dummies(pred_set, prefix=['home_team', 'away_team'], columns=['home_team', 'away_team'])
missing_cols = set(final.columns) - set(pred_set.columns)
for c in missing_cols:
    pred_set[c] = 0
pred_set = pred_set[final.columns]

# Remove winning team column
pred_set = pred_set.drop(['winning_team'], axis=1)
predictions = logreg.predict(pred_set)
for i in range(fixtures.shape[0]):
    print(backup_pred_set.iloc[i, 1] + " and " + backup_pred_set.iloc[i, 0])
    if predictions[i] == 2:
        print("Winner: " + backup_pred_set.iloc[i, 1])
        points_table.loc[points_table.Team == backup_pred_set.iloc[i, 1] , 'Points'] += 3
    elif predictions[i] == 1:
        print("Draw")
        points_table.loc[points_table.Team == backup_pred_set.iloc[i, 0] , 'Points'] += 1
        points_table.loc[points_table.Team == backup_pred_set.iloc[i, 1] , 'Points'] += 1
    elif predictions[i] == 0:
        print("Winner: " + backup_pred_set.iloc[i, 0])
        points_table.loc[points_table.Team == backup_pred_set.iloc[i, 0] , 'Points'] += 3
    print('Probability of ' + backup_pred_set.iloc[i, 1] + ' winning: ', '%.3f'%(logreg.predict_proba(pred_set)[i][2]))
    print('Probability of Draw: ', '%.3f'%(logreg.predict_proba(pred_set)[i][1]))
    print('Probability of ' + backup_pred_set.iloc[i, 0] + ' winning: ', '%.3f'%(logreg.predict_proba(pred_set)[i][0]))
    print("")
grouped_df = points_table.groupby('Group')
for key, item in grouped_df:
    print(grouped_df.get_group(key), "\n\n")

    
first_place = []
second_place = []
knockout = {}
group_16 = []
for i in range(8):
    t = list(list(grouped_df)[i][1]['Team'])
    p = list(list(grouped_df)[i][1]['Points'])
    for j in range(len(t)):
        knockout[t[j]] = p[j]
        if(j % 3 == 0 and j!=0):
            p = sorted(knockout.items(), key=lambda x: x[1],reverse=True)
            first_place.append(p[0][0])
            second_place.append(p[1][0])
            knockout = {}
def my_range(start, end, step):
    while start <= end:
        yield start
        start += step
        
for i in my_range(0, len(second_place)-1, 2):
    temp = second_place[i+1]
    second_place[i+1] = second_place[i]
    second_place[i] = temp
group_16 = list(zip(first_place,second_place))

def clean_and_predict(matches, ranking, final, logreg):

    # Initialization of auxiliary list for data cleaning
    positions = []
    group = []
    # Loop to retrieve each team's position according to FIFA ranking
    for match in matches:
        positions.append(ranking.loc[ranking['Team'] == match[0],'Position'].iloc[0])
        positions.append(ranking.loc[ranking['Team'] == match[1],'Position'].iloc[0])
    
    # Creating the DataFrame for prediction
    pred_set = []

    # Initializing iterators for while loop
    i = 0
    j = 0

    # 'i' will be the iterator for the 'positions' list, and 'j' for the list of matches (list of tuples)
    while i < len(positions):
        dict1 = {}

        # If position of first team is better, he will be the 'home' team, and vice-versa
        if positions[i] < positions[i + 1]:
            dict1.update({'home_team': matches[j][0], 'away_team': matches[j][1]})
        else:
            dict1.update({'home_team': matches[j][1], 'away_team': matches[j][0]})

        # Append updated dictionary to the list, that will later be converted into a DataFrame
        pred_set.append(dict1)
        i += 2
        j += 1

    # Convert list into DataFrame
    pred_set = pd.DataFrame(pred_set)
    backup_pred_set = pred_set

    # Get dummy variables and drop winning_team column
    pred_set = pd.get_dummies(pred_set, prefix=['home_team', 'away_team'], columns=['home_team', 'away_team'])

    # Add missing columns compared to the model's training dataset
    missing_cols2 = set(final.columns) - set(pred_set.columns)
    for c in missing_cols2:
        pred_set[c] = 0
    pred_set = pred_set[final.columns]

    # Remove winning team column
    pred_set = pred_set.drop(['winning_team'], axis=1)

    # Predict!
    predictions = logreg.predict(pred_set)
    for i in range(len(pred_set)):
        print(backup_pred_set.iloc[i, 1] + " and " + backup_pred_set.iloc[i, 0])
        if predictions[i] == 2:
            print("Winner: " + backup_pred_set.iloc[i, 1])
            group.append(backup_pred_set.iloc[i, 1])
        elif predictions[i] == 1:
            print("Draw")
        elif predictions[i] == 0:
            print("Winner: " + backup_pred_set.iloc[i, 0])
            group.append(backup_pred_set.iloc[i, 0])
        print('Probability of ' + backup_pred_set.iloc[i, 1] + ' winning: ' , '%.3f'%(logreg.predict_proba(pred_set)[i][2]))
        print('Probability of Draw: ', '%.3f'%(logreg.predict_proba(pred_set)[i][1])) 
        print('Probability of ' + backup_pred_set.iloc[i, 0] + ' winning: ', '%.3f'%(logreg.predict_proba(pred_set)[i][0]))
        print("")
    return group

group_8 = clean_and_predict(group_16, ranking, final, logreg)
group_8
temp1 = []
temp2 = []
for i in range(len(group_8)):
    if(i < len(group_8)/2):
        j = i/2
        if(j>=1):
            temp2.append(group_8[i])
        else:
            temp1.append(group_8[i])
    else:
        j = i/2
        if(j>=2 and j<3):
            temp1.append(group_8[i])
        else:
            temp2.append(group_8[i])

quarters = list(zip(temp1,temp2))
print(quarters)
group_4 = clean_and_predict(quarters, ranking, final, logreg)
a = [group_4[0],group_4[1]]
b  = [group_4[2],group_4[3]]
semi = list(zip(a,b))
print(semi)
group_2 = clean_and_predict(semi, ranking, final, logreg)
finals = [(group_2[0],group_2[1])]
print(finals)
winner = clean_and_predict(finals, ranking, final, logreg)
print("Our Predicted Winner is",winner[0])
