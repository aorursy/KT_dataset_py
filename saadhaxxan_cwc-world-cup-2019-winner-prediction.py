import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

print(os.listdir("../input/"))
#load data sets

world_cup = pd.read_csv('../input/World Cup 2019 Dataset.csv')

results = pd.read_csv('../input/results.csv')
world_cup
results.head()
df = results[(results['Team_1'] == 'Pakistan') | (results['Team_2'] == 'Pakistan')]

pakistan = df.iloc[:]

pakistan.head()
#narrowing to team patcipating in the world cup

worldcup_teams = ['England', ' South Africa', '', 'West Indies', 

            'Pakistan', 'New Zealand', 'Sri Lanka', 'Afghanistan', 

            'Australia', 'Bangladesh', 'India']

df_teams_1 = results[results['Team_1'].isin(worldcup_teams)]

df_teams_2 = results[results['Team_2'].isin(worldcup_teams)]

df_teams = pd.concat((df_teams_1, df_teams_2))

df_teams.drop_duplicates()

df_teams.count()
#dropping columns that wll not affect match outcomes

df_teams_2010 = df_teams.drop(['date','Margin', 'Ground'], axis=1)

df_teams_2010.head()
#Building the model

#the prediction label: The winning_team column will show "1" Team 1 has won and "2" if the away team has won.



df_teams_2010 = df_teams_2010.reset_index(drop=True)

df_teams_2010.loc[df_teams_2010.Winner == df_teams_2010.Team_1,'winning_team']=1

df_teams_2010.loc[df_teams_2010.Winner == df_teams_2010.Team_2, 'winning_team']=2

df_teams_2010 = df_teams_2010.drop(['winning_team'], axis=1)



df_teams_2010.head()
#convert team-1 and team-2 from categorical variables to continous inputs 

# Get dummy variables

final = pd.get_dummies(df_teams_2010, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'])



# Separate X and y sets

X = final.drop(['Winner'], axis=1)

y = final["Winner"]





# Separate train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
final.head()
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

score = logreg.score(X_train, y_train)

score2 = logreg.score(X_test, y_test)



print("Training set accuracy: ", '%.3f'%(score))

print("Test set accuracy: ", '%.3f'%(score2))
#adding ICC rankings

#the team which is positioned higher on the ICC Ranking will be considered "favourite" for the match

#and therefore, will be positioned under the "Team_1" column



# Loading new datasets

ranking = pd.read_csv('../input/icc_rankings.csv') 

fixtures = pd.read_csv('../input/fixtures.csv')



# List for storing the group stage games

pred_set = []
# Create new columns with ranking position of each team

fixtures.insert(1, 'first_position', fixtures['Team_1'].map(ranking.set_index('Team')['Position']))

fixtures.insert(2, 'second_position', fixtures['Team_2'].map(ranking.set_index('Team')['Position']))



# We only need the group stage games, so we have to slice the dataset

fixtures = fixtures.iloc[:45, :]

fixtures.tail()
# Loop to add teams to new prediction dataset based on the ranking position of each team

for index, row in fixtures.iterrows():

    if row['first_position'] < row['second_position']:

        pred_set.append({'Team_1': row['Team_1'], 'Team_2': row['Team_2'], 'winning_team': None})

    else:

        pred_set.append({'Team_1': row['Team_2'], 'Team_2': row['Team_1'], 'winning_team': None})

        

pred_set = pd.DataFrame(pred_set)

backup_pred_set = pred_set

pred_set.head()
# Get dummy variables and drop winning_team column

pred_set = pd.get_dummies(pred_set, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'])



# Add missing columns compared to the model's training dataset

missing_cols = set(final.columns) - set(pred_set.columns)

for c in missing_cols:

    pred_set[c] = 0

pred_set = pred_set[final.columns]





pred_set = pred_set.drop(['Winner'], axis=1)

pred_set.head()
#group matches 

predictions = logreg.predict(pred_set)

for i in range(fixtures.shape[0]):

    print(backup_pred_set.iloc[i, 1] + " and " + backup_pred_set.iloc[i, 0])

    if predictions[i] == 1:

        print("Winner: " + backup_pred_set.iloc[i, 1])

    

    else:

        print("Winner: " + backup_pred_set.iloc[i, 0])

    print("")
# List of tuples before 

semi = [('New Zealand', 'India'),

            ('England', 'Australia')]
def clean_and_predict(matches, ranking, final, logreg):



    # Initialization of auxiliary list for data cleaning

    positions = []



    # Loop to retrieve each team's position according to ICC ranking

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



        # If position of first team is better then this team will be the 'Team_1' team, and vice-versa

        if positions[i] < positions[i + 1]:

            dict1.update({'Team_1': matches[j][0], 'Team_2': matches[j][1]})

        else:

            dict1.update({'Team_1': matches[j][1], 'Team_2': matches[j][0]})



        # Append updated dictionary to the list, that will later be converted into a DataFrame

        pred_set.append(dict1)

        i += 2

        j += 1

        

        # Convert list into DataFrame

    pred_set = pd.DataFrame(pred_set)

    backup_pred_set = pred_set



    # Get dummy variables and drop winning_team column

    pred_set = pd.get_dummies(pred_set, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'])



    # Add missing columns compared to the model's training dataset

    missing_cols2 = set(final.columns) - set(pred_set.columns)

    for c in missing_cols2:

        pred_set[c] = 0

    pred_set = pred_set[final.columns]



    pred_set = pred_set.drop(['Winner'], axis=1)



    # Predict!

    predictions = logreg.predict(pred_set)

    for i in range(len(pred_set)):

        print(backup_pred_set.iloc[i, 1] + " and " + backup_pred_set.iloc[i, 0])

        if predictions[i] == 1:

            print("Winner: " + backup_pred_set.iloc[i, 1])

        else:

            print("Winner: " + backup_pred_set.iloc[i, 0])

        print("")
clean_and_predict(semi, ranking, final, logreg)
# Finals

finals = [('India', 'England')]
clean_and_predict(finals, ranking, final, logreg)