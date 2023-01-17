import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
goals = pd.read_csv("/kaggle/input/fifa-world-cup-2018-goals/FifaWorldCup2018Goals.csv")

goals.head()
goals.isnull().values.any()
matches = goals[['Home', 'Away', 'Stage']].drop_duplicates()

matches
total_goals_scored_by_team = goals['ScoringTeam'].value_counts()



plt.figure(figsize=(12, 6))

sns.countplot(x='ScoringTeam', data=goals, order=total_goals_scored_by_team.index)

plt.xticks(rotation=-60)

plt.xlabel('Team')

plt.ylabel('Goals scored')

print("Goals scored by every country bar plot:")
home_teams = matches['Home'].value_counts()

away_teams = matches['Away'].value_counts()

# During group stage every team needs to be both home side and away side at least once - we can just simple '+' for addition

total_matches_by_team = home_teams + away_teams



# Denmark and France played the only scoreless match so we will add it manually

total_matches_by_team['France'] += 1

total_matches_by_team['Denmark'] += 1

total_matches_by_team = pd.DataFrame(total_matches_by_team, columns=['MatchPlayed'])

total_matches_by_team.head()
def get_stage(row):

    if row['MatchPlayed'] == 3:

        return 'Group'

    if row['MatchPlayed'] == 4:

        return 'R16'

    if row['MatchPlayed'] == 5:

        return 'QF'

    

    return 'FinalFour'



total_matches_by_team['ReachedStage'] = total_matches_by_team.apply(lambda row: get_stage(row), axis=1)

total_matches_by_team.head()
average_scored_goals_per_match = total_goals_scored_by_team / total_matches_by_team['MatchPlayed']

average_scored_goals_per_match.sort_values(ascending=False).plot(kind='bar', figsize=(12, 6))

plt.xticks(rotation=-60)

plt.xlabel('Team')

plt.ylabel('Average goals scored')

print("Average goals scored per match for each country bar plot:")
goals_conceded_by_home_team = goals[goals['Home'] != goals['ScoringTeam']]['Home'].value_counts()

goals_conceded_by_away_team = goals[goals['Away'] != goals['ScoringTeam']]['Away'].value_counts()

# There are some teams that did not concede a goal as an away team so we need to use add()

total_goals_conceded_by_team = goals_conceded_by_home_team.add(goals_conceded_by_away_team, fill_value=0)

total_goals_conceded_by_team.sort_values(ascending=False).head()
average_goals_conceded_per_match = total_goals_conceded_by_team / total_matches_by_team['MatchPlayed']

average_goals_conceded_per_match.sort_values(ascending=False).plot(kind='bar', figsize=(12, 6))

plt.xticks(rotation=-60)

plt.xlabel('Team')

plt.ylabel('Average goals conceded')

print("Average goals conceded per match for each country bar plot:")
average_goals = pd.concat([average_scored_goals_per_match, average_goals_conceded_per_match], axis=1)

average_goals.columns = ['GoalsScored', 'GoalsConceded']



stage_and_average_goals = average_goals.merge(total_matches_by_team, left_index=True, right_index=True)

stage_and_average_goals.head()
sns.lmplot(x='GoalsScored', y='GoalsConceded', data=stage_and_average_goals,

           fit_reg=True,

           hue='ReachedStage', hue_order=['Group','R16','QF','FinalFour'])



# Mark particular points

# Note: These are not the real average goals values.

# The annotation values are put in order to place the label on a good spot

plt.annotate('Panama', (0.7, 3.45))

plt.annotate('Senegal', (1.3, 1.15))

plt.annotate('Tunisia', (1.55, 2.8))

plt.annotate('Denmark', (0.75, 0.3))

plt.annotate('Russia', (2.1, 1.5))

plt.annotate('Belgium', (2.15, 0.65))
goals_per_stage = goals['Stage'].value_counts()



round16 = goals_per_stage['Round16']

quarterfinals = goals_per_stage['Quarterfinals']

semifinals = goals_per_stage['Semifinals']

third_place = goals_per_stage['ThirdPlace']

final = goals_per_stage['Final']



goals_per_stage = goals_per_stage.drop(labels=['Round16', 'Quarterfinals', 'Semifinals', 'ThirdPlace', 'Final'])

goals_per_stage['Knockout'] = round16 + quarterfinals + semifinals + third_place + final

goals_per_stage.sort_values().plot(kind='barh')

plt.xlabel('Average goals scored')

plt.ylabel('Stage')
sns.countplot(x='Type', data=goals, order=goals['Type'].value_counts().index)
goals[(goals['Type'] == 'Penalty') | (goals['Type'] =='Freekick')]['ScoringTeam'].value_counts().plot(kind='bar')

plt.xticks(rotation=-60)

plt.xlabel('Team')

plt.ylabel('Standard situation goals')
goals[goals['ScoringTeam'] == 'England']['Type'].value_counts().plot(kind='pie')
regular_goals = goals[goals['Type'] != 'Own']

regular_goals.head(10)
regular_goals['Scorer'].value_counts().value_counts().plot(kind='barh')

plt.xlabel('Number of goalscorers')

plt.ylabel('Goals scored')
best_scorers = regular_goals['Scorer'].value_counts()

best_scorers.head(15)
def calculate_goal(row):

    return type_coefficient(row['Type']) * stage_coefficient(row['Stage']) * result_coefficient(row)

    

def type_coefficient(goal_type):

    if goal_type == 'Penalty':

        return 0.75

    if goal_type == 'Inside':

        return 1.0

    if goal_type == 'Outside':

        return 1.5

    if goal_type == 'Freekick':

        return 2.0

    

def stage_coefficient(stage):

    if stage.startswith('Group'):

        return 1.0

    if stage == 'Round16':

        return 1.25

    if stage == 'Quarterfinals':

        return 1.5

    if stage == 'Semifinals' or stage == 'ThirdPlace':

        return 1.75

    if stage == 'Final':

        return 2.0

    

def result_coefficient(row):

    result = row['Result'].split('-')

    

    if result[0] == result[1]:

            # Final equalizer

            if row['Result'] == row['FinalResult']:

                return 1.5

            

            # Just equalizer

            return 1.2

        

    if row['ScoringTeam'] == row['Home']:

        # Goal for home team lead

        if result[0] > result[1]:

            # Winning goal home team

            if row['Result'] == row['FinalResult']:

                return 2

            

            # Just lead

            return 1.5

        

        return 1

    

    # Goal for away team lead

    if result[0] < result[1]:

        # Winning goal away team

        if row['Result'] == row['FinalResult']:

            return 2

            

        # Just lead

        return 1.5

        

    return 1
regular_goals['Value'] = regular_goals.apply(lambda row: calculate_goal(row), axis=1)

regular_goals.head()
final_score = regular_goals[['Scorer', 'Value']].groupby(['Scorer']).sum().reset_index()

final_score.columns = ['Scorer', 'GoalsValue']

final_score.sort_values('GoalsValue', ascending=False).head(10)
real_goals = pd.DataFrame(regular_goals['Scorer'].value_counts())

real_goals.columns = ['RealGoals']



final_results = final_score.merge(real_goals, left_on='Scorer', right_index=True)

final_results
sns.lmplot(x='RealGoals', y='GoalsValue', data=final_results)



# Mark particular points

# Note: These are not the real goals count and result values.

# The annotation values are put in order to place the label on a good spot

plt.annotate('Trippier', (1.0, 5.6))

plt.annotate('Jedinak', (1.7, 1.5))

plt.annotate('Mandzukic', (2.5, 7.7))

plt.annotate('Mbappe', (3.7, 10.6))

plt.annotate('Kane', (5.8, 8.0))