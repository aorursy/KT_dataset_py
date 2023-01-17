# Importing important libraries



import pandas as pd    # for data analysis

import plotly.express as px    # for data visualisation
# fetching required datasets

salaries = pd.read_csv('/kaggle/input/nashville-sc-data-project/salaries.csv')

xgoals = pd.read_csv('/kaggle/input/nashville-sc-data-project/xgoals.csv')

xpass = pd.read_csv('/kaggle/input/nashville-sc-data-project/xpass.csv')
salaries.head()
# selecting players within the guaranteed compensation cap constraint

salaries = salaries[salaries['Guaranteed Compensation'] < 400000] 



# finding the latest salary details for each player

salaries = salaries.sort_values('Date').groupby('Player').last()



# cleaning the dataset

salaries.drop(['Season', 'Position', 'Date'], axis = 1, inplace = True)



salaries.rename(columns = {'Base Salary': 'Base Salary ($)', 

                           'Guaranteed Compensation': 'Guaranteed Compensation ($)'}, 

                inplace = True)



salaries.reset_index(inplace = True)
salaries.head()
xgoals.head()
# cleaning the dataset



xgoals.drop(['Season', 

             'Shots', 

             'SoTShots on Target', 

             'GGoals', 

             'xGxGoals', 

             'xPlaceDifference between post- and pre-shot xG models.', 

             'G-xG', 

             'PAExpected points added through scoring goals.', 

             'xPAExpected points added through taking shots.'],

            axis = 1, inplace = True)



xgoals.rename(columns = {'MinutesIncludes stoppage time.': 'Minutes Played', 

                         'KeyPKey Passes': 'Key Passes per 96', 

                         'APrimary Assists': 'A (Assists) per 96', 

                         'xAxAssists': 'xA per 96', 

                         'A-xA': 'A-xA per 96', 

                         'xG+xA': 'xG+xA per 96'},

              inplace = True)
# analysing important statistics of the data

xgoals.describe()
# setting minimum values for filtering the data



min_minutes_played = 1500 # roughly equal to the mean Minutes Played value for the xGoals dataset

min_keyp_p96 = 0.9 # roughly equal to the mean Key Passes per 96 value for the xGoals dataset
# preparing the dataset according to the set minimum values and filters



xgoals = xgoals[xgoals['Team'] != 'HOU']



xgoals = xgoals[(xgoals['Minutes Played'] >= min_minutes_played) 

                & (xgoals['Key Passes per 96'] >= min_keyp_p96) 

                & (xgoals['A-xA per 96'] >= 0)]
xgoals.head()
xpass.head()
# cleaning the dataset



xpass.drop(['Season', 

            'DistanceAverage distance of completed passes, measured in yards. Assumes 115x80 field dimensions.'], 

           axis = 1, inplace = True)



xpass.rename(columns = {'MinutesIncludes stoppage time.': 'Minutes Played', 

                        'Passes': 'Passes per 96', 

                        'Pass %Pass Completion': 'Pass %', 

                        'xPass %Expected pass completion percentage.': 'xPass %', 

                        'ScoreNumber of passes completed over/under expected.': 'Score per 96', 

                        'Per100Passes completed over/under expected, measured per 100 passes.': 'Per100', 

                        'VerticalAverage vertical distance of completed passes, measured in yards. Assumes 115x80 field dimensions.': 'Vertical', 

                        "Touch %Players' share of their team's number of touches.": 'Touch %'}, 

             inplace = True)



xpass['Pass %'] = xpass['Pass %']*100

xpass['xPass %'] = xpass['xPass %']*100

xpass['Touch %'] = xpass['Touch %']*100
# analysing important statistics of the data

xpass.describe()
# setting minimum values for filtering the data



min_minutes_played = 1500 # roughly equal to the mean Minutes Played value for the xPass dataset

min_passes_p96 = 30 # roughly equal to the mean Passes per 96 value for the xPass dataset
# preparing the dataset according to the set minimum values and filters



xpass = xpass[xpass['Team'] != 'HOU']



xpass = xpass[(xpass['Minutes Played'] >= min_minutes_played) 

              & (xpass['Passes per 96'] >= min_passes_p96) 

              & (xpass['Score per 96'] >= 0) 

              & (xpass['Per100'] >= 0) 

              & (xpass['Vertical'] > 0)]
xpass.head()
# performing inner join on the three datasets to find players common to each of these datasets

targets = pd.merge(left = xpass, right = xgoals, how = 'inner', on = 'Player')

targets = pd.merge(left = targets, right = salaries, how = 'inner', on = 'Player')



# cleaning the merged dataset

targets.drop(['Team_y', 

              'Minutes Played_y', 

              'Team', 

              'Score per 96', 

              'Per100', 

              'Vertical', 

              'A-xA per 96'], 

             axis = 1, inplace = True)



targets.rename(columns = {'Team_x': 'Team', 

                          'Minutes Played_x': 'Minutes Played'}, 

               inplace = True)
targets
targets.style.background_gradient(cmap = 'Blues')
fig = px.scatter(targets, 

                 x = 'Guaranteed Compensation ($)', 

                 y = 'Passes per 96', 

                 size = 'Touch %', 

                 hover_name = 'Player', 

                 size_max = 40, 

                 hover_data = ['Minutes Played', 'Pass %', 'xPass %'], 

                 title = "Targets' xPass Stats", 

                 color = 'Player')

fig.show()
fig = px.bar(targets, 

             x = 'Player', 

             y = 'Key Passes per 96',

             color = 'Player',

             hover_name = 'Player',

             hover_data = ['A (Assists) per 96', 'xA per 96', 'xG+xA per 96'],

             title = "Targets' xGoals Stats")

fig.show()