import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
soccer_data = pd.read_csv('../input/fifa-20-complete-player-dataset/players_20.csv') 
soccer_data.head()
EPL_list = ['Arsenal', 'Aston Villa', 'Bournemouth', 'Brighton & Hove Albion', 

            'Burnley', 'Chelsea', 'Crystal Palace','Everton', 'Leicester City', 

            'Liverpool', 'Manchester City', 'Manchester United', 'Newcastle United', 

            'Norwich City', 'Sheffield United', 'Southampton', 'Tottenham Hotspur', 

            'Watford', 'West Ham United', 'Wolverhampton Wanderers']
soccer_data['new'] = soccer_data['club'].apply(lambda x: 1 if x in EPL_list else 0)

EPL_data = soccer_data[soccer_data['new'] == 1]
EPL_data = EPL_data.dropna(axis='columns') # remove NA's

EPL_data = EPL_data[EPL_data['player_positions'] != 'GK'] # remove Goalkeepers

EPL_data = EPL_data.loc[:,~EPL_data.columns.str.contains('^goalkeeping', case=False)] # remove Goalkeepers skills 

EPL_data = EPL_data._get_numeric_data() # remove non-numerical data

EPL_data= EPL_data.drop(columns=['sofifa_id', 'new', 'value_eur', 'team_jersey_number','contract_valid_until', 'overall', 'potential'])
EPL_data.columns
from matplotlib.pyplot import figure



fig, axes = plt.subplots(2, 3, figsize=(20, 10))

# fig, ax = plt.subplots()



EPL_data.loc[:, ['attacking_crossing', 'attacking_finishing', 

                 'attacking_heading_accuracy', 'attacking_short_passing',

                 'attacking_volleys']].plot.hist(bins=12, alpha=0.3, ax=axes[0,0])



EPL_data.loc[:, ['skill_fk_accuracy', 'skill_long_passing', 

                 'skill_ball_control']].plot.hist(bins=12, alpha=0.3, ax=axes[0,1])



EPL_data.loc[:, ['movement_acceleration', 'movement_sprint_speed', 

                 'movement_agility', 'movement_reactions', 

                 'movement_balance']].plot.hist(bins=12, alpha=0.3, ax=axes[0,2])



EPL_data.loc[:, ['power_jumping', 'power_stamina', 'power_strength',

                 'power_long_shots']].plot.hist(bins=12, alpha=0.3, ax=axes[1,0])



EPL_data.loc[:, ['mentality_aggression', 'mentality_interceptions',

                 'mentality_positioning', 'mentality_vision', 'mentality_penalties',

                 'mentality_composure']].plot.hist(bins=12, alpha=0.3, ax=axes[1,1])



EPL_data.loc[:, ['defending_marking', 'defending_standing_tackle',

                 'defending_sliding_tackle']].plot.hist(bins=12, alpha=0.3, ax=axes[1,2])
EPL_data
# Use numpy to convert to arrays

import numpy as np

# Labels are the values we want to predict

labels = np.log(EPL_data['wage_eur'])

# Remove the labels from the features

# axis 1 refers to the columns

features= EPL_data.drop('wage_eur', axis = 1)

# Saving feature names for later use

feature_list = list(features.columns)

# Convert to numpy array

features = np.array(features)
# Using Skicit-learn to split data into training and testing sets

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
# Import the model we are using

from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees

rf = RandomForestRegressor(n_estimators = 1000, random_state = 0)

# Train the model on training data

rf.fit(train_features, train_labels);
# Use the forest's predict method on the test data

predictions = rf.predict(test_features)

# Calculate the absolute errors

errors = abs(predictions - test_labels)

# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
import matplotlib.pyplot as plt

plt.plot(predictions, test_labels, 'o', color='black');