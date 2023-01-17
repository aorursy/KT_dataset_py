# import packages 

import pandas as pd 

import numpy as np

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
#get the data

bball = pd.read_csv('../input/ncaa_bball_rankings.csv')



#We have the 9 columns as expected, and 353 rows

print(bball.columns)

print(bball.shape)

bball = bball.rename(columns = {'rank': 'real position'})
#change the columns with scores in to be wins - losses and number of games played as separate columns (easier for machine learning)

def split_out_scores(string):

    ## given a string, splits out the values

    ## eg '50-4' gets returned as (54,0.93) as there were 54 games played and overall 93% of games were won 

    wins_losses = string.split('-') #split out into two parts

    wins_losses = list(map(int, wins_losses)) #change from string to integer

    games_played = sum(wins_losses)

    #have to be careful to avoid dividing by zero with win proportion

    if (games_played):

        win_proportion = round(wins_losses[0] / (games_played),2)

    else:

        win_proportion = np.NaN

    return(games_played, win_proportion)

#see it in action

split_out_scores('50-4')



def split_out_scores_column(df, col):

    ## given the pandas df and a column in the df, returns the df with the two new columns

    

    #perform split_out_scores on specified column

    new = pd.DataFrame(bball[col].apply(lambda x: split_out_scores(x)).tolist()) 

    #rename

    new = new.rename(columns = {0: col+'_games_played', 1: col+'_win_proportion'})

    #new = new.rename(columns = {"0": str(col+'_games_played'), "1": str(col+'win_proportion')})

    

    combined = pd.merge(df,new,left_index = True, right_index = True)

    return(combined)



    return(combined)



print(bball.shape)





##apply the splitting on the columns that have it

for col in ['record', 'road', 'neutral', 'home', 'non_div']:

    bball = split_out_scores_column(bball,col)

    

print(bball.shape)

bball.head()

# Create target object and call it y

y = bball['real position']

# Create X





#X = bball.drop(['real position', 'school', 'record', 'road', 'neutral', 'home',  'non_div' ], axis = 1)

features = ['previous', 'record_games_played', 'record_win_proportion']

X = bball[features]



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model

bball_model = DecisionTreeRegressor(random_state=1)

# Fit Model

bball_model.fit(train_X, train_y)



# Make validation predictions and calculate mean absolute error

val_predictions = bball_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))
val_predictions_all = bball_model.predict(bball[features])

#sorted values by there previous rank

bball_model_results = pd.DataFrame({'true_values': bball['real position'],

                                    'predicted_position' : val_predictions_all,

                                    'previous' : bball['previous']})

bball_model_results = bball_model_results.sort_values (by = 'previous')

bball_model_results.head()



val_X.head()
# Using best value for max_leaf_nodes

bball_model = DecisionTreeRegressor(max_leaf_nodes=10, random_state=1)

bball_model.fit(train_X, train_y)

val_predictions = bball_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_table_positions: {:,.0f}".format(val_mae))
#running the model on the whole dataset (including train)

bball_model 
#defining all the different confernces listed 

len(bball['conference'].unique().tolist())

bball.head()

bball_model_results.head()





final_pr = bball.merge(bball_model_results, 

                       left_on = 'previous', right_on = 'previous',

                      how = 'left')

final_pr = final_pr[['real position', 'predicted_position', 'previous','school','record','conference', 'record_win_proportion', 'record_games_played', 'road_games_played', 'road_win_proportion', 'neutral_games_played', 'neutral_win_proportion', 'home_games_played', 'home_win_proportion']]

#add in whether in training data or not

final_pr['training'] = np.where(bball['previous'].isin(train_X['previous']), 'trained','untrained')



final_pr.head(100)
#write the output (not working)

final_pr.to_csv('./final_pr.csv')