# import packages 

import pandas as pd 

import numpy as np



#get the data

bball = pd.read_csv('../input/ncaa_bball_rankings.csv')



#We have the 9 columns as expected, and 353 rows

print(bball.columns)

print(bball.shape)

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
useful_cols = ...