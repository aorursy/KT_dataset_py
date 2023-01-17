# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#This will help us keep track of the submissions

sub_name_list = ['dummy', 'mdabbert', 'eschibli']

score_list = [] #We can keep the scores here



#Put the submissions in dataframe form and add to a list.

sub_list = []

temp_df = pd.read_csv("/kaggle/input/ufc253-contest-dummy-submission/task-dummy.csv")

sub_list.append(temp_df)



temp_df = pd.read_csv("/kaggle/input/ufc-253-prediction-contest-submission/my-submission.csv")

sub_list.append(temp_df)



temp_df = pd.read_csv("/kaggle/input/ufc-253-predictions-09-23/ufc_253_predictions_eschibli_09_23.csv")

sub_list.append(temp_df)

results_df = pd.read_csv("/kaggle/input/ultimate-ufc-dataset/most-recent-event.csv")



#We only need the fighter names, odds, and winner



results_df = results_df[['R_fighter', 'B_fighter', 'R_ev', 'B_ev', 'Winner']]

display(results_df)
#Returns a specific bet EV based on winning_ev and probability.

def get_bet_ev(ev, prob):

    

    return(ev*prob - (1-prob)*100)
#Used to determine the bet of each fight.  We will use probabilities and the ev to 

#determine profitable bets

def get_bet(R_prob, B_prob, R_ev, B_ev):

    red_ev = get_bet_ev(R_ev, R_prob)

    blue_ev = get_bet_ev(B_ev, B_prob)

    if red_ev > 0:

        return('Red')

    if blue_ev > 0:

        return('Blue')

    

    return 'None'
def get_profit(winner, bet, R_ev, B_ev):

    if bet == 'None':

        return 0

    if (bet == 'Blue' and winner == 'Blue'):

        return B_ev

    if (bet == 'Red' and winner == 'Red'):

        return R_ev

    else:

        return (-100)
#Let's make a helper function to make this easier



def get_score(sub, results):

#    display(sub)

#    display(results)

    #Let's merge the two dataframes

    merge_df = pd.merge(sub, results)

    #display(merge_df)

    #We can get the proper bet by using a lambda function

    merge_df['Bet'] = merge_df.apply(lambda x: get_bet(x['R_prob'],x['B_prob'],x['R_ev'],x['B_ev']), axis=1)

    merge_df['Profit'] = merge_df.apply(lambda x: get_profit(x['Winner'], x['Bet'], x['R_ev'], x['B_ev']), axis=1)

    display(merge_df)

    return(sum(merge_df['Profit']))
z = 0

score_list.append(get_score(sub_list[z], results_df))

print(f"{sub_name_list[z]}'s bets saw a total profit of {score_list[z]}")
z = 1

score_list.append(get_score(sub_list[z], results_df))

print(f"{sub_name_list[z]}'s bets saw a total profit of {score_list[z]}")
z = 2

score_list.append(get_score(sub_list[z], results_df))

print(f"{sub_name_list[z]}'s bets saw a total profit of {score_list[z]}")