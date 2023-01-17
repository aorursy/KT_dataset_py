import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Import your submission or use sample submission
df_sample_sub = pd.read_csv('../input/SampleSubmissionStage2.csv')
# Load in the teams database
df_teams = pd.read_csv('../input/Teams.csv')



# Define Matchups here
# Make sure to get the names right! (see below for hints of how to get names right)

matchups = {
            1:{'t1':'Oklahoma','t2':'Rhode Island','winner':'t2'},
            2:{'t1':'Tennessee','t2':'Wright St','winner':'t1'},
            3:{'t1':'UNC Greensboro','t2':'Gonzaga','winner':'t2'},
            4:{'t1':'Penn','t2':'Kansas','winner':'t2'},
            5:{'t1':'Iona','t2':'Duke','winner':'t2'},
            6:{'t1':'Loyola-Chicago','t2':'Miami FL','winner':'t1'},
            7:{'t1':'S Dakota St','t2':'Ohio St','winner':'t2'},
            8:{'t1':'Seton Hall','t2':'NC State','winner':'t1'},
            9:{'t1':'Villanova','t2':'Radford','winner':'t1'},
            10:{'t1':'Kentucky','t2':'Davidson','winner':'t1'},
            11:{'t1':'Houston','t2':'San Diego St','winner':'t1'},
            12:{'t1':'Texas Tech','t2':'SF Austin','winner':'t1'},
            13:{'t1':'Alabama','t2':'Virginia Tech','winner':'t1'},
            14:{'t1':'Buffalo','t2':'Arizona','winner':'t1'},
            15:{'t1':'Florida','t2':'St Bonaventure','winner':'t1'},
            16:{'t1':'Montana','t2':'Michigan','winner':'t2'},
           }


def calculate_current_log_loss(matchups):
    count = 0
    cum_log_loss = 0.
    for entry in matchups:
        # Get the correct team IDs
        t1_ID = df_teams[df_teams.TeamName==matchups[entry]['t1']].TeamID.values
        t2_ID = df_teams[df_teams.TeamName==matchups[entry]['t2']].TeamID.values
        # Logic to sort out the game string and the the outcome
        if (t1_ID < t2_ID) & (matchups[entry]['winner']=='t1'):
            outcome = 1
            game_string = '2018_'+str(t1_ID[0])+'_'+str(t2_ID[0])
        elif (t1_ID < t2_ID) & (matchups[entry]['winner']=='t2'):
            outcome = 0
            game_string = '2018_'+str(t1_ID[0])+'_'+str(t2_ID[0])
        elif (t1_ID > t2_ID) & (matchups[entry]['winner']=='t1'):
            outcome = 0
            game_string = '2018_'+str(t2_ID[0])+'_'+str(t1_ID[0])
        elif (t1_ID > t2_ID) & (matchups[entry]['winner']=='t2'):
            outcome = 1
            game_string = '2018_'+str(t2_ID[0])+'_'+str(t1_ID[0])
        else:
            print("Something's gone terribly wrong...")
        #print('game string ',game_string)

        # Get the prediction for the current matchup
        pred = df_sample_sub[df_sample_sub.ID==game_string].Pred.values[0]
        #pred = 0.5
        #print('predicted outcome ',pred)

        # Add to log loss

        cum_log_loss = cum_log_loss + outcome*np.log(pred) + (1.-outcome)*np.log(1.-pred)

        # Increment number of matchups
        count = count + 1
        new_t1 = df_teams[df_teams.TeamID==int(game_string.split('_')[1])].TeamName.values
        new_t2 = df_teams[df_teams.TeamID==int(game_string.split('_')[2])].TeamName.values
        print('\n\n',
              5*' *',
              '\n',
              new_t1,
             '\n\n\t vs\n\n',
              new_t2,
              '\n\n',
              'Prediction: {:.2%}'.format(pred),
              '\n\n',
              'Outcome   : {:.2%}'.format(float(outcome)),
              '\n\n',
              5*' *',
             '\n\n',
             )

    # Normalize Log Loss
    log_loss = cum_log_loss / count
    # Display Results
    print('current log loss: ', -log_loss)
    return -log_loss
# Call function on matchups
calculate_current_log_loss(matchups)
# if you're unsure about names use this
for team in df_teams.TeamName:
    if 'Dakota' in team:
        print(team)
