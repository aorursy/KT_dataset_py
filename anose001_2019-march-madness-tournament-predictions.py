# Import libraries needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
import os
#from subprocess import check_output
#print(check_output(["ls", "../input/mens-machine-learning-competition-2019"]).decode("utf8"))
def load_and_construct_features_in_df(year):
    """
    Input: year: Season year, e.g. 2017
    Output: df: DataFrame with constructed features that can be used for training a ML model.
    """
    # Read in data
    #directory = 'G:/My Drive/Research Work/7 Kaggle/2019 NCAA/mens-machine-learning-competition-2019/DataFiles/'
    directory = os.getcwd().replace('\\','/')
    df_teams_seeds = pd.read_csv('../input/mens-machine-learning-competition-2019/datafiles/NCAATourneySeeds.csv').query('Season =='+str(year))

    # Strip letters in seed column an return an integer
    df_teams_seeds['Seed'] = df_teams_seeds['Seed'].apply(lambda x: int(x[1:3]))
    #print df_teams_seeds.head()

    df_tournament_results = pd.read_csv('../input/mens-machine-learning-competition-2019/datafiles/RegularSeasonDetailedResults.csv').query('Season=='+str(year))
    #print df_tournament_results.head()

    #Add 'WSeed','LSeed' feature to Detailed Tournament list by merging with renamed lists
    # and calculate SeedDifference and add as a feature to df_final
    df_renamed_win = df_teams_seeds.rename(columns={'TeamID':'WTeamID','Seed':'WSeed'})
    df_renamed_loss = df_teams_seeds.rename(columns={'TeamID':'LTeamID','Seed':'LSeed'})
    df_mixed = df_tournament_results.merge(df_renamed_win, how='left', on=['Season','WTeamID'])
    df_final = df_mixed.merge(df_renamed_loss, how='left', on=['Season','LTeamID'])
    
    ### FEATURE CONSTRUCTION STARTS HERE ###
    df_final['SeedDifference'] = df_final['WSeed'] - df_final['LSeed'] 
    df_final['ScoreDifference'] = df_final['WScore'] - df_final['LScore'] 
    #df_final.head()

    # DTOVR (Difference in 'Turnover Ratio'): 100 * TOV / (FGA + (0.44 * FTA) + AST + TOV)
    df_final['WTOVR'] = 100* df_final['WTO']/(df_final['WFGA'] + (0.44*df_final['WFTA']) + df_final['WAst'] + df_final['WTO'])
    df_final['LTOVR'] = 100* df_final['LTO']/(df_final['LFGA'] + (0.44*df_final['LFTA']) + df_final['LAst'] + df_final['LTO'])
    df_final['DTOVR'] = df_final['WTOVR'] - df_final['LTOVR']
    # DeFG% (Difference in 'effective FieldGoal percentage'):  (FGM + 0.5 * 3PM) / FGA.
    df_final['WeFG%'] = (df_final['WFGM'] + 0.5*df_final['WFGM3'])/(df_final['WFGA'] + df_final['WFGA3'])
    df_final['LeFG%'] = (df_final['LFGM'] + 0.5*df_final['LFGM3'])/(df_final['LFGA'] + df_final['LFGA3'])
    df_final['DeFG%'] = df_final['WeFG%'] - df_final['LeFG%']

    #df_final['DeFG%'].describe()
    # DFTAR (Difference in 'FreeThrow Attempt rate'):  FTA / FGA.
    df_final['WFTAR'] = df_final['WFTA']/df_final['WFGA']
    df_final['LFTAR'] = df_final['LFTA']/df_final['LFGA']
    df_final['DFTAR'] = df_final['WFTAR'] - df_final['LFTAR']
    # DOR% (Difference in 'Offensive rebound %'):  WOR / LDR+WOR. Notice that we divide by the 
    # a) number of OffensiveRebounds + number of OpponentsDefensiveRebounds (We look at one side of the board).
    df_final['WOR%'] = df_final['WOR']/(df_final['LDR'] + df_final['WOR'])
    df_final['LOR%'] = df_final['LOR']/(df_final['LOR'] + df_final['WDR'])
    df_final['DOR%'] = df_final['WOR%'] - df_final['LOR%']
    # b) DDR% (Difference in 'Defensive rebound %'):  WDR / LOR+WDR.
    df_final['WDR%'] = df_final['WDR']/(df_final['LOR'] + df_final['WDR'])
    df_final['LDR%'] = df_final['LDR']/(df_final['LDR'] + df_final['WOR'])
    df_final['DDR%'] = df_final['WDR%'] - df_final['LDR%']
    # c) DTDR% (Difference in 'Total Defensive rebound %'):  WDR / LDR+WDR.
    df_final['WTDR%'] = df_final['WDR']/(df_final['LDR'] + df_final['WDR'])
    df_final['LTDR%'] = df_final['LDR']/(df_final['LDR'] + df_final['WDR'])
    df_final['DTDR%'] = df_final['WTDR%'] - df_final['LTDR%']
    # d)
    df_final['WTOR%'] = df_final['WOR']/(df_final['LOR'] + df_final['WOR'])
    df_final['LTOR%'] = df_final['LOR']/(df_final['LOR'] + df_final['WOR'])
    df_final['DTOR%'] = df_final['WTOR%'] - df_final['LTOR%']
    # Adding PIE value to df_final
    df_final['WPIE'] = (df_final['WScore'] + df_final['WFGM'] + df_final['WFTM'] - df_final['WFGA'] 
                        - df_final['WFTA'] + df_final['WDR'] + (0.5*df_final['WOR']) + df_final['WAst'] 
                        + df_final['WStl'] + (0.5*df_final['WBlk']) - df_final['WPF'] - df_final['WTO']
                        )/(
                        df_final['WScore'] + df_final['LScore'] + df_final['WFGM'] + df_final['LFGM'] 
                        + df_final['WFTM'] + df_final['LFTM'] - df_final['WFGA'] - df_final['LFGA'] 
                        + df_final['WDR'] + df_final['LDR'] + (0.5*df_final['WOR'] + 0.5*df_final['LOR']) 
                        + df_final['WAst'] + df_final['LAst'] + df_final['WStl'] + df_final['LStl'] 
                        + (0.5*df_final['WBlk'] + 0.5*df_final['LBlk']) - df_final['WPF'] - df_final['LPF']
                        - df_final['WTO'] - df_final['WTO']
                        )

    df_final['LPIE'] = (df_final['LScore'] + df_final['LFGM'] + df_final['LFTM'] - df_final['LFGA']
                     - df_final['LFTA'] + df_final['LDR'] + (0.5*df_final['LOR']) + df_final['LAst']
                     + df_final['LStl'] + (0.5*df_final['LBlk']) - df_final['LPF'] - df_final['LTO']
                     )/(
                     df_final['WScore'] + df_final['LScore'] + df_final['WFGM'] + df_final['LFGM'] 
                     + df_final['WFTM'] + df_final['LFTM'] - df_final['WFGA'] - df_final['LFGA'] 
                     + df_final['WDR'] + df_final['LDR'] + (0.5*df_final['WOR'] + 0.5*df_final['LOR']) 
                     + df_final['WAst'] + df_final['LAst'] + df_final['WStl'] + df_final['LStl'] 
                     + (0.5*df_final['WBlk'] + 0.5*df_final['LBlk'])  - df_final['WPF'] - df_final['LPF']
                     - df_final['WTO'] - df_final['WTO']
                     )
    df_final['DPIE'] = df_final['WPIE'] - df_final['LPIE']
    return df_final
df_final = load_and_construct_features_in_df(year=2016)
df_final.head()
# Plot Correlations of features
import matplotlib.gridspec as gridspec
features = ['DPIE','DeFG%','DTDR%','DFTAR','DTOVR','DOR%','DDR%']
plt.figure(figsize=(10,10))
gs = gridspec.GridSpec(3, 3,
                       width_ratios=[1,1,1],
                       height_ratios=[1,1,1],
                       hspace=0.32, wspace=0.32)
counter = 0
plt.suptitle('Feature correlation (2016 season)', y=0.92, size= 18)
for feature in features:
    ax = plt.subplot(gs[counter])
    plt.plot(df_final['ScoreDifference'], df_final[feature], marker='.', markerfacecolor = None, markeredgewidth = 0, linestyle='none', zorder= 1, label = feature[1:], alpha = 0.15)
    df_grouped = df_final.groupby(['ScoreDifference']).describe().reset_index()
    plt.plot(df_grouped['ScoreDifference'], df_grouped[feature]['mean'], marker='.', color='C1', linestyle='none', zorder=2, label='')
    if counter in [4,5,6]:
        plt.xlabel('Score Difference', size=13)
    if counter in [0,3,6]:
        plt.ylabel('Feature Difference', size=13)
    plt.legend(fontsize=18)
    plt.tick_params(labelsize=13)
    counter += 1
plt.savefig('FeatureCorrelation.png')
# Extracting training data from the final dataframe of the whole 2016 season
def get_training_data(df_final, columns):
    """
    Input:  df_final: Dataframe with all features that can be used for estimating game results.
            Columns: List of column labels. Column labels must be present in df_final.
    Output: X_train, Y_train datasets, which can be directly fed into the LogReg model
    """
    for feature in columns:
        if feature not in df_final.columns:
            raise Exception('The feature {}'.format(feature)+' is not in the DataFrame. All features must be labels of a column of the dataframe.')
    
    df_wins = pd.DataFrame()
    df_losses = pd.DataFrame()
    for feature in columns:
        df_wins[str(feature)] = df_final[str(feature)]
        df_losses[str(feature)] = -df_final[str(feature)]
    df_wins['Result'] = 1
    df_losses['Result'] = 0

    training_data = pd.concat([df_losses, df_wins], axis=0)

    if len(columns) == 1:
        X_train = training_data[columns].values.reshape(-1,1)
    else:
        X_train = training_data[columns].values
    Y_train = training_data.Result.values
    return (X_train, Y_train)

X_train, Y_train = get_training_data(df_final, columns=['DPIE','DeFG%', 'DOR%','DTDR%','DFTAR','DTOVR'])
corr = np.corrcoef(X_train, rowvar=0)  # correlation matrix
w, v = np.linalg.eig(corr)        # eigen values & eigen vectors
print(w)
#print v
# 'DDR%','DOR%' are collinear. 'DTOR%' and 'DTDR%' are almost collinear.
# 'DPIE','DeFG%', 'DOR%','DTDR%','DFTAR','DTOVR'
# Extracting training data from the final dataframe of the whole 2016 season
X_train, Y_train = get_training_data(df_final, columns=['DOR%', 'DPIE', 'DTOVR', 'DeFG%'])
def train_ML_model(X_train, Y_train, cv, model):
    '''
    Input:  X_train, Y_train: X and Y training data
            cv: Cross-validation integer value
            model: A string of either: 'LinReg','LogReg','DecTree','SVC'.
    Output: GridSearchedObject, ClassifierObject, LogLoss, Regularization Parameter C
    '''
    if model not in ['LinReg','LogReg','DecTree','SVC']:
        raise Exception('Model needs to be LinReg, LogReg, DecTree or SVC')

    X_train, Y_train = shuffle(X_train, Y_train)
    if model == 'LinReg':
        linreg =LinearRegression()
        paramsLinReg = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
        clf = GridSearchCV(linreg, paramsLinReg,cv=cv)
        clf.fit(X_train, Y_train)
        print('Best score: {:.4}'.format(clf.best_score_ ))
        C_dummy = 0
        return clf, linreg, clf.best_score_ , C_dummy
    if model == 'LogReg':
        logreg = LogisticRegression()
        params = {'C': np.logspace(start=-15, stop=15, num=200), 'solver': ['liblinear']} #Parmeter of ML model
        clf = GridSearchCV(logreg, params, scoring='neg_log_loss', refit=True, cv=cv)
        clf.fit(X_train, Y_train)
        print('Best log_loss: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C']))
        return clf, logreg, clf.best_score_, clf.best_params_['C']
    if model == 'SVC':
        SVCreg = SVC(probability=True)
        params = {'C': np.logspace(start=-2, stop=3, num=10), 'kernel':['rbf']}
        clf = GridSearchCV(SVCreg, params, scoring='neg_log_loss', refit=True, cv=cv)
        clf.fit(X_train, Y_train)
        print('Best log_loss: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C']))
        return clf, SVCreg, clf.best_score_, clf.best_params_['C']
    if model == 'DecTree':
        Tree = DecisionTreeClassifier()
        params = {'max_depth': range(1,8),'min_samples_split': np.arange(2,7),
                  'min_samples_leaf':np.arange(1,5), 'max_features': np.arange(1,X_train.shape[1]+1)} #Parmeter of ML model
        clf = GridSearchCV(Tree, params, scoring='neg_log_loss', refit=True, cv=cv)
        clf.fit(X_train, Y_train)
        print('Best log_loss: {:.4}, with best max_depth: {}, best min_samples_split: {}, min_samples_leaf:{}, max_features_split:{}'.format(clf.best_score_, clf.best_params_['max_depth'], clf.best_params_['min_samples_split'], clf.best_params_['min_samples_leaf'], clf.best_params_['max_features']))
        return clf, Tree, clf.best_score_, clf.best_params_['max_depth']
clf, reg, score, C = train_ML_model(X_train, Y_train, cv=3, model='LogReg')
# At first we will take the average value over every season game
# Load the list that made it to the tournament
def generate_tournament_team_list(year):
    """
    List of teams will be generated that made it into the Tournament.
    Output: List and copy of list.
    """
    df_teams = pd.read_csv('../input/mens-machine-learning-competition-2019/datafiles/NCAATourneyCompactResults.csv').query('Season=='+str(year))
    df_teams = df_teams.sort_values(by=['WTeamID','LTeamID'])
    #print(df_teams)

    a = df_teams.drop_duplicates(subset=['WTeamID'])['WTeamID'] #List of winning teams
    b = df_teams.drop_duplicates(subset=['LTeamID'])['LTeamID'] # List of loosing Teams
    Team_IDs = a.append(b,ignore_index=True).drop_duplicates().values.tolist() # Add both lists and drop duplicates
    Team_IDs_copy = a.append(b,ignore_index=True).drop_duplicates().values.tolist()# List of TeamIDs
    Team_IDs_copy = sorted(Team_IDs_copy)
    Team_IDs = sorted(Team_IDs)
    print ('Number of teams playing in '+str(year)+' Tournament: '+ str(len(Team_IDs))+'/68 (64 + The first four).')
    return Team_IDs, Team_IDs_copy
Team_IDs, Team_IDs_copy = generate_tournament_team_list(year=2017)

def generate_df_for_prediction(df_final, columns, year):
    '''
    Input:  df_dinal: DF that contains features which will be used as estimators. Require 'WTeamID' and 'LTeamID' features.
            columns: List of features that will be averaged. Columns must be contained in df_final
    Output: df with TeamID and averaged feature over the past season.
    '''
    for feature in columns:
        if 'W'+feature not in df_final.columns or 'L'+feature not in df_final.columns:
            raise Exception('The feature {}'.format(feature)+' is not in the DataFrame. All features must be labels of a column of the dataframe.')

    Team_IDs, Team_IDs_copy = generate_tournament_team_list(year)
            
    df_for_prediction = pd.DataFrame(columns=['ID'] + columns)
    counter = 0
    for ID in Team_IDs:
        df_for_prediction.at[counter, 'ID'] = str(ID)
        for feature in columns:
            a = df_final[df_final['WTeamID']==ID]['W'+str(feature)] # Gives a colum of a specific team with the feature called 'feature'.
            b = df_final[df_final['LTeamID']==ID]['L'+str(feature)]
            AVERAGE = a.append(b,ignore_index=True).mean()
            df_for_prediction.at[counter, feature] = AVERAGE
            #df_for_prediction = df_for_prediction.append({'ID':str(ID), str(feature):AVERAGE}, ignore_index=True)
            del a
            del b
            del AVERAGE
        counter += 1
    return df_for_prediction
df_for_prediction = generate_df_for_prediction(df_final, columns=['OR%', 'PIE', 'TOVR', 'eFG%'], year=2016)
df_for_prediction.head()
#df_for_prediction
# Make a DF which provides Game_ID and the difference of an indicator/feature, e.g. 'DeFG%'
def generate_prediction_features(df_for_prediction, columns, year):
    """
    Input:  df_for_prediction: Obtained from generate_df_for_prediction(df_final, columns).
            columns: Features containes in df_for_prediction e.g. ['eFG%','PIE']
            year: Seaon year
    Outout: X_Prediction columns
    """
    for feature in columns:
        if feature not in df_for_prediction.columns:
            raise Exception('The feature {}'.format(feature)+' is not in the DataFrame. All features must be labels of a column of the dataframe.')
    # ['eFG%','PIE']
    col = ['ID','Season','Team1','Team2']
    for feature in columns:
        cols = col + [feature+'1'] + [feature+'2'] + ['D'+feature]
    #cols = ['ID','Season','Team1','Team2','eFG%1','eFG%2','DeFG%']
    df_prediction = pd.DataFrame(columns=cols)
    #Tournament list need to be regenerate every time generater predictions is called because the teams are removed from the list
    Team_IDs, Team_IDs_copy = generate_tournament_team_list(year=year)
    print('Number of teams playing in '+str(year)+' Tournnament: '+ str(len(Team_IDs))+'/68 (64 + The first four).')
    dummy_a = Team_IDs
    dummy_b = Team_IDs_copy
    counter = 0
    for ID1 in dummy_a:
        #print ID
        dummy_b.remove(ID1)
        for ID2 in dummy_b:
            Game_ID = str(year)+'_'+str(ID1)+'_'+str(ID2)
            df_prediction.at[counter, 'ID'] = Game_ID # Add Game_ID value to row = counter and cell='ID' 
            df_prediction.at[counter, 'Season'] = year
            df_prediction.at[counter, 'Team1'] = ID1
            df_prediction.at[counter, 'Team2'] = ID2
            for feature in columns:
                dummy_var_a = df_for_prediction[df_for_prediction['ID'] == str(ID1)][str(feature)].values[0]
                dummy_var_b = df_for_prediction[df_for_prediction['ID'] == str(ID2)][str(feature)].values[0]
                df_prediction.at[counter, feature+'1'] = dummy_var_a
                df_prediction.at[counter, feature+'2'] = dummy_var_b
                df_prediction.at[counter, 'D'+feature] = dummy_var_a - dummy_var_b
            counter += 1
    for features in col: # convert values in features column to float values
        df_prediction['D'+feature] = df_prediction['D'+feature].apply(lambda x: float(x))
    #print len(df_prediction)
    return df_prediction
df_pred = generate_prediction_features(df_for_prediction, columns=['OR%', 'PIE', 'TOVR', 'eFG%'], year=2016)
df_pred.head()
#df_pred['DeFG%'].describe()
def get_predcitions(df_pred, columns, classifier):
    '''Input:   df_pred: DF which containes predictors, e.g. 'DeFG%','DPIE' 
                columns: Colums with labels of predictors, e.g. ['DeFG%','DPIE']
                classifier: A classifier from the machine learning section
       Output: Dataframe with updated predictions based of ML classifier model.
    '''
    for feature in columns:
        if feature not in df_pred.columns:
            raise Exception('The feature {}'.format(feature)+' is not in the DataFrame. All features must be labels of a column of the dataframe.')
    if len(columns) == 1:
        X_pred_columns = df_pred[columns].values.reshape(-1,1)
    elif len(columns) > 1:
        X_pred_columns = df_pred[columns].values
    elif len(columns) == 0:
        return 'Columns should be a list of feature and at least contain 1 feature.'
    
    if type(reg) == type(LinearRegression()):
        X_pred = classifier.predict(X_pred_columns)
    else:
        X_pred = classifier.predict_proba(X_pred_columns)[:,1]
    #print X_pred
    df_pred['Pred'] = X_pred
    df_pred['PredWL'] = df_pred['Pred'].apply(lambda x: 0 if x <= 0.5 else 1)
    return df_pred
df_predicted_values = get_predcitions(df_pred, columns= ['DOR%', 'DPIE', 'DTOVR', 'DeFG%'], classifier=clf)
df_predicted_values.head()
def check_correct_answers(df_predicted_values ,columns, year, print_df_head = False):
    '''
    Input: df_predicted_values: DataFrame with ID (year_team1_team2) and predictions (0 to 1) or predictions as win or loss (0 or 1)
           columns: List of features, e.g. ['eFG%','PIE'].
           year: Season year
    Output: Printed number of games predicted correctly.
    '''
    directory = os.getcwd().replace('\\','/')
    # Load in results
    '../input/datafiles/NCAATourneyCompactResults.csv'
    df_results = pd.read_csv('../input/mens-machine-learning-competition-2019/datafiles/NCAATourneyCompactResults.csv').query('Season=='+str(year))
    df_results['Resu'] = 1
    df_results.head()
    df_results = df_results.sort_values(by=['WTeamID','LTeamID'])
    df_results['ID'] = df_results['Season'].apply(lambda x: str(x))+'_'+df_results['WTeamID'].apply(lambda x: str(x))+'_'+df_results['LTeamID'].apply(lambda x: str(x))
   
    #iterate over df_results['ID'] and swap if team1<team2 (Game_ID = Year_Team1_Team2)
    for index, row in df_results.iterrows():
        team1, team2 = row['ID'].split('_')[1], row['ID'].split('_')[2]
        if float(team1) > float(team2): #This is the case that gives NAN right now
            #print row['ID'].split('_')
            swapped_ID = row['ID'].split('_')[0]+'_'+team2+'_'+team1
            df_results.at[index,'ID'] = swapped_ID
            df_results.at[index,'Resu'] = 0
    #df_results.head()

    # Merge df_results with df_predicted_values
    df_pred_result = df_results.merge(df_predicted_values, how='left', on=['ID'])
    cols = ['Season_x','ID','WTeamID','Team1','WScore','LTeamID','LScore','Team2','Resu','PredWL','Pred']
    for feature in columns:
        cols = cols + [feature+'1', feature+'2', 'D'+feature] #'eFG%1','eFG%2','DeFG%','PIE1', 'PIE2', 'DPIE'
    df_pred_result = df_pred_result[cols]
    if print_df_head == True:
        print(df_pred_result.head())

    # Number of correctly estimated games y
    numOfWins = (df_pred_result['PredWL'] == df_pred_result['Resu']).sum()
    gamesPlayed = float(len((df_pred_result['PredWL'])))
    numOfWinsPerc = round(100*(numOfWins/gamesPlayed),1)
    print(str(year)+' Tournament games predicted correctly: '+ str(numOfWins)+' out of '+ str(gamesPlayed)+'. ('+str(numOfWinsPerc)+'%)')
    return numOfWins, gamesPlayed, numOfWinsPerc
    
numOfWins, gamesPlayed, numOfWinsPerc = check_correct_answers(df_predicted_values, columns = ['eFG%','PIE'], year = 2016, print_df_head = False)
#print numOfWins, gamesPlayed, numOfWinsPerc
# Summarize all functions
year=2016
df_final = load_and_construct_features_in_df(year)
#print df_final.columns

def main(df_final, columns, year, model):
    """Input: Columns: List in the form of ['eFG%','PIE'] instead of ['DeFG%','DPIE']
              model: model: A string of either: 'LinReg','LogReg','DecTree','SVC'.
    """
    columns1 = columns # e.g., ['eFG%','PIE']
    columns2 = list(map(lambda x: 'D'+x, columns)) # e.g., ['DeFG%','DPIE']
    
    X_train, Y_train = get_training_data(df_final, columns = columns2)
    clf, reg, score, C = train_ML_model(X_train, Y_train, cv=3, model=model)

    df_for_prediction = generate_df_for_prediction(df_final, columns= columns1, year=year)
    df_pred = generate_prediction_features(df_for_prediction, columns= columns1, year=year)
    df_predicted_values = get_predcitions(df_pred, columns= columns2, classifier=clf)

    numOfWins, gamesPlayed, numOfWinsPerc = check_correct_answers(df_predicted_values, columns = columns1, year = year, print_df_head = False)
    return year, columns, numOfWins, gamesPlayed, numOfWinsPerc, score, C, clf
columns=['PIE','OR%', 'TOVR']
columns=['PIE']
columns=['OR%', 'PIE', 'TOVR', 'eFG%']
year, columns, numOfWins, gamesPlayed, numOfWinsPerc, score, C, clf = main(df_final, columns=columns, year=year, model='LogReg')
# Note: Before 2011 only 65 Teams played in the NBA.
# This cell was run offline
#Get model parameters for one year but different features
#import itertools

#df_stats = pd.DataFrame(columns=['Season','Columns','WinsPred','GamesPlayed','WinsPred%','Score','C','Coef weight'])
#counter = 0
#for year in range(2016,2017): # for one year
#    df_final = load_and_construct_features_in_df(year)
#    model = 'LinReg' # model: A string of either: 'LinReg','LogReg','DecTree','SVC'
#    columns_main = ['TOVR','eFG%','FTAR','OR%','TDR%','PIE']
    ## The combination of column elements is done here, with 7,6,5,4 elements in the column vector.
#    for k in [6,5,4,3,2,1]: #[6,5,4,3,2,1]
#        list_of_feature_combinations = list(itertools.combinations(columns_main,k))
#        #print len(list_of_feature_combinations)
#        #print list_of_feature_combinations
#        for element in list_of_feature_combinations:
#            col = sorted(list(element))
#            year, columns, numOfWins, gamesPlayed, numOfWinsPerc, score, C, clf= main(df_final, columns=col, year=year, model=model)
#            df_stats.at[counter,'Season'] = year
#            df_stats.at[counter,'Columns'] = columns
#            df_stats.at[counter,'WinsPred'] = numOfWins
#            df_stats.at[counter,'GamesPlayed'] = gamesPlayed
#            df_stats.at[counter,'WinsPred%'] = numOfWinsPerc
#            df_stats.at[counter,'Score'] = score
#            df_stats.at[counter,'C'] = C
#            if model == 'SVC':
#                df_stats.at[counter,'Coef weight'] = 0
#            elif model == 'DecTree':
#                df_stats.at[counter,'Coef weight'] = clf.best_estimator_.feature_importances_ 
#            elif model == 'LogReg':
#                df_stats.at[counter,'Coef weight'] = clf.best_estimator_.coef_[0]
#            elif model == 'LinReg':
#                df_stats.at[counter,'Coef weight'] = clf.best_estimator_.coef_
#            counter += 1
#df_stats
##df_stats.to_csv('ModelFeatureAnalysis_of_LinReg_1-6feat_2016.csv', index=False)
##df_stats.to_csv('ModelFeatureAnalysis_of_LogReg_1-6feat_2016.csv', index=False)
##df_stats.to_csv('ModelFeatureAnalysis_of_SVC_1-6feat_2016.csv', index=False)
##df_stats.to_csv('ModelFeatureAnalysis_of_Tree_1-6feat_2016.csv', index=False)
direc = '../input/features-and-models/'
LinReg_2016_FeatureAnalysis = pd.read_csv(direc+'ModelFeatureAnalysis_of_LinReg_1-6feat_2016.csv')
LinReg_2016_FeatureAnalysis['#Features'] = LinReg_2016_FeatureAnalysis['Columns'].apply(lambda x: len(x.strip('[]').split(',')))
LogReg_2016_FeatureAnalysis = pd.read_csv(direc+'ModelFeatureAnalysis_of_LogReg_1-6feat_2016.csv')
LogReg_2016_FeatureAnalysis['#Features'] = LogReg_2016_FeatureAnalysis['Columns'].apply(lambda x: len(x.strip('[]').split(',')))
Tree_2016_FeatureAnalysis = pd.read_csv(direc+'ModelFeatureAnalysis_of_Tree_1-6feat_2016.csv')
Tree_2016_FeatureAnalysis['#Features'] = Tree_2016_FeatureAnalysis['Columns'].apply(lambda x: len(x.strip('[]').split(',')))
SVC_2016_FeatureAnalysis = pd.read_csv(direc+'ModelFeatureAnalysis_of_SVC_1-6feat_2016.csv')
SVC_2016_FeatureAnalysis['#Features'] = SVC_2016_FeatureAnalysis['Columns'].apply(lambda x: len(x.strip('[]').split(',')))

# Number of features vs WinsPred%
plt.scatter(LinReg_2016_FeatureAnalysis['#Features']+0.15, LinReg_2016_FeatureAnalysis['WinsPred%'], color='C0', alpha=0.35, label='LinReg', zorder=2)
plt.plot(LinReg_2016_FeatureAnalysis.groupby(by=['#Features']).mean().reset_index()['#Features']+0.15, 
            LinReg_2016_FeatureAnalysis.groupby(by=['#Features']).mean().reset_index()['WinsPred%'], 'o-',color='C0', label='AVG')

plt.scatter(LogReg_2016_FeatureAnalysis['#Features']-0.15, LogReg_2016_FeatureAnalysis['WinsPred%'], color='C1', alpha=0.35, label='LogReg')
plt.plot(LogReg_2016_FeatureAnalysis.groupby(by=['#Features']).mean().reset_index()['#Features']-0.15, 
            LogReg_2016_FeatureAnalysis.groupby(by=['#Features']).mean().reset_index()['WinsPred%'], 'o-',color='C1', label='AVG', zorder=4)

plt.scatter(SVC_2016_FeatureAnalysis['#Features']+0.05, SVC_2016_FeatureAnalysis['WinsPred%'], color='C2', alpha=0.35, label='SVC')
plt.plot(SVC_2016_FeatureAnalysis.groupby(by=['#Features']).mean().reset_index()['#Features']+0.05, 
            SVC_2016_FeatureAnalysis.groupby(by=['#Features']).mean().reset_index()['WinsPred%'], 'o-',color='C2', label='AVG', zorder=3)

plt.scatter(Tree_2016_FeatureAnalysis['#Features']-0.05, Tree_2016_FeatureAnalysis['WinsPred%'], color='C3', alpha=0.35, label='Tree')
plt.plot(Tree_2016_FeatureAnalysis.groupby(by=['#Features']).mean().reset_index()['#Features']-0.05, 
            Tree_2016_FeatureAnalysis.groupby(by=['#Features']).mean().reset_index()['WinsPred%'], 'o-',color='C3', label='AVG', zorder=1)


plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,  ncol=2)
plt.xlabel('Number of features', size=13)
plt.ylabel('% of correct predictions', size=13)
plt.tick_params(labelsize=13)
plt.savefig('NumberOfFeaturesvsWins.png', dpi=300, bbox_inches='tight')
dummy = SVC_2016_FeatureAnalysis[SVC_2016_FeatureAnalysis['#Features'] == 1]
plt.scatter(dummy['Columns'], dummy['WinsPred%'])
plt.ylabel('% of correct predictions')
#plt.figure(figsize=(10,10))
Tree_6_Feature = Tree_2016_FeatureAnalysis[Tree_2016_FeatureAnalysis['#Features'] == 6]

def convert_strList_to_floats(df):
    if str(df) == str(Tree_6_Feature):
        dummy = list(map(lambda x: x[:5], df['Coef weight'][0][1:-1].split(' ')[0:-1]))
        return np.array(list(map(lambda x: float(x), [d for d in dummy if d != ''])))
    else:
        dummy = list(map(lambda x: x[:5], df['Coef weight'][0][1:-1].split(' ')[1:]))
        return np.array(list(map(lambda x: float(x), [d for d in dummy if d != ''])))

LogReg_6_Feature = LogReg_2016_FeatureAnalysis[LogReg_2016_FeatureAnalysis['#Features'] == 6]
yLog = convert_strList_to_floats(LogReg_6_Feature)/abs(convert_strList_to_floats(LogReg_6_Feature)).sum()
LinReg_6_Feature = LinReg_2016_FeatureAnalysis[LinReg_2016_FeatureAnalysis['#Features'] == 6]
yLin = convert_strList_to_floats(LinReg_6_Feature)/abs(convert_strList_to_floats(LinReg_6_Feature)).sum()
Tree_6_Feature = Tree_2016_FeatureAnalysis[Tree_2016_FeatureAnalysis['#Features'] == 6]
yTree= convert_strList_to_floats(Tree_6_Feature)/abs(convert_strList_to_floats(Tree_6_Feature)).sum()
plt.bar(np.arange(0, len(yLog),1)-0.2, abs(yLog), width=0.2, label='LogReg')
plt.bar(np.arange(0, len(yLin),1)+0.0, abs(yLin), width=0.2, label='LinReg')
plt.bar(np.arange(0, len(yTree),1)+0.2, abs(yTree), width=0.2, label='Tree')
plt.xlabel('feature name', size=13)
plt.ylabel('$\mid$ coefficient weight $\mid$', size=13)
plt.tick_params(labelsize=13)
featureList= list(map(lambda x: x[1:-1] ,LogReg_2016_FeatureAnalysis['Columns'][0][1:-1].split(', ')))
plt.xticks(range(0, len(featureList)),featureList)  # Set locations and labels
plt.legend()
plt.savefig('CoefficientWeight.png', dpa='1000')
LogReg_2016_FeatureAnalysis[LogReg_2016_FeatureAnalysis['#Features'] == 2]
LogReg_2016_FeatureAnalysis[LogReg_2016_FeatureAnalysis['#Features'] == 5].sort_values(by=['WinsPred%'])
# This cell was run offline
## Train and evalute models for different seasons
#for model in ['LinReg','LogReg','DecTree','SVC']: # 'LinReg','LogReg','DecTree','SVC'
#    df_stats = pd.DataFrame(columns=['Season', 'Columns', 'WinsPred', 'GamesPlayed', 'WinsPred%', 'Coef weight', 'Score', 'C'])
#    counter = 0
#    columns = ['OR%', 'PIE', 'TOVR', 'eFG%'] # PIE', 'OR%', 'TOVR', 'eFG%'
#    for year in range(2003,2019):
#        print model
#        df_final = load_and_construct_features_in_df(year)   
#        year, columns, numOfWins, gamesPlayed, numOfWinsPerc, score, C, clf= main(df_final, columns=columns, year=year, model=model)
#        df_stats.at[counter,'Season'] = year
#        df_stats.at[counter,'Columns'] = columns
#        df_stats.at[counter,'WinsPred'] = numOfWins
#        df_stats.at[counter,'GamesPlayed'] = gamesPlayed
#        df_stats.at[counter,'WinsPred%'] = numOfWinsPerc
#        df_stats.at[counter,'Score'] = score
#        df_stats.at[counter,'C'] = C
#        if model == 'SVC':
#            df_stats.at[counter,'Coef weight'] = 0
#        elif model == 'DecTree':
#            df_stats.at[counter,'Coef weight'] = clf.best_estimator_.feature_importances_ 
#        elif model == 'LogReg':
#            df_stats.at[counter,'Coef weight'] = clf.best_estimator_.coef_[0]
#        elif model == 'LinReg':
#            df_stats.at[counter,'Coef weight'] = clf.best_estimator_.coef_
#        counter += 1
#    print df_stats
#    df_stats.to_csv('ModelAnalysis_of_'+str(model)+'_2003_2018_4features.csv', index=False)
#plt.figure(figsize=(8,8))
direc = '../input/features-and-models/'
Lin_reg_2003_2018 = pd.read_csv(direc+'ModelAnalysis_of_LinReg_2003_2018_6features.csv')
Lin_reg_2003_2018['#Features'] = Lin_reg_2003_2018['Columns'].apply(lambda x: len(x.strip('[]').split(',')))
LogReg_2003_2018 = pd.read_csv(direc+'ModelAnalysis_of_LogReg_2003_2018_6features.csv')
LogReg_2003_2018['#Features'] = LogReg_2003_2018['Columns'].apply(lambda x: len(x.strip('[]').split(',')))
SVCReg_2003_2018 = pd.read_csv(direc+'ModelAnalysis_of_SVCReg_2003_2018_6features.csv')
SVCReg_2003_2018['#Features'] = SVCReg_2003_2018['Columns'].apply(lambda x: len(x.strip('[]').split(',')))
TreeReg_2003_2018 = pd.read_csv(direc+'ModelAnalysis_of_DecTree_2003_2018_6features.csv')
TreeReg_2003_2018['#Features'] = TreeReg_2003_2018['Columns'].apply(lambda x: len(x.strip('[]').split(',')))

plt.plot(LogReg_2003_2018['Season'], LogReg_2003_2018['WinsPred%'], zorder=4, label='LogReg, AVG='+str(round(LogReg_2003_2018['WinsPred%'].mean(),2))+'%')
plt.plot(SVCReg_2003_2018['Season'], SVCReg_2003_2018['WinsPred%'], zorder=3, label='SVC, AVG='+str(round(SVCReg_2003_2018['WinsPred%'].mean(),2))+'%')
plt.plot(Lin_reg_2003_2018['Season'], Lin_reg_2003_2018['WinsPred%'], zorder=2, label='LinReg, AVG='+str(round(Lin_reg_2003_2018['WinsPred%'].mean(),2))+'%')
plt.plot(TreeReg_2003_2018['Season'], TreeReg_2003_2018['WinsPred%'], zorder=1, label='Tree, AVG='+str(round(TreeReg_2003_2018['WinsPred%'].mean(),2))+'%')
plt.xlabel('Season')
plt.ylabel('% of correct predictions')
plt.legend()
plt.savefig('ModelsCompared6features.png')
#plt.figure(figsize=(10,10))
direc = '../input/features-and-models/'
Lin_reg_2003_2018noOR = pd.read_csv(direc+'ModelAnalysis_of_LinReg_2003_2018_5features_noOR.csv')
Lin_reg_2003_2018noOR['#Features'] = Lin_reg_2003_2018noOR['Columns'].apply(lambda x: len(x.strip('[]').split(',')))
LogReg_2003_2018noOR = pd.read_csv(direc+'ModelAnalysis_of_LogReg_2003_2018_5features_noOR.csv')
LogReg_2003_2018noOR['#Features'] = LogReg_2003_2018noOR['Columns'].apply(lambda x: len(x.strip('[]').split(',')))
SVCReg_2003_2018noOR = pd.read_csv(direc+'ModelAnalysis_of_SVC_2003_2018_5features_noOR.csv')
SVCReg_2003_2018noOR['#Features'] = SVCReg_2003_2018noOR['Columns'].apply(lambda x: len(x.strip('[]').split(',')))
TreeReg_2003_2018noOR = pd.read_csv(direc+'ModelAnalysis_of_DecTree_2003_2018_5features_noOR.csv')
TreeReg_2003_2018noOR['#Features'] = TreeReg_2003_2018noOR['Columns'].apply(lambda x: len(x.strip('[]').split(',')))

plt.plot(LogReg_2003_2018noOR['Season'], LogReg_2003_2018noOR['WinsPred%'], zorder=4, label='LogReg, AVG='+str(round(LogReg_2003_2018noOR['WinsPred%'].mean(),2))+'%')
plt.plot(SVCReg_2003_2018noOR['Season'], SVCReg_2003_2018noOR['WinsPred%'], zorder=3, label='SVC, AVG='+str(round(SVCReg_2003_2018noOR['WinsPred%'].mean(),2))+'%')
plt.plot(Lin_reg_2003_2018noOR['Season'], Lin_reg_2003_2018noOR['WinsPred%'], zorder=2, label='LinReg, AVG='+str(round(Lin_reg_2003_2018noOR['WinsPred%'].mean(),2))+'%')
plt.plot(TreeReg_2003_2018noOR['Season'], TreeReg_2003_2018noOR['WinsPred%'], zorder=1, label='Tree, AVG='+str(round(TreeReg_2003_2018noOR['WinsPred%'].mean(),2))+'%')
plt.xlabel('Season')
plt.ylabel('% of correct predictions')
plt.legend()
plt.savefig('ModelsCompared5featuresNoOR%.png')
direc = '../input/features-and-models/'
Lin_reg_2003_2018_4feat = pd.read_csv(direc+'ModelAnalysis_of_LinReg_2003_2018_4features.csv')
Lin_reg_2003_2018_4feat['#Features'] = Lin_reg_2003_2018_4feat['Columns'].apply(lambda x: len(x.strip('[]').split(',')))
LogReg_2003_2018_4feat = pd.read_csv(direc+'ModelAnalysis_of_LogReg_2003_2018_4features.csv')
LogReg_2003_2018_4feat['#Features'] = LogReg_2003_2018_4feat['Columns'].apply(lambda x: len(x.strip('[]').split(',')))
SVCReg_2003_2018_4feat = pd.read_csv(direc+'ModelAnalysis_of_SVC_2003_2018_4features.csv')
SVCReg_2003_2018_4feat['#Features'] = SVCReg_2003_2018_4feat['Columns'].apply(lambda x: len(x.strip('[]').split(',')))
TreeReg_2003_2018_4feat = pd.read_csv(direc+'ModelAnalysis_of_DecTree_2003_2018_4features.csv')
TreeReg_2003_2018_4feat['#Features'] = TreeReg_2003_2018_4feat['Columns'].apply(lambda x: len(x.strip('[]').split(',')))

plt.plot(LogReg_2003_2018_4feat['Season'], LogReg_2003_2018_4feat['WinsPred%'], zorder=4, label='LogReg, AVG='+str(round(LogReg_2003_2018_4feat['WinsPred%'].mean(),2))+'%')
plt.plot(SVCReg_2003_2018_4feat['Season'], SVCReg_2003_2018_4feat['WinsPred%'], zorder=3, label='SVC, AVG='+str(round(SVCReg_2003_2018_4feat['WinsPred%'].mean(),2))+'%')
plt.plot(Lin_reg_2003_2018_4feat['Season'], Lin_reg_2003_2018_4feat['WinsPred%'], zorder=2, label='LinReg, AVG='+str(round(Lin_reg_2003_2018_4feat['WinsPred%'].mean(),2))+'%')
plt.plot(TreeReg_2003_2018_4feat['Season'], TreeReg_2003_2018_4feat['WinsPred%'], zorder=1, label='Tree, AVG='+str(round(TreeReg_2003_2018_4feat['WinsPred%'].mean(),2))+'%')
plt.xlabel('Season')
plt.ylabel('% of correct predictions')
plt.legend()
plt.savefig('ModelsCompared4features.png')