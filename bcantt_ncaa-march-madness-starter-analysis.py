# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns
women_players = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020-Womens-Data/WPlayers.csv')

man_teams = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MTeams.csv')

woman_comp_results = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020-Womens-Data/WDataFiles_Stage1/WRegularSeasonCompactResults.csv')

woman_detailed_results = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020-Womens-Data/WDataFiles_Stage1/WRegularSeasonDetailedResults.csv')
woman_detailed_results
woman_comp_results
woman_comp_results['score_diff'] = woman_comp_results['WScore'] - woman_comp_results['LScore']

woman_detailed_results['score_diff'] = woman_detailed_results['WScore'] - woman_detailed_results['LScore']
ax = sns.barplot(x="WLoc", y="score_diff", data=woman_comp_results)
ax = sns.barplot(x="WLoc", y="score_diff", data=woman_detailed_results)
ax = sns.scatterplot(x="WLoc", y="score_diff", data=woman_comp_results)
the_best_scoring_teams = woman_comp_results.groupby('WTeamID').mean().sort_values('score_diff',ascending =False)
the_best_scoring_teams
woman_detailed_results.corr().sort_values('score_diff',ascending =False)
the_total_score = woman_detailed_results.groupby('WTeamID').mean().sort_values('score_diff',ascending =False) 
the_total_score['total_score'] = the_total_score['score_diff'] +  the_total_score['WFGM']* 0.5 + the_total_score['WAst'] * 0.5+ the_total_score['WStl'] * 0.4 
the_total_score
woman_detailed_results.groupby('WTeamID').count()
the_final_table = pd.merge(woman_detailed_results.groupby('WTeamID').count()['Season'],the_total_score['total_score'],left_index = True,right_index = True)
the_final_table.columns = ['win_count','total_score']
the_final_table.sort_values('win_count',ascending =False)
the_final_table.corr().iloc[1][0]
tourney_detailed_results = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020-Womens-Data/WDataFiles_Stage1/WNCAATourneyDetailedResults.csv')
tourney_detailed_results.groupby('WTeamID').count()
tourney_detailed_results['score_diff'] = tourney_detailed_results['WScore'] - tourney_detailed_results['LScore']
tourney_total_score = tourney_detailed_results.groupby('WTeamID').mean().sort_values('score_diff',ascending =False) 
tourney_total_score.columns
tourney_total_score.corr().sort_values('win',ascending = False)
tourney_total_score['total_score_tourney'] = tourney_total_score['score_diff'] + tourney_total_score['DayNum'] * 0.6 + tourney_total_score['WFGM']* 0.5 + tourney_total_score['WAst'] * 0.3 
the_final_table_tourney = pd.merge(tourney_detailed_results.groupby('WTeamID').count()['Season'],tourney_total_score['total_score_tourney'],left_index = True,right_index = True)
the_final_table_tourney.columns = ['win_count_tourney','total_score_tourney']
def take_samples(number):

    the_list_of_corr = []

    sample = woman_detailed_results.loc[515*number:515*(number+1)]

    the_total_score = sample.groupby('WTeamID').mean().sort_values('score_diff',ascending =False)

    the_total_score['total_score'] = the_total_score['score_diff']  + the_total_score['WScore'] * 0.5 + the_total_score['WFGM']* 0.5 + the_total_score['WAst'] * 0.3+ the_total_score['WDR'] * 0.2 + the_total_score['WBlk'] * 0.2

    the_final_table = pd.merge(sample.groupby('WTeamID').count()['Season'],the_total_score['total_score'],left_index = True,right_index = True)

    the_final_table.columns = ['win_count_league','total_score_league']

    the_final_table = pd.merge(the_final_table_tourney,the_final_table,left_index = True,right_index = True)

    the_final_table['win_count_tourney'] = the_final_table['win_count_tourney'].rank(pct=True)

    the_final_table['total_score_league'] = the_final_table['total_score_league'].rank(pct=True)

    the_list_of_corr.append(the_final_table.corr().iloc[0][3])

    return the_list_of_corr
the_list_of_similarity = [take_samples(i)[0] for i in range(100)]
take_samples(1)
import matplotlib.pyplot as plt

plt.plot(the_list_of_similarity)

plt.ylabel('similartiy correlations')

plt.show()
the_final_table_tourney_league_match = pd.merge(woman_detailed_results,the_final_table_tourney,left_index = True,right_index = True)
the_final_table_tourney_league_match['total_score'] = the_final_table_tourney_league_match['score_diff']  + the_final_table_tourney_league_match['WScore'] * 0.5 + the_final_table_tourney_league_match['WFGM']* 0.5 + the_final_table_tourney_league_match['WAst'] * 0.3+ the_final_table_tourney_league_match['WDR'] * 0.2 + the_final_table_tourney_league_match['WBlk'] * 0.2
the_final_table_tourney_league_match.loc[the_final_table_tourney_league_match.WLoc == "H","WLoc"] = 1

the_final_table_tourney_league_match.loc[the_final_table_tourney_league_match.WLoc == "A","WLoc"] = 2

the_final_table_tourney_league_match.loc[the_final_table_tourney_league_match.WLoc == "N","WLoc"] = 3
from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn import linear_model

from sklearn import svm

from sklearn import tree

import xgboost as xgb

from sklearn.ensemble import BaggingRegressor

import numpy as np 

import pandas as pd 

import random

from sklearn.decomposition import PCA
X = the_final_table_tourney_league_match.drop('win_count_tourney',axis = 1)

y = the_final_table_tourney_league_match['win_count_tourney']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
regr = RandomForestRegressor()

regr.fit(X_train, y_train)



predictions = regr.predict(X_test)


pca = PCA(n_components=3)

principalComponents_train = pca.fit_transform(X)

sum(pca.explained_variance_ratio_)
X['component_1'] = [i[0] for i in principalComponents_train]

X['component_2'] = [i[1] for i in principalComponents_train]

X['component_3'] = [i[2] for i in principalComponents_train]
X = the_final_table_tourney_league_match.drop('win_count_tourney',axis = 1)

y = the_final_table_tourney_league_match['win_count_tourney']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
regr = RandomForestRegressor(n_estimators = 400,min_samples_split = 2,min_samples_leaf = 1,max_features= 'sqrt',max_depth =None,bootstrap= False)

regr.fit(X_train, y_train)



predictions = regr.predict(X_test)
mean_squared_error(predictions.round(), y_test)
ml_df = pd.DataFrame(predictions.round(),y_test).reset_index()
ml_df.corr()
the_final_table_tourney_league_match.corr().sort_values('win_count_tourney',ascending = False)
def take_samples(number):

    the_list_of_corr = []

    the_list_of_parameters = []

    sample = woman_detailed_results.loc[200*number:200*(number+1)]

    random_num_list = [random.randint(-100,100) for i in range(11)]

    the_total_score = sample.groupby('WTeamID').mean().sort_values('score_diff',ascending =False)

    the_total_score['total_score'] = the_total_score['score_diff'] * random_num_list[0] + the_total_score['WScore'] * random_num_list[1]+ the_total_score['WFGM']* random_num_list[2] + the_total_score['WAst'] * random_num_list[3] + the_total_score['WDR'] * random_num_list[4] + the_total_score['WBlk'] * random_num_list[5] + the_total_score['WFGM3'] * random_num_list[6] + the_total_score['WFGA3'] * random_num_list[7] + the_total_score['WFTA'] * random_num_list[8] + the_total_score['WOR'] * random_num_list[9] + the_total_score['WFTM'] * random_num_list[10]

   

    the_final_table = pd.merge(sample.groupby('WTeamID').count()['Season'],the_total_score['total_score'],left_index = True,right_index = True)

    the_final_table.columns = ['win_count_league','total_score_league']

    the_final_table = pd.merge(the_final_table_tourney,the_final_table,left_index = True,right_index = True)

    the_final_table['win_count_tourney'] = the_final_table['win_count_tourney'].rank(pct=True)

    the_final_table['total_score_league'] = the_final_table['total_score_league'].rank(pct=True)

    the_list_of_corr.append(the_final_table.corr().iloc[0][3])

    the_list_of_parameters.append(random_num_list)

    return the_list_of_corr,the_list_of_parameters
take_samples(0)
the_list_of_similarity_parameters = [take_samples(i) for i in range(100)]

the_similarity_list = [i[0] for i in the_list_of_similarity_parameters]
import matplotlib.pyplot as plt

plt.plot(the_similarity_list)

plt.ylabel('similartiy correlations')

plt.show()
the_max_value = 0

parameter_list = []

for i in range(100):

    the_list_of_similarity_parameters = [take_samples(i) for i in range(200)]

    the_similarity_list = [i[0] for i in the_list_of_similarity_parameters]

    the_parameter_list  = [i[1] for i in the_list_of_similarity_parameters]

    if max(the_similarity_list)[0] > the_max_value:

        the_max_value = max(the_similarity_list)[0]

        parameter_list = the_parameter_list[the_similarity_list.index(max(the_similarity_list)[0])]

        print(the_max_value)

        print(parameter_list)

        

    