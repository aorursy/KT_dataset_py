# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import sqlalchemy

from sqlalchemy import create_engine # database connection



from IPython.display import display, clear_output

from sklearn import metrics

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import BernoulliNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn import preprocessing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
engine  = create_engine("sqlite:///../input/database.sqlite")

countries = pd.read_sql_query('SELECT * FROM Country;', engine)

countries.rename(columns={'id':'country_id', 'name':'Country'}, inplace=True)

countries.head()


leagues = pd.read_sql_query('SELECT * FROM League;', engine)

leagues.rename(columns={'id':'league_id', 'name':'League'}, inplace=True)

leagues
# Select a number of seasons in a list or just everything

#matches = pd.read_sql_query('SELECT * FROM Match where league_id = 1729 and season in ("2010/2011", "2011/2012", "2012/2013", "2013/2014", "2014/2015", "2015/2016");'

#                                          , engine)

#                           "2008/2009", "2009/2010",         \

matches = pd.read_sql_query('SELECT * FROM Match where league_id = 1729 ;', engine)



#matches.info()

#matches.head()

# matches.tail()

#matches.shape

matches.describe

# matches.dtypes

#matches.loc[(matches["season"]=="2012/2013")].count()
matches = matches[matches.columns[:11]]

# sample = matches.league_id == 1729

#matches.dtypes
matches
teams = pd.read_sql_query('SELECT * FROM Team;', engine)

teams.head()

#teams.loc[teams['team_long_name'] == 'Blackburn Rovers']
# Add home team name column

matches = pd.merge(left=matches, right=teams, how='left', left_on='home_team_api_id', right_on='team_api_id')

matches = matches.drop(['country_id','league_id', 'home_team_api_id', 'id_y', 'team_api_id', 'team_fifa_api_id', 'team_short_name'], axis=1)

matches.rename(columns={'id_x':'match_id', 'team_long_name':'home_team'}, inplace=True)

#matches.tail()
# Add away team name column

matches = pd.merge(left=matches, right=teams, how='left', left_on='away_team_api_id', right_on='team_api_id')

matches = matches.drop(['id', 'match_api_id', 'away_team_api_id','team_api_id', 'team_fifa_api_id', 'team_short_name'], axis=1)

matches.rename(columns={'id_x':'match_id', 'team_long_name':'away_team'}, inplace=True)

matches
#check_matches = matches.loc[((matches["home_team"]=="Arsenal") | (matches["away_team"]=="Arsenal"))

#& (matches["season"]=="2011/2012")].count()

#check_matches

# matches[:-1]

#matches.loc[matches['home_team'] == 'West Ham']

#unique_teams.sort_values('team')
# Add in this season (16/17) matches

latest_match_data = [

{'match_id':6000, 'season':'2016/2017', 'stage':1, 'date':'42595', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Burnley', 'away_team':'Swansea City'},

{'match_id':6001, 'season':'2016/2017', 'stage':1, 'date':'42595', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Crystal Palace', 'away_team':'West Bromwich Albion'},

{'match_id':6002, 'season':'2016/2017', 'stage':1, 'date':'42595', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Everton', 'away_team':'Tottenham Hotspur'},

{'match_id':6003, 'season':'2016/2017', 'stage':1, 'date':'42595', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Hull City', 'away_team':'Leicester City'},

{'match_id':6004, 'season':'2016/2017', 'stage':1, 'date':'42595', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Manchester City', 'away_team':'Sunderland'},

{'match_id':6005, 'season':'2016/2017', 'stage':1, 'date':'42595', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Middlesbrough', 'away_team':'Stoke City'},

{'match_id':6006, 'season':'2016/2017', 'stage':1, 'date':'42595', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Southampton', 'away_team':'Watford'},

{'match_id':6007, 'season':'2016/2017', 'stage':1, 'date':'42596', 'home_team_goal':3, 'away_team_goal':4, 'home_team':'Arsenal', 'away_team':'Liverpool'},

{'match_id':6008, 'season':'2016/2017', 'stage':1, 'date':'42596', 'home_team_goal':1, 'away_team_goal':3, 'home_team':'Bournemouth', 'away_team':'Manchester United'},

{'match_id':6009, 'season':'2016/2017', 'stage':1, 'date':'42597', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Chelsea', 'away_team':'West Ham United'},

{'match_id':6010, 'season':'2016/2017', 'stage':2, 'date':'42601', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Manchester United', 'away_team':'Southampton'},

{'match_id':6011, 'season':'2016/2017', 'stage':2, 'date':'42602', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Burnley', 'away_team':'Liverpool'},

{'match_id':6012, 'season':'2016/2017', 'stage':2, 'date':'42602', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Leicester City', 'away_team':'Arsenal'},

{'match_id':6013, 'season':'2016/2017', 'stage':2, 'date':'42602', 'home_team_goal':1, 'away_team_goal':4, 'home_team':'Stoke City', 'away_team':'Manchester City'},

{'match_id':6014, 'season':'2016/2017', 'stage':2, 'date':'42602', 'home_team_goal':0, 'away_team_goal':2, 'home_team':'Swansea City', 'away_team':'Hull City'},

{'match_id':6015, 'season':'2016/2017', 'stage':2, 'date':'42602', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Tottenham Hotspur', 'away_team':'Crystal Palace'},

{'match_id':6016, 'season':'2016/2017', 'stage':2, 'date':'42602', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Watford', 'away_team':'Chelsea'},

{'match_id':6017, 'season':'2016/2017', 'stage':2, 'date':'42602', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'West Bromwich Albion', 'away_team':'Everton'},

{'match_id':6018, 'season':'2016/2017', 'stage':2, 'date':'42603', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Sunderland', 'away_team':'Middlesbrough'},

{'match_id':6019, 'season':'2016/2017', 'stage':2, 'date':'42603', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'West Ham United', 'away_team':'Bournemouth'},

{'match_id':6020, 'season':'2016/2017', 'stage':3, 'date':'42609', 'home_team_goal':3, 'away_team_goal':0, 'home_team':'Chelsea', 'away_team':'Burnley'},

{'match_id':6021, 'season':'2016/2017', 'stage':3, 'date':'42609', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Crystal Palace', 'away_team':'Bournemouth'},

{'match_id':6022, 'season':'2016/2017', 'stage':3, 'date':'42609', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Everton', 'away_team':'Stoke City'},

{'match_id':6023, 'season':'2016/2017', 'stage':3, 'date':'42609', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Hull City', 'away_team':'Manchester United'},

{'match_id':6024, 'season':'2016/2017', 'stage':3, 'date':'42609', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Leicester City', 'away_team':'Swansea City'},

{'match_id':6025, 'season':'2016/2017', 'stage':3, 'date':'42609', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Southampton', 'away_team':'Sunderland'},

{'match_id':6026, 'season':'2016/2017', 'stage':3, 'date':'42609', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Tottenham Hotspur', 'away_team':'Liverpool'},

{'match_id':6027, 'season':'2016/2017', 'stage':3, 'date':'42609', 'home_team_goal':1, 'away_team_goal':3, 'home_team':'Watford', 'away_team':'Arsenal'},

{'match_id':6028, 'season':'2016/2017', 'stage':3, 'date':'42610', 'home_team_goal':3, 'away_team_goal':1, 'home_team':'Manchester City', 'away_team':'West Ham United'},

{'match_id':6029, 'season':'2016/2017', 'stage':3, 'date':'42610', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'West Bromwich Albion', 'away_team':'Middlesbrough'},

{'match_id':6030, 'season':'2016/2017', 'stage':4, 'date':'42623', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Arsenal', 'away_team':'Southampton'},

{'match_id':6031, 'season':'2016/2017', 'stage':4, 'date':'42623', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Bournemouth', 'away_team':'West Bromwich Albion'},

{'match_id':6032, 'season':'2016/2017', 'stage':4, 'date':'42623', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Burnley', 'away_team':'Hull City'},

{'match_id':6033, 'season':'2016/2017', 'stage':4, 'date':'42623', 'home_team_goal':4, 'away_team_goal':1, 'home_team':'Liverpool', 'away_team':'Leicester City'},

{'match_id':6034, 'season':'2016/2017', 'stage':4, 'date':'42623', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Manchester United', 'away_team':'Manchester City'},

{'match_id':6035, 'season':'2016/2017', 'stage':4, 'date':'42623', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Middlesbrough', 'away_team':'Crystal Palace'},

{'match_id':6036, 'season':'2016/2017', 'stage':4, 'date':'42623', 'home_team_goal':0, 'away_team_goal':4, 'home_team':'Stoke City', 'away_team':'Tottenham Hotspur'},

{'match_id':6037, 'season':'2016/2017', 'stage':4, 'date':'42623', 'home_team_goal':2, 'away_team_goal':4, 'home_team':'West Ham United', 'away_team':'Watford'},

{'match_id':6038, 'season':'2016/2017', 'stage':4, 'date':'42624', 'home_team_goal':2, 'away_team_goal':2, 'home_team':'Swansea City', 'away_team':'Chelsea'},

{'match_id':6039, 'season':'2016/2017', 'stage':4, 'date':'42625', 'home_team_goal':0, 'away_team_goal':3, 'home_team':'Sunderland', 'away_team':'Everton'},

{'match_id':6040, 'season':'2016/2017', 'stage':5, 'date':'42629', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Chelsea', 'away_team':'Liverpool'},

{'match_id':6041, 'season':'2016/2017', 'stage':5, 'date':'42630', 'home_team_goal':3, 'away_team_goal':1, 'home_team':'Everton', 'away_team':'Middlesbrough'},

{'match_id':6042, 'season':'2016/2017', 'stage':5, 'date':'42630', 'home_team_goal':1, 'away_team_goal':4, 'home_team':'Hull City', 'away_team':'Arsenal'},

{'match_id':6043, 'season':'2016/2017', 'stage':5, 'date':'42630', 'home_team_goal':3, 'away_team_goal':0, 'home_team':'Leicester City', 'away_team':'Burnley'},

{'match_id':6044, 'season':'2016/2017', 'stage':5, 'date':'42630', 'home_team_goal':4, 'away_team_goal':0, 'home_team':'Manchester City', 'away_team':'Bournemouth'},

{'match_id':6045, 'season':'2016/2017', 'stage':5, 'date':'42630', 'home_team_goal':4, 'away_team_goal':2, 'home_team':'West Bromwich Albion', 'away_team':'West Ham United'},

{'match_id':6046, 'season':'2016/2017', 'stage':5, 'date':'42631', 'home_team_goal':4, 'away_team_goal':1, 'home_team':'Crystal Palace', 'away_team':'Stoke City'},

{'match_id':6047, 'season':'2016/2017', 'stage':5, 'date':'42631', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Southampton', 'away_team':'Swansea City'},

{'match_id':6048, 'season':'2016/2017', 'stage':5, 'date':'42631', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Tottenham Hotspur', 'away_team':'Sunderland'},

{'match_id':6049, 'season':'2016/2017', 'stage':5, 'date':'42631', 'home_team_goal':3, 'away_team_goal':1, 'home_team':'Watford', 'away_team':'Manchester United'},

{'match_id':6050, 'season':'2016/2017', 'stage':6, 'date':'42637', 'home_team_goal':3, 'away_team_goal':0, 'home_team':'Arsenal', 'away_team':'Chelsea'},

{'match_id':6051, 'season':'2016/2017', 'stage':6, 'date':'42637', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Bournemouth', 'away_team':'Everton'},

{'match_id':6052, 'season':'2016/2017', 'stage':6, 'date':'42637', 'home_team_goal':5, 'away_team_goal':1, 'home_team':'Liverpool', 'away_team':'Hull City'},

{'match_id':6053, 'season':'2016/2017', 'stage':6, 'date':'42637', 'home_team_goal':4, 'away_team_goal':1, 'home_team':'Manchester United', 'away_team':'Leicester City'},

{'match_id':6054, 'season':'2016/2017', 'stage':6, 'date':'42637', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Middlesbrough', 'away_team':'Tottenham Hotspur'},

{'match_id':6055, 'season':'2016/2017', 'stage':6, 'date':'42637', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Stoke City', 'away_team':'West Bromwich Albion'},

{'match_id':6056, 'season':'2016/2017', 'stage':6, 'date':'42637', 'home_team_goal':2, 'away_team_goal':3, 'home_team':'Sunderland', 'away_team':'Crystal Palace'},

{'match_id':6057, 'season':'2016/2017', 'stage':6, 'date':'42637', 'home_team_goal':1, 'away_team_goal':3, 'home_team':'Swansea City', 'away_team':'Manchester City'},

{'match_id':6058, 'season':'2016/2017', 'stage':6, 'date':'42638', 'home_team_goal':0, 'away_team_goal':3, 'home_team':'West Ham United', 'away_team':'Southampton'},

{'match_id':6059, 'season':'2016/2017', 'stage':6, 'date':'42639', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Burnley', 'away_team':'Watford'},

{'match_id':6060, 'season':'2016/2017', 'stage':7, 'date':'42643', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Everton', 'away_team':'Crystal Palace'},

{'match_id':6061, 'season':'2016/2017', 'stage':7, 'date':'42644', 'home_team_goal':0, 'away_team_goal':2, 'home_team':'Hull City', 'away_team':'Chelsea'},

{'match_id':6062, 'season':'2016/2017', 'stage':7, 'date':'42644', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Sunderland', 'away_team':'West Bromwich Albion'},

{'match_id':6063, 'season':'2016/2017', 'stage':7, 'date':'42644', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Swansea City', 'away_team':'Liverpool'},

{'match_id':6064, 'season':'2016/2017', 'stage':7, 'date':'42644', 'home_team_goal':2, 'away_team_goal':2, 'home_team':'Watford', 'away_team':'Bournemouth'},

{'match_id':6065, 'season':'2016/2017', 'stage':7, 'date':'42644', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'West Ham United', 'away_team':'Middlesbrough'},

{'match_id':6066, 'season':'2016/2017', 'stage':7, 'date':'42645', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Burnley', 'away_team':'Arsenal'},

{'match_id':6067, 'season':'2016/2017', 'stage':7, 'date':'42645', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Leicester City', 'away_team':'Southampton'},

{'match_id':6068, 'season':'2016/2017', 'stage':7, 'date':'42645', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Manchester United', 'away_team':'Stoke City'},

{'match_id':6069, 'season':'2016/2017', 'stage':7, 'date':'42645', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Tottenham Hotspur', 'away_team':'Manchester City'},

{'match_id':6070, 'season':'2016/2017', 'stage':8, 'date':'42658', 'home_team_goal':3, 'away_team_goal':0, 'home_team':'Chelsea', 'away_team':'Leicester City'},

{'match_id':6071, 'season':'2016/2017', 'stage':8, 'date':'42658', 'home_team_goal':3, 'away_team_goal':2, 'home_team':'Arsenal', 'away_team':'Swansea City'},

{'match_id':6072, 'season':'2016/2017', 'stage':8, 'date':'42658', 'home_team_goal':6, 'away_team_goal':1, 'home_team':'Bournemouth', 'away_team':'Hull City'},

{'match_id':6073, 'season':'2016/2017', 'stage':8, 'date':'42658', 'home_team_goal':3, 'away_team_goal':1, 'home_team':'Manchester City', 'away_team':'Everton'},

{'match_id':6074, 'season':'2016/2017', 'stage':8, 'date':'42658', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Stoke City', 'away_team':'Sunderland'},

{'match_id':6075, 'season':'2016/2017', 'stage':8, 'date':'42658', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'West Bromwich Albion', 'away_team':'Tottenham Hotspur'},

{'match_id':6076, 'season':'2016/2017', 'stage':8, 'date':'42658', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Crystal Palace', 'away_team':'West Ham United'},

{'match_id':6077, 'season':'2016/2017', 'stage':8, 'date':'42659', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Middlesbrough', 'away_team':'Watford'},

{'match_id':6078, 'season':'2016/2017', 'stage':8, 'date':'42659', 'home_team_goal':3, 'away_team_goal':1, 'home_team':'Southampton', 'away_team':'Burnley'},

{'match_id':6079, 'season':'2016/2017', 'stage':8, 'date':'42660', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Liverpool', 'away_team':'Manchester United'},

{'match_id':6080, 'season':'2016/2017', 'stage':9, 'date':'42665', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Bournemouth', 'away_team':'Tottenham Hotspur'},

{'match_id':6081, 'season':'2016/2017', 'stage':9, 'date':'42665', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Arsenal', 'away_team':'Middlesbrough'},

{'match_id':6082, 'season':'2016/2017', 'stage':9, 'date':'42665', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Burnley', 'away_team':'Everton'},

{'match_id':6083, 'season':'2016/2017', 'stage':9, 'date':'42665', 'home_team_goal':4, 'away_team_goal':0, 'home_team':'Chelsea', 'away_team':'Manchester United'},

{'match_id':6084, 'season':'2016/2017', 'stage':9, 'date':'42665', 'home_team_goal':0, 'away_team_goal':2, 'home_team':'Hull City', 'away_team':'Stoke City'},

{'match_id':6085, 'season':'2016/2017', 'stage':9, 'date':'42665', 'home_team_goal':3, 'away_team_goal':1, 'home_team':'Leicester City', 'away_team':'Crystal Palace'},

{'match_id':6086, 'season':'2016/2017', 'stage':9, 'date':'42665', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Liverpool', 'away_team':'West Bromwich Albion'},

{'match_id':6087, 'season':'2016/2017', 'stage':9, 'date':'42665', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Manchester City', 'away_team':'Southampton'},

{'match_id':6088, 'season':'2016/2017', 'stage':9, 'date':'42665', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Swansea City', 'away_team':'Watford'},

{'match_id':6089, 'season':'2016/2017', 'stage':9, 'date':'42665', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'West Ham United', 'away_team':'Sunderland'},

{'match_id':6090, 'season':'2016/2017', 'stage':10, 'date':'42672', 'home_team_goal':2, 'away_team_goal':4, 'home_team':'Crystal Palace', 'away_team':'Liverpool'},

{'match_id':6091, 'season':'2016/2017', 'stage':10, 'date':'42672', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Manchester United', 'away_team':'Burnley'},

{'match_id':6092, 'season':'2016/2017', 'stage':10, 'date':'42672', 'home_team_goal':1, 'away_team_goal':4, 'home_team':'Sunderland', 'away_team':'Arsenal'},

{'match_id':6093, 'season':'2016/2017', 'stage':10, 'date':'42672', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Tottenham Hotspur', 'away_team':'Leicester City'},

{'match_id':6094, 'season':'2016/2017', 'stage':10, 'date':'42672', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Watford', 'away_team':'Hull City'},

{'match_id':6095, 'season':'2016/2017', 'stage':10, 'date':'42672', 'home_team_goal':0, 'away_team_goal':4, 'home_team':'West Bromwich Albion', 'away_team':'Manchester City'},

{'match_id':6096, 'season':'2016/2017', 'stage':10, 'date':'42672', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Middlesbrough', 'away_team':'Bournemouth'},

{'match_id':6097, 'season':'2016/2017', 'stage':10, 'date':'42673', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Everton', 'away_team':'West Ham United'},

{'match_id':6098, 'season':'2016/2017', 'stage':10, 'date':'42673', 'home_team_goal':0, 'away_team_goal':2, 'home_team':'Southampton', 'away_team':'Chelsea'},

{'match_id':6099, 'season':'2016/2017', 'stage':10, 'date':'42674', 'home_team_goal':3, 'away_team_goal':1, 'home_team':'Stoke City', 'away_team':'Swansea City'},

{'match_id':6100, 'season':'2016/2017', 'stage':11, 'date':'42679', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Bournemouth', 'away_team':'Sunderland'},

{'match_id':6101, 'season':'2016/2017', 'stage':11, 'date':'42679', 'home_team_goal':3, 'away_team_goal':2, 'home_team':'Burnley', 'away_team':'Crystal Palace'},

{'match_id':6102, 'season':'2016/2017', 'stage':11, 'date':'42679', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Manchester City', 'away_team':'Middlesbrough'},

{'match_id':6103, 'season':'2016/2017', 'stage':11, 'date':'42679', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'West Ham United', 'away_team':'Stoke City'},

{'match_id':6104, 'season':'2016/2017', 'stage':11, 'date':'42679', 'home_team_goal':5, 'away_team_goal':0, 'home_team':'Chelsea', 'away_team':'Everton'},

{'match_id':6105, 'season':'2016/2017', 'stage':11, 'date':'42680', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Arsenal', 'away_team':'Tottenham Hotspur'},

{'match_id':6106, 'season':'2016/2017', 'stage':11, 'date':'42680', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Hull City', 'away_team':'Southampton'},

{'match_id':6107, 'season':'2016/2017', 'stage':11, 'date':'42680', 'home_team_goal':6, 'away_team_goal':1, 'home_team':'Liverpool', 'away_team':'Watford'},

{'match_id':6108, 'season':'2016/2017', 'stage':11, 'date':'42680', 'home_team_goal':1, 'away_team_goal':3, 'home_team':'Swansea City', 'away_team':'Manchester United'},

{'match_id':6109, 'season':'2016/2017', 'stage':11, 'date':'42680', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Leicester City', 'away_team':'West Bromwich Albion'},

{'match_id':6110, 'season':'2016/2017', 'stage':12, 'date':'42693', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Manchester United', 'away_team':'Arsenal'},

{'match_id':6111, 'season':'2016/2017', 'stage':12, 'date':'42693', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Crystal Palace', 'away_team':'Manchester City'},

{'match_id':6112, 'season':'2016/2017', 'stage':12, 'date':'42693', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Everton', 'away_team':'Swansea City'},

{'match_id':6113, 'season':'2016/2017', 'stage':12, 'date':'42693', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Southampton', 'away_team':'Liverpool'},

{'match_id':6114, 'season':'2016/2017', 'stage':12, 'date':'42693', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Stoke City', 'away_team':'Bournemouth'},

{'match_id':6115, 'season':'2016/2017', 'stage':12, 'date':'42693', 'home_team_goal':3, 'away_team_goal':0, 'home_team':'Sunderland', 'away_team':'Hull City'},

{'match_id':6116, 'season':'2016/2017', 'stage':12, 'date':'42693', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Watford', 'away_team':'Leicester City'},

{'match_id':6117, 'season':'2016/2017', 'stage':12, 'date':'42693', 'home_team_goal':3, 'away_team_goal':2, 'home_team':'Tottenham Hotspur', 'away_team':'West Ham United'},

{'match_id':6118, 'season':'2016/2017', 'stage':12, 'date':'42694', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Middlesbrough', 'away_team':'Chelsea'},

{'match_id':6119, 'season':'2016/2017', 'stage':12, 'date':'42695', 'home_team_goal':4, 'away_team_goal':0, 'home_team':'West Bromwich Albion', 'away_team':'Burnley'},

{'match_id':6119, 'season':'2016/2017', 'stage':12, 'date':'42695', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'West Bromwich Albion', 'away_team':'Burnley'},

{'match_id':6120, 'season':'2016/2017', 'stage':13, 'date':'42700', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Burnley', 'away_team':'Manchester City'},

{'match_id':6121, 'season':'2016/2017', 'stage':13, 'date':'42700', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Hull City', 'away_team':'West Bromwich Albion'},

{'match_id':6122, 'season':'2016/2017', 'stage':13, 'date':'42700', 'home_team_goal':2, 'away_team_goal':2, 'home_team':'Leicester City', 'away_team':'Middlesbrough'},

{'match_id':6123, 'season':'2016/2017', 'stage':13, 'date':'42700', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Liverpool', 'away_team':'Sunderland'},

{'match_id':6124, 'season':'2016/2017', 'stage':13, 'date':'42700', 'home_team_goal':5, 'away_team_goal':4, 'home_team':'Swansea City', 'away_team':'Crystal Palace'},

{'match_id':6125, 'season':'2016/2017', 'stage':13, 'date':'42700', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Chelsea', 'away_team':'Tottenham Hotspur'},

{'match_id':6126, 'season':'2016/2017', 'stage':13, 'date':'42701', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Watford', 'away_team':'Stoke City'},

{'match_id':6127, 'season':'2016/2017', 'stage':13, 'date':'42701', 'home_team_goal':3, 'away_team_goal':1, 'home_team':'Arsenal', 'away_team':'Bournemouth'},

{'match_id':6128, 'season':'2016/2017', 'stage':13, 'date':'42701', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Manchester United', 'away_team':'West Ham United'},

{'match_id':6129, 'season':'2016/2017', 'stage':13, 'date':'42701', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Southampton', 'away_team':'Everton'},

{'match_id':6130, 'season':'2016/2017', 'stage':14, 'date':'42707', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Manchester City', 'away_team':'Chelsea'},

{'match_id':6131, 'season':'2016/2017', 'stage':14, 'date':'42707', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Crystal Palace', 'away_team':'Southampton'},

{'match_id':6132, 'season':'2016/2017', 'stage':14, 'date':'42707', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Stoke City', 'away_team':'Burnley'},

{'match_id':6133, 'season':'2016/2017', 'stage':14, 'date':'42707', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Sunderland', 'away_team':'Leicester City'},

{'match_id':6134, 'season':'2016/2017', 'stage':14, 'date':'42707', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Tottenham Hotspur', 'away_team':'Swansea City'},

{'match_id':6135, 'season':'2016/2017', 'stage':14, 'date':'42707', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'West Bromwich Albion', 'away_team':'Watford'},

{'match_id':6136, 'season':'2016/2017', 'stage':14, 'date':'42707', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'West Ham United', 'away_team':'Arsenal'},

{'match_id':6137, 'season':'2016/2017', 'stage':14, 'date':'42708', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Bournemouth', 'away_team':'Liverpool'},

{'match_id':6138, 'season':'2016/2017', 'stage':14, 'date':'42708', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Everton', 'away_team':'Manchester United'},

{'match_id':6139, 'season':'2016/2017', 'stage':14, 'date':'42709', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Middlesbrough', 'away_team':'Hull City'}

]



latest_matches = pd.DataFrame(latest_match_data, columns=['match_id', 'season', 'stage', 'date',

                                                          'home_team_goal', 'away_team_goal',

                                                          'home_team','away_team'])

latest_matches



# Add to full training data to predict current season

matches = pd.concat([matches, latest_matches])

matches = matches.reset_index(drop=True)

matches
# Create a "full set" of match data that can be used with feature engineering later

# Only the season's we require

full_matches = matches.copy()



# Regression model (see notes later in notebook)

# Build the initial model and update for each subsequent season

unique_seasons = pd.Series(matches['season'].unique())

regress_seasons = pd.Series(['2008/2009', '2009/2010', '2010/2011'])

regress_matches = full_matches.loc[full_matches['season'].isin(regress_seasons)]



remaining_seasons = unique_seasons[~unique_seasons.isin(regress_seasons)]



# Remove the regression seasons from the general training matches

#full_matches = full_matches.loc[full_matches['season'].isin(remaining_seasons)]

#full_matches.reset_index(drop=True, inplace=True)



# Optimum appears to be for season 10/11 onwards

exclude_seasons = pd.Series(['2008/2009', '2009/2010', '2010/2011'])

#

include_seasons = unique_seasons[~unique_seasons.isin(exclude_seasons)]

full_matches = full_matches.loc[full_matches['season'].isin(include_seasons)]

#full_matches.reset_index(drop=True, inplace=True)





full_matches
# Cope with newly promoted teams with limited or no stats

# Work out an average of newly promoted teams from matches in their first season and use that

# For new teams in the season I am predicting



###################################################################################

# 1. Identify newly promoted teams and their first season

# 2. Select all of their matches into a dataframe similar to matches

# Build a model specific to new teams

# 3. Aggregate their data as a single team - average performance of all new teams

# 4. This can be added to the main model as team 'Newly Promoted"

# 5. For prediction, any newly promoted team uses the new team average

###################################################################################



# For now, remove new teams from the test set

# new_teams = ['Watford', 'Bournemouth', 'Leicester']

# test_matches = test_matches[~test_matches['home_team'].isin(new_teams)] 

# test_matches = test_matches[~test_matches['away_team'].isin(new_teams)]



# Newly promoted teams each season

team_data = {'team':['West Bromwich Albion', 'Stoke City', 'Hull City',

                     'Wolverhampton Wanderers', 'Birmingham City', 'Burnley',

                     'Newcastle United', 'West Bromwich Albion', 'Blackpool',

                     'Queens Park Rangers', 'Norwich City', 'Swansea City',

                     'Reading', 'Southampton', 'West Ham United',

                     'Cardiff City', 'Crystal Palace', 'Hull City',

                     'Leicester City', 'Burnley', 'Queens Park Rangers',

                     'Bournemouth', 'Watford', 'Norwich City',

                     'Burnley', 'Middlesbrough', 'Hull City'

                    ],

             'season':["2008/2009", "2008/2009", "2008/2009",

                       "2009/2010", "2009/2010", "2009/2010",

                       "2010/2011", "2010/2011", "2010/2011",

                       "2011/2012", "2011/2012", "2011/2012",

                       "2012/2013", "2012/2013", "2012/2013",

                       "2013/2014", "2013/2014", "2013/2014",

                       "2014/2015", "2014/2015", "2014/2015",

                       "2015/2016", "2015/2016", "2015/2016",

                       "2016/2017", "2016/2017", "2016/2017"

                      ]

            }

new_teams = pd.DataFrame(team_data, columns=['team', 'season'])



# new teams for "2016/2017" are 'Burnley', 'Middlesbrough', 'Hull City'
def create_new_team_matches(match_list, team_list):

    # Select all new team matches into a dataframe similar to matches

    new_team_match_list = pd.DataFrame()

    for index, row in team_list.iterrows():

        new_team_match_list = pd.concat([new_team_match_list, match_list.loc[((match_list['home_team'] == row['team']) |

                                                      (match_list['away_team'] == row['team'])) &

                                                       (match_list['season'] == row['season']) ]])



    # Remove duplicates by identifying index dupes and then using that as a filter

    new_team_match_list = new_team_match_list[~new_team_match_list.index.duplicated()]

    return new_team_match_list





new_team_matches = create_new_team_matches(matches, new_teams)

# Various ways to check that I have what I want:



# new_team_matches_agg = new_team_matches.groupby(['season', 'away_team']).count()

#new_team_matches

# new_team_matches.loc[new_team_matches['season'] == "2011/2012"]

# new_team_matches.loc[new_team_matches['match_id'] == 1897]
# Rename newly promoted teams to generic "Promoted"



for index, row in new_teams.iterrows():

    for index1, row1 in new_team_matches.iterrows():

        if (row1['home_team'] == row['team']) & (row1['season'] == row['season']):

            new_team_matches.loc[index1, 'home_team'] = 'Promoted'

        if (row1['away_team'] == row['team']) & (row1['season'] == row['season']):

            new_team_matches.loc[index1, 'away_team'] = 'Promoted'



#    new_team_matches = pd.concat([new_team_matches,matches.loc[((matches['home_team'] == row['team']) |

#                                                  (matches['away_team'] == row['team'])) &

#                                                   (matches['season'] == row['season']) ]])

new_team_matches
# Remove season 15/16 into a separate test set if testing against this

test_matches = matches[matches.season == "2015/2016"]

# If predicting this season (16/17), leave in 15/16 for the model - comment out

# matches = matches[matches.season != "2015/2016"]



# Set up the test matches, renaming new teams to 'Promoted' to use the new team percentages

# 15/16 new_teams = ['Watford', 'Bournemouth', 'Leicester']

# new teams for "2016/2017" are 'Burnley', 'Middlesbrough', 'Hull'

this_season_new_teams = pd.DataFrame({'team':['Watford', 'Bournemouth', 'Leicester City']})



for index, row in this_season_new_teams.iterrows():

    for index1, row1 in test_matches.iterrows():

        if row1['home_team'] == row['team']:

            test_matches.loc[index1, 'home_team'] = 'Promoted'

        if row1['away_team'] == row['team']:

            test_matches.loc[index1, 'away_team'] = 'Promoted'



test_matches = test_matches.reset_index(drop=True)

#test_matches

# this_season_new_teams
# Function to determine whether result is a win/draw/lose

# Passed the test_match dataframe, returns result



def determine_home_result(match):

    if match['home_team_goal'] > match['away_team_goal']:

        return 'win'

    elif match['home_team_goal'] < match['away_team_goal']:

        return 'lose'

    else:

        return 'draw'





# Set the home team result based on score. Will be used to compare predictions

test_matches['home_team_result'] = test_matches.apply(determine_home_result, axis=1)

test_matches

# matches
''' Included for info as a part of my journey

# First look to state the result as home or away win/draw/lose - 6 possibilites

# Here is an experiment with using a function and apply to get a single column with

# result but it didn't lend itself to further manipulation



# Functions to determine whether result is a win/draw/lose

# Passed the match dataframe, returns result



def determine_home_result(match):

    if match['home_team_goal'] > match['away_team_goal']:

        return 'win'

    elif match['home_team_goal'] < match['away_team_goal']:

        return 'loss'

    else:

        return 'draw'





def determine_away_result(match):

    if match['away_team_goal'] > match['home_team_goal']:

        return 'win'

    elif match['away_team_goal'] < match['home_team_goal']:

        return 'loss'

    else:

        return 'draw'





matches['home_team_result'] = matches.apply(determine_home_result, axis=1)

matches['away_team_result'] = matches.apply(determine_away_result, axis=1)

'''
# First look to state the result as home or away win/draw/lose - 6 possibilites

# Create a binary result

def determine_result(match_list):

    match_list['home_win'] = np.where(match_list['home_team_goal'] > match_list['away_team_goal'], 1, 0)

    match_list['home_draw'] = np.where(match_list['home_team_goal'] == match_list['away_team_goal'], 1, 0)

    match_list['home_lose'] = np.where(match_list['home_team_goal'] < match_list['away_team_goal'], 1, 0)

    match_list['away_win'] = np.where(match_list['home_team_goal'] < match_list['away_team_goal'], 1, 0)

    match_list['away_draw'] = np.where(match_list['home_team_goal'] == match_list['away_team_goal'], 1, 0)

    match_list['away_lose'] = np.where(match_list['home_team_goal'] > match_list['away_team_goal'], 1, 0)



determine_result(matches)

matches
''' Included for info as a part of my journey

Looked at various ways of getting these results, some of the code I fiddled with:

# Creating a team list first

team = matches.ix[:,7]

team.drop_duplicates(inplace=True)



# Count number of HW's where home_team is same as team name

# tmp2 = pd.value_counts(matches['home_team_result'] [])

# tmp2 = matches.groupby(['home_team','home_team_result'])['home_team_result'].count()



That finally lead to group by and aggregation

'''



# Next create a new dataframe with a row for team and totals for results:

# Team | Home Win | HD | HL | AW | AD | AL

def aggregate_team_results(match_list):

    # First aggregate the home matches

    team_res_agg = match_list.groupby(['home_team']).agg({'home_win':sum,

                                                           'home_draw':sum,

                                                           'home_lose':sum

                                                          })



    # Now append the away scores for the same team

    team_res_agg = pd.concat([team_res_agg,

                                  match_list.groupby(['away_team']).agg({'away_win':sum,

                                                                      'away_draw':sum,

                                                                      'away_lose':sum

                                                                     })

                                 ], axis=1).reset_index()



    # type(team_results_agg)

    team_res_agg.rename(columns={'index':'team'}, inplace=True)

    return team_res_agg

    

    

team_results_agg = aggregate_team_results(matches)

team_results_agg
# Now create the model for promoted teams and aggregate to a single team. Then add to

# the main model



# First look to state the result as home or away win/draw/lose - 6 possibilites

# Create a binary result



new_team_matches['home_win'] = np.where(new_team_matches['home_team_goal'] > new_team_matches['away_team_goal'], 1, 0)

new_team_matches['home_draw'] = np.where(new_team_matches['home_team_goal'] == new_team_matches['away_team_goal'], 1, 0)

new_team_matches['home_lose'] = np.where(new_team_matches['home_team_goal'] < new_team_matches['away_team_goal'], 1, 0)

new_team_matches['away_win'] = np.where(new_team_matches['home_team_goal'] < new_team_matches['away_team_goal'], 1, 0)

new_team_matches['away_draw'] = np.where(new_team_matches['home_team_goal'] == new_team_matches['away_team_goal'], 1, 0)

new_team_matches['away_lose'] = np.where(new_team_matches['home_team_goal'] > new_team_matches['away_team_goal'], 1, 0)



# Aggregate the home matches

new_team_results_agg = new_team_matches.groupby(['home_team']).agg({'home_win':sum,

                                                       'home_draw':sum,

                                                       'home_lose':sum

                                                      })



# Now append the away scores for the same team

new_team_results_agg = pd.concat([new_team_results_agg,

                              new_team_matches.groupby(['away_team']).agg({'away_win':sum,

                                                                  'away_draw':sum,

                                                                  'away_lose':sum

                                                                 })

                             ], axis=1).reset_index()



new_team_results_agg.rename(columns={'index':'team'}, inplace=True)



# Pull out the Promoted team row and add to main model

single_new_team_results_agg = new_team_results_agg.loc[new_team_results_agg['team'] == 'Promoted']

team_results_agg = pd.concat([team_results_agg, single_new_team_results_agg])

team_results_agg = team_results_agg.reset_index(drop=True)

#team_results_agg
# Now convert these absolute numbers into percentage of the total home or away matches

team_results_agg['home_win_pct'] = team_results_agg['home_win'] / (team_results_agg['home_win'] +

                                                                   team_results_agg['home_draw'] +

                                                                   team_results_agg['home_lose'])

team_results_agg['home_draw_pct'] = team_results_agg['home_draw'] / (team_results_agg['home_win'] +

                                                                   team_results_agg['home_draw'] +

                                                                   team_results_agg['home_lose'])

team_results_agg['home_lose_pct'] = team_results_agg['home_lose'] / (team_results_agg['home_win'] +

                                                                   team_results_agg['home_draw'] +

                                                                   team_results_agg['home_lose'])



team_results_agg['away_win_pct'] = team_results_agg['away_win'] / (team_results_agg['away_win'] +

                                                                   team_results_agg['away_draw'] +

                                                                   team_results_agg['away_lose'])

team_results_agg['away_draw_pct'] = team_results_agg['away_draw'] / (team_results_agg['away_win'] +

                                                                   team_results_agg['away_draw'] +

                                                                   team_results_agg['away_lose'])

team_results_agg['away_lose_pct'] = team_results_agg['away_lose'] / (team_results_agg['away_win'] +

                                                                   team_results_agg['away_draw'] +

                                                                   team_results_agg['away_lose'])



team_results_agg



# team_results_agg_pct =  lambda x: x / team_results_agg.sum(axis=1)
#test_matches
# Get a list of the matches for the season

predict_results = test_matches[['home_team', 'away_team']]





# new teams for "2016/2017" are 'Burnley', 'Middlesbrough', 'Hull'

# Or predict this week's matches 

#this_week_data = {'home_team':['Bournemouth', 'Arsenal', 'Promoted', 'Chelsea', 'Promoted', 

#                               'Leicester City', 'Liverpool', 'Manchester City', 'Swansea City', 'West Ham United'],

#                   'away_team':['Tottenham Hotspur', 'Promoted', 'Everton', 'Manchester United', 'Stoke City', 

#                                'Crystal Palace', 'West Bromwich Albion', 'Southampton', 'Watford', 'Sunderland']}





this_week_matches = pd.DataFrame(this_week_data,

                                 columns=['home_team', 'away_team'])

this_week_matches

predict_results = this_week_matches[['home_team', 'away_team']]





#predict_results
# Build dataframe for the home team results model

home_results = team_results_agg[['team', 'home_win_pct', 

                                  'home_draw_pct', 'home_lose_pct']]

home_results.columns=['team', 'win', 'draw', 'lose']



#home_results
# Build dataframes for the away teams results model

away_results = team_results_agg[['team', 'away_win_pct', 

                                  'away_draw_pct', 'away_lose_pct']]

away_results.columns=['team', 'win', 'draw', 'lose']



#away_results
# 1. Append W/D/L for home team

predict_results = pd.merge(left=predict_results, right=home_results, how='left', left_on='home_team', right_on='team')

predict_results = predict_results.drop(['team'], axis=1)

#matches.rename(columns={'id_x':'match_id', 'team_long_name':'home_team'}, inplace=True)

predict_results



# 2. Get W/D/L for away team

predict_results = pd.merge(left=predict_results, right=away_results, how='left', left_on='away_team', right_on='team')

predict_results = predict_results.drop(['team'], axis=1)



# 3. Take average of the two for - remember a home win is an away lose

predict_results['win'] = (predict_results['win_x'] + predict_results['lose_y']) / 2

predict_results['draw'] = (predict_results['draw_x'] + predict_results['draw_y']) / 2

predict_results['lose'] = (predict_results['lose_x'] + predict_results['win_y']) / 2

predict_results = predict_results.drop(['win_x', 'draw_x', 'lose_x', 'win_y', 'draw_y', 'lose_y'], axis=1)

#predict_results
# Function to determine whether the highest prediction is for win/draw/lose



def predict_home_result(match):

    if (match['win'] >= match['draw']) & (match['win'] >= match['lose']):

        return 'win' # Favour a home win if probability equal

    elif (match['lose'] > match['win']) & (match['lose'] > match['draw']):

        return 'lose'

    else:

        return 'draw'







predict_results['home_team_result'] = predict_results.apply(predict_home_result, axis=1)

predict_results
# Straight comparison - count of equal / total to get %

#tmp_score = 

(predict_results[predict_results['home_team_result'] 

                 == test_matches['home_team_result']].count()) / test_matches['home_team_result'].count()

#tmp_score
# Get the actual result into the prediction table...

predict_results['actual_result'] = test_matches['home_team_result']

predict_results
'''

# Just check a set of all win predictions

predict_results['home_team_result'] = 'win'

predict_results

'''
#(predict_results[predict_results['home_team_result'] 

#                 == test_matches['home_team_result']].count()) / test_matches['home_team_result'].count()
# What result gets predicted better?

predict_correct = predict_results[['home_team_result', 'actual_result']][predict_results['actual_result'] == predict_results['home_team_result']]

predict_analysis = predict_correct.groupby('home_team_result').count()

predict_total = predict_results[['home_team_result', 'actual_result']]

predict_analysis['Total'] = predict_total.groupby('home_team_result').count()

predict_analysis['% Correct'] = predict_analysis['actual_result'] / predict_analysis['Total']



predict_analysis

# predict_total

# predict_correct

# predict_win
actuals = predict_results[predict_results['actual_result'] == 'win']

#actuals
test_f = pd.DataFrame(full_matches['season'],columns=['season'])

#test_run_results = pd.DataFrame(data, columns=['Test Description','Summary Percentage'])

test_f = test_f.loc[test_f['season'] != '2015/2016']



test_f.shape

#full_matches.shape
subset_matches = matches.loc[(matches['home_team'] == 'Arsenal') & (matches['away_team'] == 'Liverpool')]

# matches['home_draw'].loc[(matches['home_team'] == 'Arsenal') & (matches['away_team'] == 'Liverpool')]

# targets = np.array(matches['home_draw'].loc[(matches['home_team'] == 'Arsenal') & (matches['away_team'] == 'Liverpool')])

# targets
#test_matches['home_team_result'].values
# Convert home & team into a binary feature, ie Arsenal_h or Arsenal_a

match_features = pd.get_dummies(matches['home_team']).rename(columns=lambda x: str(x) + '_h')

match_features = pd.concat([match_features, pd.get_dummies(matches['away_team']).rename(columns=lambda x: str(x) + '_a')],

                         axis=1)

#match_features
# home matches v Liverpool - to train model

a_l_matches = match_features.loc[(match_features['Arsenal_h'] == 1) & (match_features['Liverpool_a'] == 1)]

# a_l_matches.drop(['match_id', 'season', 'stage', 'date', 'match_api_id', 'home_team_goal', 'away_team_goal', 

#                  'home_team', 'away_team', 'away_win', 'away_draw', 'away_lose'], axis=1, inplace=True)





# Create target results (W/D/L)

# targets = np.array([0,1,0,0,0,1,1,0]) # win

# targets = np.array(matches['home_draw'].loc[(matches['home_team'] == 'Arsenal') & (matches['away_team'] == 'Liverpool')])

# targets = np.array(matches['home_lose'].loc[(matches['home_team'] == 'Arsenal') & (matches['away_team'] == 'Liverpool')])

#targets = np.array([1,0,1,2,1,0,0,1]) 

targets = np.array(['draw','win','draw','lose','draw','win','win'])

x = a_l_matches.values



# def run_model(x, targets):

# Train, then predict

model = MultinomialNB()

model.fit(x, targets)



#test_arsenal = a_l_matches.values

test_arsenal = a_l_matches.iloc[0].values

#test_arsenal.reshape(1, -1)

# test_arsenal

# predicted = model.predict(test_arsenal)

predicted = model.predict_proba(test_arsenal)

predicted



#metrics.classification_report(targets, predicted)

#metrics.confusion_matrix(targets, predicted)

# run_model(x, targets)
# Set up the matches data how I need it



# Add binary feature for W/D/L home and away

determine_result(full_matches)



# Sort in date order

full_matches.sort_values(by='date', inplace=True)



full_matches
# Cater for new teams by setting the new team for that season to a generic name

# Change home team names first, then away teams

def set_promoted_teams(new_team_list, match_list):

    for index, row in new_team_list.iterrows():

        for index1, row1 in match_list.iterrows():

            if (row1['home_team'] == row['team']) & (row1['season'] == row['season']):

                match_list.loc[index1, 'home_team'] = 'Promoted'

            if (row1['away_team'] == row['team']) & (row1['season'] == row['season']):

                match_list.loc[index1, 'away_team'] = 'Promoted'

    
# Create a regression type model for each season, for each team:

# Build a W/D/L average for a season, and as the new season progresses, subtract the results

# from the average model. So if a team normally has 18 wins in a season, and they have had 9,

# then their chances of more wins will reduce (as they regress to their normal pattern).

# 

# First, build an average season for each team based on at least 3 seasons

# Then use this for the whole of the following season, recalculating the current regression

# after each match result.



# Build a regression model for each subsequent season after the first 3



regr_model_seasons = pd.DataFrame(regress_seasons, columns=['season'])

regr_model_remain_seasons = pd.DataFrame(remaining_seasons, columns=['season'])

regr_model_remain_seasons = regr_model_remain_seasons.reset_index(drop=True)

regr_model_team_results_agg = pd.DataFrame()



# Build a model for each season, then add to the main dataframe

for i, row in regr_model_remain_seasons.iterrows():

    # Select the relevant matches

    regr_model_matches = matches.loc[matches['season'].isin(regr_model_seasons['season'])]

    regr_model_this_seas_team_results_agg = aggregate_team_results(regr_model_matches)



    # Do the same for new teams

    regr_model_new_teams = new_teams.loc[new_teams['season'].isin(regr_model_seasons['season'])]

    regr_model_new_team_matches = create_new_team_matches(regr_model_matches, regr_model_new_teams)



    set_promoted_teams(regr_model_new_teams, regr_model_new_team_matches)

    regr_model_new_team_results_agg = aggregate_team_results(regr_model_new_team_matches)

    

    # Aggregate promoted and divide the results by 3 for promoted teams

    regr_model_single_new_team_results_agg = regr_model_new_team_results_agg.loc[regr_model_new_team_results_agg['team'] == 'Promoted']

    regr_model_single_new_team_results_agg = (regr_model_single_new_team_results_agg.iloc[:,1:7].applymap(lambda x: x/3))

    regr_model_single_new_team_results_agg.insert(0, 'team', 'Promoted')



    #, aggregate and add

    regr_model_this_seas_team_results_agg = pd.concat([regr_model_this_seas_team_results_agg, regr_model_single_new_team_results_agg])

    regr_model_this_seas_team_results_agg = regr_model_this_seas_team_results_agg.reset_index(drop=True)



    # Finalise the model by creating the average number of games per season

    # No rounding at this stage, but accuracy should be ok

    number_seasons = len(regr_model_seasons)

    regr_model_team_names = pd.DataFrame(regr_model_this_seas_team_results_agg['team'])

    regr_model_this_seas_team_results_agg = (regr_model_this_seas_team_results_agg.iloc[:,1:7].applymap(lambda x: x/number_seasons))

    regr_model_this_seas_team_results_agg = pd.concat([regr_model_team_names, regr_model_this_seas_team_results_agg], axis=1)

    regr_model_this_seas_team_results_agg['season'] = regr_model_remain_seasons['season'].ix[i]

    

    # Add this season's values to main table

    regr_model_team_results_agg = pd.concat([regr_model_team_results_agg, regr_model_this_seas_team_results_agg])

    regr_model_team_results_agg = regr_model_team_results_agg.reset_index(drop=True)

    

    # Add to season list at bottom of loop

    regr_model_seasons = regr_model_seasons.append(regr_model_remain_seasons.ix[i])



regr_model_team_results_agg
regr_model_team_results_agg.loc[regr_model_team_results_agg['team'] == 'Promoted']
# Build seprate dataframes for the home and away teams

regr_model_team_results_agg

regr_model_home_team_results_agg = regr_model_team_results_agg[['season', 'team', 'home_win', 'home_draw', 'home_lose']]

regr_model_home_team_results_agg.columns=['r_season', 'r_team', 'r_h_win', 'r_h_draw', 'r_h_lose']



regr_model_away_team_results_agg = regr_model_team_results_agg[['season', 'team', 'away_win', 'away_draw', 'away_lose']]

regr_model_away_team_results_agg.columns=['r_season', 'r_team', 'r_a_win', 'r_a_draw', 'r_a_lose']



regr_model_home_team_results_agg
# Add regression season figures, then calculate, first home, then away

# Create a regression features table

regr_features = full_matches.copy()

set_promoted_teams(new_teams, regr_features)



# Ceate a df of unique teams

unique_teams = pd.DataFrame(full_matches.home_team.unique(), columns=['team'])



regr_home_features = pd.merge(left=regr_features, right=regr_model_home_team_results_agg, how='left', left_on=['home_team', 'season'], right_on=['r_team', 'r_season'])

regr_home_features.sort_values(by=['home_team', 'date'], inplace=True)

regr_home_features.reset_index(drop=True, inplace=True)





# Set up home regression figures

previous_season = 'blank'

previous_team = 'blank'



for i, row in regr_home_features.iterrows():

    if (row['season'] != previous_season) or (row['home_team'] != previous_team):

        # New season or team

        previous_season = row['season']

        previous_team = row['home_team']

    else:

        # Wins

        if regr_home_features.ix[i-1, 'home_win'] == 1:

            regr_home_features.ix[i, 'r_h_win'] = regr_home_features.ix[i-1, 'r_h_win'] - 1

        else:

            regr_home_features.ix[i, 'r_h_win'] = regr_home_features.ix[i-1, 'r_h_win']

        # Draws

        if regr_home_features.ix[i-1, 'home_draw'] == 1:

            regr_home_features.ix[i, 'r_h_draw'] = regr_home_features.ix[i-1, 'r_h_draw'] - 1

        else:

            regr_home_features.ix[i, 'r_h_draw'] = regr_home_features.ix[i-1, 'r_h_draw']

        # Losses

        if regr_home_features.ix[i-1, 'home_lose'] == 1:

            regr_home_features.ix[i, 'r_h_lose'] = regr_home_features.ix[i-1, 'r_h_lose'] - 1

        else:

            regr_home_features.ix[i, 'r_h_lose'] = regr_home_features.ix[i-1, 'r_h_lose']



def replace_negatives(value):

    if value < 0:

        return 0

    else:

        return value

    

    

regr_home_features['r_h_win'] = regr_home_features['r_h_win'].apply(replace_negatives) 

regr_home_features['r_h_draw'] = regr_home_features['r_h_draw'].apply(replace_negatives) 

regr_home_features['r_h_lose'] = regr_home_features['r_h_lose'].apply(replace_negatives) 



regr_home_features

#regr_features 

#tmp_idx = (regr_model_home_team_results_agg['r_team'] == 'Arsenal') & (regr_model_home_team_results_agg['r_season'] == '2016/2017')
# Set up away regression figures

regr_away_features = pd.merge(left=regr_features, right=regr_model_away_team_results_agg, how='left', left_on=['away_team', 'season'], right_on=['r_team', 'r_season'])

regr_away_features.sort_values(by=['away_team', 'date'], inplace=True)

regr_away_features.reset_index(drop=True, inplace=True)

previous_season = 'blank'

previous_team = 'blank'



for i, row in regr_away_features.iterrows():

    if (row['season'] != previous_season) or (row['away_team'] != previous_team):

        # New season or team

        previous_season = row['season']

        previous_team = row['away_team']

    else:

        # Wins

        if regr_away_features.ix[i-1, 'away_win'] == 1:

            regr_away_features.ix[i, 'r_a_win'] = regr_away_features.ix[i-1, 'r_a_win'] - 1

        else:

            regr_away_features.ix[i, 'r_a_win'] = regr_away_features.ix[i-1, 'r_a_win']

        # Draws

        if regr_away_features.ix[i-1, 'away_draw'] == 1:

            regr_away_features.ix[i, 'r_a_draw'] = regr_away_features.ix[i-1, 'r_a_draw'] - 1

        else:

            regr_away_features.ix[i, 'r_a_draw'] = regr_away_features.ix[i-1, 'r_a_draw']

        # Losses

        if regr_away_features.ix[i-1, 'away_lose'] == 1:

            regr_away_features.ix[i, 'r_a_lose'] = regr_away_features.ix[i-1, 'r_a_lose'] - 1

        else:

            regr_away_features.ix[i, 'r_a_lose'] = regr_away_features.ix[i-1, 'r_a_lose']



regr_away_features['r_a_win'] = regr_away_features['r_a_win'].apply(replace_negatives) 

regr_away_features['r_a_draw'] = regr_away_features['r_a_draw'].apply(replace_negatives) 

regr_away_features['r_a_lose'] = regr_away_features['r_a_lose'].apply(replace_negatives) 

regr_away_features
# Convert to a percentage

regr_home_features['r_h_w_pct'] = regr_home_features['r_h_win'] / (regr_home_features['r_h_win'] + 

                                                                   regr_home_features['r_h_draw'] +

                                                                   regr_home_features['r_h_lose'])

regr_home_features['r_h_d_pct'] = regr_home_features['r_h_draw'] / (regr_home_features['r_h_win'] + 

                                                                   regr_home_features['r_h_draw'] +

                                                                   regr_home_features['r_h_lose'])



regr_home_features['r_h_l_pct'] = regr_home_features['r_h_lose'] / (regr_home_features['r_h_win'] + 

                                                                   regr_home_features['r_h_draw'] +

                                                                   regr_home_features['r_h_lose'])



regr_home_features = regr_home_features.fillna(0)

regr_home_features
# Convert away to a percentage

regr_away_features['r_a_w_pct'] = regr_away_features['r_a_win'] / (regr_away_features['r_a_win'] + 

                                                                   regr_away_features['r_a_draw'] +

                                                                   regr_away_features['r_a_lose'])

regr_away_features['r_a_d_pct'] = regr_away_features['r_a_draw'] / (regr_away_features['r_a_win'] + 

                                                                   regr_away_features['r_a_draw'] +

                                                                   regr_away_features['r_a_lose'])

regr_away_features['r_a_l_pct'] = regr_away_features['r_a_lose'] / (regr_away_features['r_a_win'] + 

                                                                   regr_away_features['r_a_draw'] +

                                                                   regr_away_features['r_a_lose'])

regr_away_features = regr_away_features.fillna(0)

regr_away_features
#regr_away_features = regr_away_features.fillna(0)

#regr_away_features[regr_away_features.isnull().any(axis=1)]
# Tidy up

final_regr_home_features = regr_home_features[['match_id', 'r_h_w_pct', 'r_h_d_pct', 'r_h_l_pct']]

final_regr_away_features = regr_away_features[['match_id', 'r_a_w_pct', 'r_a_d_pct', 'r_a_l_pct']]

final_regr_away_features.head()
#final_regr_away_features[final_regr_away_features.isnull().any(axis=1)]

#regr_away_features.loc[regr_away_features['match_id'] == 4387]
# Build a separate features lookup df focused on each team. Can calculate various features:

# - Streak - Need to look at a team's run regardless of home or away

# - Goal Difference



# Loop through team list

team_features = pd.DataFrame()

for index, row in unique_teams.iterrows():

    # Then create a mask for that team - an index - on matches

    # Use that index to select all matches for a team

    single_team_matches_idx = (full_matches['home_team'] == row['team']) | (full_matches['away_team'] == row['team'])

    single_team_result = full_matches.loc[single_team_matches_idx]

                                          #, ['home_team', 'match_id', 'season', 'date',

                                                                    #'home_team_goal', 'away_team_goal',

                                                                    #'home_win', 'home_draw', 'home_lose',

                                                                    #'away_win', 'away_draw', 'away_lose']]

    

    # Create the structure for streaks

    single_team_result['streak_team'] = row['team']

    team_features = pd.concat([team_features, single_team_result])



team_features.sort_values(by=['streak_team', 'date'], inplace=True)

team_features.reset_index(drop=True, inplace=True)



team_features
#team_features.loc[team_features['match_id'] == 2493]

                   #'2015/2016') & (team_features['streak_team'] == 'Leicester City')]

#team_features.loc[team_features['streak_team'] == 'Arsenal']

#unique_teams
# Unused code - part of my journey!!

def create_streak(matches_df):

    # Build a streak based on a set of matches and the result

    # [TODO to generalise, pass dataframe and column working on]



    # Need to 'reset' the streak each time zero is found

    matches_df['c'] = (matches_df['home_win'] == 0).cumsum() # Streak of Not winning 

    matches_df['a'] = (matches_df['c'] == 0).astype(int) # Give a value of 1

    matches_df['b'] = matches_df.groupby( 'c' ).cumcount()



    matches_df['streak'] = matches_df['b'] + matches_df['a']



    return matches_df





#single_team_matches = single_team_matches.groupby('home_team', sort=False).apply(create_streak)



def streak_team_result(match):

    if match['home_team'] == match['streak_team']:

        match['win'] = match['home_win']

        match['draw'] = match['home_draw']

        match['lose'] = match['home_lose']

    else:

        match['win'] = match['away_win']

        match['draw'] = match['away_draw']

        match['lose'] = match['away_lose']



    return match





#team_features = team_features.apply(streak_team_result, axis=1)

# team_features
# For each season create features, and reset at new season

# Depending on if team is home or away set appropriate streak team's W/D/L & goal difference

# Calculate features from this week's score but record against next week.



previous_season = 'blank'

previous_team = 'blank'

last_match_id = team_features['match_id'].iloc[-1]



for i, row in team_features.iterrows():

    if (row['season'] != previous_season) or (row['streak_team'] != previous_team):

        # New season or team, set first row to zero

        team_features.ix[i, 'win_streak'] = 0

        team_features.ix[i, 'draw_streak'] = 0

        team_features.ix[i, 'lose_streak'] = 0

        team_features.ix[i, 'goal_diff'] = 0

        previous_season = row['season']

        previous_team = row['streak_team']



    if row['match_id'] != last_match_id: # Only update next row if not last row of table

        # Check home result

        if row['home_team'] == row['streak_team']:

            # Wins

            if row['home_win'] == 1:

                team_features.ix[i+1, 'win_streak'] = team_features.ix[i, 'win_streak'] + 1

            else:

                team_features.ix[i+1, 'win_streak'] = 0

            # Draws

            if row['home_draw'] == 1:

                team_features.ix[i+1, 'draw_streak'] = team_features.ix[i, 'draw_streak'] + 1

            else:

                team_features.ix[i+1, 'draw_streak'] = 0

            # Losses

            if row['home_lose'] == 1:

                team_features.ix[i+1, 'lose_streak'] = team_features.ix[i, 'lose_streak'] + 1

            else:

                team_features.ix[i+1, 'lose_streak'] = 0



            team_features.ix[i+1, 'goal_diff'] = team_features.ix[i, 'goal_diff'] + (team_features.ix[i, 'home_team_goal'] -

                                                                                     team_features.ix[i, 'away_team_goal'])            

        

        else: # An away result

            # Wins

            if row['away_win'] == 1:

                team_features.ix[i+1, 'win_streak'] = team_features.ix[i, 'win_streak'] + 1

            else:

                team_features.ix[i+1, 'win_streak'] = 0

            # Draws

            if row['away_draw'] == 1:

                team_features.ix[i+1, 'draw_streak'] = team_features.ix[i, 'draw_streak'] + 1

            else:

                team_features.ix[i+1, 'draw_streak'] = 0

            # Losses

            if row['away_lose'] == 1:

                team_features.ix[i+1, 'lose_streak'] = team_features.ix[i, 'lose_streak'] + 1

            else:

                team_features.ix[i+1, 'lose_streak'] = 0



            team_features.ix[i+1, 'goal_diff'] = team_features.ix[i, 'goal_diff'] + (team_features.ix[i, 'away_team_goal'] -

                                                                                     team_features.ix[i, 'home_team_goal'])            



team_features
# Remove negatives in Goal Difference by scaling to 0 to 1



x = team_features['goal_diff'].values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

team_features['goal_diff'] = pd.DataFrame(x_scaled)

team_features
#team_features.loc[(team_features['season'] == '2011/2012') & (team_features['home_team'] == 'Arsenal')]

#regr_model_home_team_results_agg['r_h_win'].loc[regr_idx]

#team_features['season'].unique()
# Tidy up

team_features = team_features.drop(['season', 'date', 'home_team_goal', 'away_team_goal',

                                    'home_team', 'away_team', 'home_win', 'home_draw', 'home_lose', 

                                    'away_win', 'away_draw', 'away_lose'], axis=1)

team_features.reset_index(drop=True, inplace=True)

team_features
# Add Features as needed

# - both home and away teams need their own streaks and goald diffs

final_team_features = full_matches[['match_id', 'home_team', 'away_team']]



# Add streaks 

team_streaks = team_features.iloc[:,:-1] # Ignore goal difference at this point

final_team_features = pd.merge(left=final_team_features, right=team_streaks, how='left', left_on=['home_team', 'match_id'], right_on=['streak_team', 'match_id'])

final_team_features = pd.merge(left=final_team_features, right=team_streaks, how='left', left_on=['away_team', 'match_id'], right_on=['streak_team', 'match_id'], suffixes=('_home', '_away'))

#final_team_features = final_team_features.drop(['streak_team_home', 'streak_team_away', 'stage_home', 'stage_away'], axis=1)



final_team_features
team_streaks
# Add goal diff



team_goal_diffs = team_features.copy()

team_goal_diffs = team_goal_diffs.drop(['win_streak', 'draw_streak', 'lose_streak'], axis=1)

final_team_features = pd.merge(left=final_team_features, right=team_goal_diffs, how='left', left_on=['home_team', 'match_id'], right_on=['streak_team', 'match_id'])

final_team_features = pd.merge(left=final_team_features, right=team_goal_diffs, how='left', left_on=['away_team', 'match_id'], right_on=['streak_team', 'match_id'], suffixes=('_home', '_away'))

final_team_features = final_team_features.drop(['home_team', 'away_team', 'streak_team_home', 'streak_team_away', 'stage_home', 'stage_away'], axis=1)



#final_team_features = final_team_features.drop(['home_team', 'away_team'], axis=1)

final_team_features
# Add regression

final_team_features = pd.merge(left=final_team_features, right=final_regr_home_features, how='left', left_on=['match_id'], right_on=['match_id'])

final_team_features = pd.merge(left=final_team_features, right=final_regr_away_features, how='left', left_on=['match_id'], right_on=['match_id'], suffixes=('_home', '_away'))

final_team_features = final_team_features.drop(['match_id'], axis=1)



#final_team_features
final_team_features
# Set the new team for that season to a generic 'Promoted'

set_promoted_teams(new_teams, full_matches)

#full_matches
# Need all seasons for team binary feature

# Convert home & team into a binary feature, ie Arsenal_h or Arsenal_a

full_match_features = pd.DataFrame(full_matches[['season', 'stage']],

                                   columns=[['season', 'stage']])



full_match_features = pd.concat([full_match_features, pd.get_dummies(full_matches['home_team']).rename(columns=lambda x: str(x) + '_h')],

                                axis=1)

full_match_features = pd.concat([full_match_features, pd.get_dummies(full_matches['away_team']).rename(columns=lambda x: str(x) + '_a')],

                         axis=1)



# Add a draw indicator for the most likely week of draws - week 24

#full_match_features['draw_ind'] = [1 if x == 24 else 0 for x in full_match_features['stage']]





# [This didn't improve the score initially, but keep trying it...]

# Convert Stage (week number) into a group with 8 week intervals

# Roughly equates to mid Aug-Oct, Oct-Dec, Dec-Feb, Feb-Apr, Apr-May

bins = [0, 8, 16, 25, 32, 40]

bin_names = ['start', 'early', 'xmas', 'latter', 'end']

full_match_features['bin'] = pd.cut(full_match_features['stage'], bins, labels=bin_names)

# Convert to a binary feature

full_match_features = pd.concat([full_match_features, pd.get_dummies(full_match_features['bin']).rename(columns=lambda x: str(x))],

                         axis=1)

full_match_features = full_match_features.drop(['bin'], axis=1)



# Try a binary for season identifier

full_match_features = pd.concat([full_match_features, pd.get_dummies(full_matches['season'])], axis=1)



# Add streak features and Goal Diff 

final_team_features.reset_index(drop=True, inplace=True)

full_match_features.reset_index(drop=True, inplace=True)

full_match_features = pd.concat([full_match_features, final_team_features], axis=1)



full_match_features
full_match_features.columns
# Parameters to change depending on season and wek we are running for

this_season = '2016/2017'

this_week = 14



# If predicting this season (16/17), remove latest week from training set

train_match_features = full_match_features.loc[(full_match_features['season'] != this_season) | 

                                              (full_match_features['season'] == this_season) &

                                                  (full_match_features['stage'] < this_week)].copy()



# Or Remove 15/16 from training set

#train_match_features = full_match_features.loc[full_match_features['season'] != '2015/2016'].copy()



train_match_features.drop(['season'], axis=1, inplace=True)

train_match_features
train_match_features[2030:2060]
# Get all the target results for training

# First Add the home team result column to the matches dataframe

full_matches['home_team_result'] = full_matches.apply(determine_home_result, axis=1)



# If predicting this season (16/17), remove latest week from training results

train_matches = full_matches.loc[(full_matches['season'] != this_season) | 

                                              (full_matches['season'] == this_season) &

                                                  (full_matches['stage'] < this_week)].copy()



# Or Remove 15/16 from training set

#train_matches = full_matches.loc[full_matches['season'] != '2015/2016'].copy()



targets = train_matches['home_team_result'].values

train_matches
# Get the test matches in correct format:



# Just a week of match data if predicting this season (16/17)

test_match_features = full_match_features.loc[(full_match_features['season'] == this_season) &

                                              (full_match_features['stage'] == this_week)].copy()



# Or 15/16 matches from training set

#test_match_features = full_match_features.loc[full_match_features['season'] == '2015/2016'].copy()



test_match_features.drop(['season'], axis=1, inplace=True)

test_match_features
# Get all the test results for comparison

# If predicting this season (16/17)

model_test_matches = full_matches.loc[(full_matches['season'] == this_season) &

                                      (full_matches['stage'] == this_week)].copy()



# Or 15/16 matches

#model_test_matches = full_matches.loc[full_matches['season'] == '2015/2016'].copy()



model_test_matches = model_test_matches.reset_index(drop=True)

#model_test_matches
#model_test_matches
# Train, then predict on a variety of models - comment out the ones you don't need

#model = MultinomialNB()

model = LogisticRegression(C=2)

#model = SVC(kernel = 'linear', C=1.5, probability=True)

#model = BernoulliNB()



#model = RandomForestClassifier(oob_score=True, n_estimators=15000)

#model = KNeighborsClassifier()

#model = DecisionTreeClassifier()

#model = GaussianNB()



model.fit(train_match_features.values, targets)

predicted = model.predict_proba(test_match_features.values)

# predicted



# Format the output into a DF with columns

predicted_table = pd.DataFrame(predicted,columns=['draw', 'lose', 'win'])



# Compare predicted with test actual results 

predicted_table['predict_res'] = predicted_table.apply(predict_home_result, axis=1)

predicted_table['actual_res'] = model_test_matches['home_team_result']

# predicted_table



# Straight comparison - count of equal / total to get %

#tmp_score = 

(predicted_table[predicted_table['predict_res'] 

                 == model_test_matches['home_team_result']].count()) / model_test_matches['home_team_result'].count()

#tmp_score
compare_results = model_test_matches[['match_id', 'stage', 'home_team_goal', 

                                    'away_team_goal', 'home_team', 'away_team']].copy()

compare_results.rename(columns={'home_team_goal':'h_goal', 'away_team_goal':'a_goal'}, inplace=True)

compare_results = pd.concat([compare_results, predicted_table], axis=1)

compare_results
#predicted_table
# What result gets predicted better?

model_predict_correct = predicted_table[['predict_res', 'actual_res']][predicted_table['actual_res'] == predicted_table['predict_res']]

model_predict_analysis = model_predict_correct.groupby('predict_res').count()

model_predict_total = predicted_table[['predict_res', 'actual_res']]

model_predict_analysis['Total'] = model_predict_total.groupby('predict_res').count()

model_predict_analysis['Test_total'] = model_predict_total.groupby('actual_res').count()

model_predict_analysis['Precision'] = model_predict_analysis['actual_res'] / model_predict_analysis['Total']

model_predict_analysis['Recall'] = model_predict_analysis['actual_res'] / model_predict_analysis['Test_total']

model_predict_analysis.loc['TOTAL']= model_predict_analysis.sum()



model_predict_analysis

# predict_total

# predict_correct

# predict_win
# I need to save all the various results into a persistent table but for now will

# Manually record various rest runs and results

data = [{'Test Description':"Season 13-15, first run, with error",

         'Summary Percentage':.356

        },

        {'Test Description':"Season 13-15, second run, fixed bug",

         'Summary Percentage':.445

        },

        {'Test Description':"Season 13-15, all home wins",

         'Summary Percentage':.419

        },

        {'Test Description':"Season 08-15",

         'Summary Percentage':.445

        },

        {'Test Description':"Season 08-15 with new teams model included",

         'Summary Percentage':.4474

        },

        {'Test Description':"Season 08-15, all home wins with new teams included",

         'Summary Percentage':.413

        },

        {'Test Description':"Season 08-15, MultiNB first attempt, game results only",

         'Summary Percentage':.4394

        },

        {'Test Description':"Season 08-15, MultiNB, game results only, with new teams",

         'Summary Percentage':.4632

        },

        {'Test Description':"Season 08-15, MultiNB, game results & streak",

         'Summary Percentage':.40

        },

        {'Test Description':"Season 08-15, Randomforest, 1st attempt, game results & streak",

         'Summary Percentage':.4026

        },

        {'Test Description':"Season 08-15, Randomforest, game results only",

         'Summary Percentage':.4263

        },

        {'Test Description':"Season 08-15, MultiNB, results & stage",

         'Summary Percentage':.4658

        },

        {'Test Description':"Season 08-15, LogRegression, results & stage",

         'Summary Percentage':.4447

        },

        {'Test Description':"Season 08-15, K-NN, results & stage",

         'Summary Percentage':.4132

        },

        {'Test Description':"Season 08-15, Decision Tree, results & stage",

         'Summary Percentage':.3763

        },

        {'Test Description':"Season 08-15, SVM, results & stage",

         'Summary Percentage':.4210

        },

        {'Test Description':"Season 08-15, BernoulliNB, results & stage",

         'Summary Percentage':.4658

        },

        {'Test Description':"Season 08-15, GaussianNB, results & stage",

         'Summary Percentage':.4342

        },

        {'Test Description':"Season 08-15, MultiNB, res, stage_bins",

         'Summary Percentage':.4500

        },

        {'Test Description':"Season 08-15, MultiNB, res, goal diff",

         'Summary Percentage':.3895

        },

        {'Test Description':"Season 08-15, MultiNB, res, goal diff, streaks",

         'Summary Percentage':.40

        },

        {'Test Description':"Season 08-15, LogRegression, results & goal diff",

         'Summary Percentage':.4526

        },

        {'Test Description':"Season 10-15, SVC, C=1.5, all feats minus bin stage",

         'Summary Percentage':.4763

        },

        {'Test Description':"Season 10-15, Randomforest, all feats minus bin stage",

         'Summary Percentage':.4605

        },

        {'Test Description':"Season 10-15, simple model",

         'Summary Percentage':.4710

        },

        {'Test Description':"Season 11-15, LR, all feats",

         'Summary Percentage':.4763

        }

       ]

test_run_results = pd.DataFrame(data, columns=['Test Description','Summary Percentage'])

test_run_results.sort_values(by='Summary Percentage', ascending=False, inplace=True)

test_run_results
'''

# Rough fitting a RandomForestClassifier to determine feature importance.



model = RandomForestClassifier(oob_score=True, n_estimators=10000)

model.fit(train_match_features.values, targets)

feature_importance = model.feature_importances_



# Make importances relative to max importance

feature_importance = 100.0 * (feature_importance / feature_importance.max())



# Threshold to determine when to drop features. The percentage of the most important feature's importance value

fi_threshold = 1



# Get the indexes of all the features over the importance threshold

important_idx = np.where(feature_importance > fi_threshold)[0]



# Create a list of all the feature names [above the importance threshold]

features_list = train_match_features.columns

important_features = features_list[important_idx]

important_features



# get the sorted indexes of important features

#sorted_idx = np.argsort(feature_importance[important_idx])[::-1]

#print('Features sorted by importance (DESC):\n', important_features[sorted_idx])

'''
draw_count = full_matches.groupby(['season', 'stage']).agg({'home_draw':sum})

#draw_count = draw_count.rename(columns={'index':'stage'})



draw_count2 = draw_count.groupby(level=1).mean()



#team_results_agg = matches.groupby(['home_team']).agg({'home_win':sum,

#                                                       'home_draw':sum,

#                                                       'home_lose':sum

#                                                      })

draw_count2

#.describe()



#plt.plot(draw_count2)

#plt.show()