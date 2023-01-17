# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import sqlalchemy

from sqlalchemy import create_engine # database connection

import sqlite3



from IPython.display import display, clear_output

from sklearn.naive_bayes import MultinomialNB



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Parameters to change depending on season and wek we are running for

this_season = '2017/2018'

this_week = 17
# engine  = create_engine("sqlite:///../input/database.sqlite")

with sqlite3.connect('../input/database.sqlite') as engine:

#    matches = pd.read_sql_query('SELECT * FROM Match where league_id = 1729 and season in ("2010/2011", "2011/2012", "2012/2013", "2013/2014", "2014/2015", "2015/2016");'

#                                          , engine)

    matches = pd.read_sql_query('SELECT * FROM Match where league_id = 1729 ;', engine)





matches = matches[matches.columns[:11]]

teams = pd.read_sql_query('SELECT * FROM Team;', engine)



# Add team names & tidy up

matches = pd.merge(left=matches, right=teams, how='left', left_on='home_team_api_id', right_on='team_api_id')

matches = matches.drop(['country_id','league_id', 'home_team_api_id', 'id_y', 'team_api_id', 'team_short_name'], axis=1)

matches.rename(columns={'id_x':'match_id', 'team_long_name':'home_team'}, inplace=True)

matches = pd.merge(left=matches, right=teams, how='left', left_on='away_team_api_id', right_on='team_api_id')

matches = matches.drop(['id', 'match_api_id', 'away_team_api_id','team_api_id', 'team_short_name'], axis=1)

matches.rename(columns={'id_x':'match_id', 'team_long_name':'away_team'}, inplace=True)



matches.tail()
# Add in this season (16/17) matches - ned to convert to csv at some point

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

{'match_id':6139, 'season':'2016/2017', 'stage':14, 'date':'42709', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Middlesbrough', 'away_team':'Hull City'},

{'match_id':6140, 'season':'2016/2017', 'stage':15, 'date':'42714', 'home_team_goal':3, 'away_team_goal':2, 'home_team':'Watford', 'away_team':'Everton'},

{'match_id':6141, 'season':'2016/2017', 'stage':15, 'date':'42714', 'home_team_goal':3, 'away_team_goal':1, 'home_team':'Arsenal', 'away_team':'Stoke City'},

{'match_id':6142, 'season':'2016/2017', 'stage':15, 'date':'42714', 'home_team_goal':3, 'away_team_goal':2, 'home_team':'Burnley', 'away_team':'Bournemouth'},

{'match_id':6143, 'season':'2016/2017', 'stage':15, 'date':'42714', 'home_team_goal':3, 'away_team_goal':3, 'home_team':'Hull City', 'away_team':'Crystal Palace'},

{'match_id':6144, 'season':'2016/2017', 'stage':15, 'date':'42714', 'home_team_goal':3, 'away_team_goal':0, 'home_team':'Swansea City', 'away_team':'Sunderland'},

{'match_id':6145, 'season':'2016/2017', 'stage':15, 'date':'42714', 'home_team_goal':4, 'away_team_goal':2, 'home_team':'Leicester City', 'away_team':'Manchester City'},

{'match_id':6146, 'season':'2016/2017', 'stage':15, 'date':'42715', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Chelsea', 'away_team':'West Bromwich Albion'},

{'match_id':6147, 'season':'2016/2017', 'stage':15, 'date':'42715', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Manchester United', 'away_team':'Tottenham Hotspur'},

{'match_id':6148, 'season':'2016/2017', 'stage':15, 'date':'42715', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Southampton', 'away_team':'Middlesbrough'},

{'match_id':6149, 'season':'2016/2017', 'stage':15, 'date':'42715', 'home_team_goal':2, 'away_team_goal':2, 'home_team':'Liverpool', 'away_team':'West Ham United'},

{'match_id':6150, 'season':'2016/2017', 'stage':15, 'date':'42717', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Bournemouth', 'away_team':'Leicester City'},

{'match_id':6151, 'season':'2016/2017', 'stage':15, 'date':'42717', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Everton', 'away_team':'Arsenal'},

{'match_id':6152, 'season':'2016/2017', 'stage':15, 'date':'42718', 'home_team_goal':0, 'away_team_goal':3, 'home_team':'Middlesbrough', 'away_team':'Liverpool'},

{'match_id':6153, 'season':'2016/2017', 'stage':15, 'date':'42718', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Sunderland', 'away_team':'Chelsea'},

{'match_id':6154, 'season':'2016/2017', 'stage':15, 'date':'42718', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'West Ham United', 'away_team':'Burnley'},

{'match_id':6155, 'season':'2016/2017', 'stage':15, 'date':'42718', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Crystal Palace', 'away_team':'Manchester United'},

{'match_id':6156, 'season':'2016/2017', 'stage':15, 'date':'42718', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Manchester City', 'away_team':'Watford'},

{'match_id':6157, 'season':'2016/2017', 'stage':15, 'date':'42718', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Stoke City', 'away_team':'Southampton'},

{'match_id':6158, 'season':'2016/2017', 'stage':15, 'date':'42718', 'home_team_goal':3, 'away_team_goal':0, 'home_team':'Tottenham Hotspur', 'away_team':'Hull City'},

{'match_id':6159, 'season':'2016/2017', 'stage':15, 'date':'42718', 'home_team_goal':3, 'away_team_goal':1, 'home_team':'West Bromwich Albion', 'away_team':'Swansea City'},

{'match_id':6160, 'season':'2016/2017', 'stage':16, 'date':'42721', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Crystal Palace', 'away_team':'Chelsea'},

{'match_id':6161, 'season':'2016/2017', 'stage':16, 'date':'42721', 'home_team_goal':3, 'away_team_goal':0, 'home_team':'Middlesbrough', 'away_team':'Swansea City'},

{'match_id':6162, 'season':'2016/2017', 'stage':16, 'date':'42721', 'home_team_goal':2, 'away_team_goal':2, 'home_team':'Stoke City', 'away_team':'Leicester City'},

{'match_id':6163, 'season':'2016/2017', 'stage':16, 'date':'42721', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Sunderland', 'away_team':'Watford'},

{'match_id':6164, 'season':'2016/2017', 'stage':16, 'date':'42721', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'West Ham United', 'away_team':'Hull City'},

{'match_id':6165, 'season':'2016/2017', 'stage':16, 'date':'42721', 'home_team_goal':0, 'away_team_goal':2, 'home_team':'West Bromwich Albion', 'away_team':'Manchester United'},

{'match_id':6166, 'season':'2016/2017', 'stage':16, 'date':'42722', 'home_team_goal':1, 'away_team_goal':3, 'home_team':'Bournemouth', 'away_team':'Southampton'},

{'match_id':6167, 'season':'2016/2017', 'stage':16, 'date':'42722', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Manchester City', 'away_team':'Arsenal'},

{'match_id':6168, 'season':'2016/2017', 'stage':16, 'date':'42722', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Tottenham Hotspur', 'away_team':'Burnley'},

{'match_id':6169, 'season':'2016/2017', 'stage':16, 'date':'42723', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Everton', 'away_team':'Liverpool'},

{'match_id':6170, 'season':'2016/2017', 'stage':17, 'date':'42730', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Watford', 'away_team':'Crystal Palace'},

{'match_id':6171, 'season':'2016/2017', 'stage':17, 'date':'42730', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Arsenal', 'away_team':'West Bromwich Albion'},

{'match_id':6172, 'season':'2016/2017', 'stage':17, 'date':'42730', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Burnley', 'away_team':'Middlesbrough'},

{'match_id':6173, 'season':'2016/2017', 'stage':17, 'date':'42730', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Chelsea', 'away_team':'Bournemouth'},

{'match_id':6174, 'season':'2016/2017', 'stage':17, 'date':'42730', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Leicester City', 'away_team':'Everton'},

{'match_id':6175, 'season':'2016/2017', 'stage':17, 'date':'42730', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Manchester United', 'away_team':'Sunderland'},

{'match_id':6176, 'season':'2016/2017', 'stage':17, 'date':'42730', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Swansea City', 'away_team':'West Ham United'},

{'match_id':6177, 'season':'2016/2017', 'stage':17, 'date':'42730', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Hull City', 'away_team':'Manchester City'},

{'match_id':6178, 'season':'2016/2017', 'stage':17, 'date':'42731', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Liverpool', 'away_team':'Stoke City'},

{'match_id':6179, 'season':'2016/2017', 'stage':17, 'date':'42731', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Southampton', 'away_team':'Tottenham Hotspur'},

{'match_id':6180, 'season':'2016/2017', 'stage':18, 'date':'42734', 'home_team_goal':2, 'away_team_goal':2, 'home_team':'Hull City', 'away_team':'Everton'},

{'match_id':6181, 'season':'2016/2017', 'stage':18, 'date':'42735', 'home_team_goal':4, 'away_team_goal':1, 'home_team':'Burnley', 'away_team':'Sunderland'},

{'match_id':6182, 'season':'2016/2017', 'stage':18, 'date':'42735', 'home_team_goal':4, 'away_team_goal':2, 'home_team':'Chelsea', 'away_team':'Stoke City'},

{'match_id':6183, 'season':'2016/2017', 'stage':18, 'date':'42735', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Leicester City', 'away_team':'West Ham United'},

{'match_id':6184, 'season':'2016/2017', 'stage':18, 'date':'42735', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Manchester United', 'away_team':'Middlesbrough'},

{'match_id':6185, 'season':'2016/2017', 'stage':18, 'date':'42735', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Southampton', 'away_team':'West Bromwich Albion'},

{'match_id':6186, 'season':'2016/2017', 'stage':18, 'date':'42735', 'home_team_goal':0, 'away_team_goal':3, 'home_team':'Swansea City', 'away_team':'Bournemouth'},

{'match_id':6187, 'season':'2016/2017', 'stage':18, 'date':'42735', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Liverpool', 'away_team':'Manchester City'},

{'match_id':6188, 'season':'2016/2017', 'stage':18, 'date':'42736', 'home_team_goal':1, 'away_team_goal':4, 'home_team':'Watford', 'away_team':'Tottenham Hotspur'},

{'match_id':6189, 'season':'2016/2017', 'stage':18, 'date':'42736', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Arsenal', 'away_team':'Crystal Palace'},

{'match_id':6190, 'season':'2016/2017', 'stage':18, 'date':'42737', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Middlesbrough', 'away_team':'Leicester City'},

{'match_id':6191, 'season':'2016/2017', 'stage':18, 'date':'42737', 'home_team_goal':3, 'away_team_goal':0, 'home_team':'Everton', 'away_team':'Southampton'},

{'match_id':6192, 'season':'2016/2017', 'stage':18, 'date':'42737', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Manchester City', 'away_team':'Burnley'},

{'match_id':6193, 'season':'2016/2017', 'stage':18, 'date':'42737', 'home_team_goal':2, 'away_team_goal':2, 'home_team':'Sunderland', 'away_team':'Liverpool'},

{'match_id':6194, 'season':'2016/2017', 'stage':18, 'date':'42737', 'home_team_goal':3, 'away_team_goal':1, 'home_team':'West Bromwich Albion', 'away_team':'Hull City'},

{'match_id':6195, 'season':'2016/2017', 'stage':18, 'date':'42737', 'home_team_goal':0, 'away_team_goal':2, 'home_team':'West Ham United', 'away_team':'Manchester United'},

{'match_id':6196, 'season':'2016/2017', 'stage':18, 'date':'42738', 'home_team_goal':3, 'away_team_goal':3, 'home_team':'Bournemouth', 'away_team':'Arsenal'},

{'match_id':6197, 'season':'2016/2017', 'stage':18, 'date':'42738', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Crystal Palace', 'away_team':'Swansea City'},

{'match_id':6198, 'season':'2016/2017', 'stage':18, 'date':'42738', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Stoke City', 'away_team':'Watford'},

{'match_id':6199, 'season':'2016/2017', 'stage':18, 'date':'42739', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Tottenham Hotspur', 'away_team':'Chelsea'},

{'match_id':6200, 'season':'2016/2017', 'stage':19, 'date':'42749', 'home_team_goal':4, 'away_team_goal':0, 'home_team':'Tottenham Hotspur', 'away_team':'West Bromwich Albion'},

{'match_id':6201, 'season':'2016/2017', 'stage':19, 'date':'42749', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Burnley', 'away_team':'Southampton'},

{'match_id':6202, 'season':'2016/2017', 'stage':19, 'date':'42749', 'home_team_goal':3, 'away_team_goal':1, 'home_team':'Hull City', 'away_team':'Bournemouth'},

{'match_id':6203, 'season':'2016/2017', 'stage':19, 'date':'42749', 'home_team_goal':1, 'away_team_goal':3, 'home_team':'Sunderland', 'away_team':'Stoke City'},

{'match_id':6204, 'season':'2016/2017', 'stage':19, 'date':'42749', 'home_team_goal':0, 'away_team_goal':4, 'home_team':'Swansea City', 'away_team':'Arsenal'},

{'match_id':6205, 'season':'2016/2017', 'stage':19, 'date':'42749', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Watford', 'away_team':'Middlesbrough'},

{'match_id':6206, 'season':'2016/2017', 'stage':19, 'date':'42749', 'home_team_goal':3, 'away_team_goal':0, 'home_team':'West Ham United', 'away_team':'Crystal Palace'},

{'match_id':6207, 'season':'2016/2017', 'stage':19, 'date':'42749', 'home_team_goal':0, 'away_team_goal':3, 'home_team':'Leicester City', 'away_team':'Chelsea'},

{'match_id':6208, 'season':'2016/2017', 'stage':19, 'date':'42750', 'home_team_goal':4, 'away_team_goal':0, 'home_team':'Everton', 'away_team':'Manchester City'},

{'match_id':6209, 'season':'2016/2017', 'stage':19, 'date':'42750', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Manchester United', 'away_team':'Liverpool'},

{'match_id':6210, 'season':'2016/2017', 'stage':20, 'date':'42756', 'home_team_goal':2, 'away_team_goal':3, 'home_team':'Liverpool', 'away_team':'Swansea City'},

{'match_id':6211, 'season':'2016/2017', 'stage':20, 'date':'42756', 'home_team_goal':2, 'away_team_goal':2, 'home_team':'Bournemouth', 'away_team':'Watford'},

{'match_id':6212, 'season':'2016/2017', 'stage':20, 'date':'42756', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Crystal Palace', 'away_team':'Everton'},

{'match_id':6213, 'season':'2016/2017', 'stage':20, 'date':'42756', 'home_team_goal':1, 'away_team_goal':3, 'home_team':'Middlesbrough', 'away_team':'West Ham United'},

{'match_id':6214, 'season':'2016/2017', 'stage':20, 'date':'42756', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Stoke City', 'away_team':'Manchester United'},

{'match_id':6215, 'season':'2016/2017', 'stage':20, 'date':'42756', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'West Bromwich Albion', 'away_team':'Sunderland'},

{'match_id':6216, 'season':'2016/2017', 'stage':20, 'date':'42756', 'home_team_goal':2, 'away_team_goal':2, 'home_team':'Manchester City', 'away_team':'Tottenham Hotspur'},

{'match_id':6217, 'season':'2016/2017', 'stage':20, 'date':'42757', 'home_team_goal':3, 'away_team_goal':0, 'home_team':'Southampton', 'away_team':'Leicester City'},

{'match_id':6218, 'season':'2016/2017', 'stage':20, 'date':'42757', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Arsenal', 'away_team':'Burnley'},

{'match_id':6219, 'season':'2016/2017', 'stage':20, 'date':'42757', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Chelsea', 'away_team':'Hull City'},

{'match_id':6220, 'season':'2016/2017', 'stage':21, 'date':'42766', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Arsenal', 'away_team':'Watford'},

{'match_id':6221, 'season':'2016/2017', 'stage':21, 'date':'42766', 'home_team_goal':0, 'away_team_goal':2, 'home_team':'Bournemouth', 'away_team':'Crystal Palace'},

{'match_id':6222, 'season':'2016/2017', 'stage':21, 'date':'42766', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Burnley', 'away_team':'Leicester City'},

{'match_id':6223, 'season':'2016/2017', 'stage':21, 'date':'42766', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Middlesbrough', 'away_team':'West Bromwich Albion'},

{'match_id':6224, 'season':'2016/2017', 'stage':21, 'date':'42766', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Sunderland', 'away_team':'Tottenham Hotspur'},

{'match_id':6225, 'season':'2016/2017', 'stage':21, 'date':'42766', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Swansea City', 'away_team':'Southampton'},

{'match_id':6226, 'season':'2016/2017', 'stage':21, 'date':'42766', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Liverpool', 'away_team':'Chelsea'},

{'match_id':6227, 'season':'2016/2017', 'stage':21, 'date':'42767', 'home_team_goal':0, 'away_team_goal':4, 'home_team':'West Ham United', 'away_team':'Manchester City'},

{'match_id':6228, 'season':'2016/2017', 'stage':21, 'date':'42767', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Manchester United', 'away_team':'Hull City'},

{'match_id':6229, 'season':'2016/2017', 'stage':21, 'date':'42767', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Stoke City', 'away_team':'Everton'},

{'match_id':6230, 'season':'2016/2017', 'stage':22, 'date':'42770', 'home_team_goal':3, 'away_team_goal':1, 'home_team':'Chelsea', 'away_team':'Arsenal'},

{'match_id':6231, 'season':'2016/2017', 'stage':22, 'date':'42770', 'home_team_goal':0, 'away_team_goal':4, 'home_team':'Crystal Palace', 'away_team':'Sunderland'},

{'match_id':6232, 'season':'2016/2017', 'stage':22, 'date':'42770', 'home_team_goal':6, 'away_team_goal':3, 'home_team':'Everton', 'away_team':'Bournemouth'},

{'match_id':6233, 'season':'2016/2017', 'stage':22, 'date':'42770', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Hull City', 'away_team':'Liverpool'},

{'match_id':6234, 'season':'2016/2017', 'stage':22, 'date':'42770', 'home_team_goal':1, 'away_team_goal':3, 'home_team':'Southampton', 'away_team':'West Ham United'},

{'match_id':6235, 'season':'2016/2017', 'stage':22, 'date':'42770', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Watford', 'away_team':'Burnley'},

{'match_id':6236, 'season':'2016/2017', 'stage':22, 'date':'42770', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'West Bromwich Albion', 'away_team':'Stoke City'},

{'match_id':6237, 'season':'2016/2017', 'stage':22, 'date':'42770', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Tottenham Hotspur', 'away_team':'Middlesbrough'},

{'match_id':6238, 'season':'2016/2017', 'stage':22, 'date':'42771', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Manchester City', 'away_team':'Swansea City'},

{'match_id':6239, 'season':'2016/2017', 'stage':22, 'date':'42771', 'home_team_goal':0, 'away_team_goal':3, 'home_team':'Leicester City', 'away_team':'Manchester United'},

{'match_id':6240, 'season':'2016/2017', 'stage':23, 'date':'42777', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Arsenal', 'away_team':'Hull City'},

{'match_id':6241, 'season':'2016/2017', 'stage':23, 'date':'42777', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Manchester United', 'away_team':'Watford'},

{'match_id':6242, 'season':'2016/2017', 'stage':23, 'date':'42777', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Middlesbrough', 'away_team':'Everton'},

{'match_id':6243, 'season':'2016/2017', 'stage':23, 'date':'42777', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Stoke City', 'away_team':'Crystal Palace'},

{'match_id':6244, 'season':'2016/2017', 'stage':23, 'date':'42777', 'home_team_goal':0, 'away_team_goal':4, 'home_team':'Sunderland', 'away_team':'Southampton'},

{'match_id':6245, 'season':'2016/2017', 'stage':23, 'date':'42777', 'home_team_goal':2, 'away_team_goal':2, 'home_team':'West Ham United', 'away_team':'West Bromwich Albion'},

{'match_id':6246, 'season':'2016/2017', 'stage':23, 'date':'42777', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Liverpool', 'away_team':'Tottenham Hotspur'},

{'match_id':6247, 'season':'2016/2017', 'stage':23, 'date':'42778', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Burnley', 'away_team':'Chelsea'},

{'match_id':6248, 'season':'2016/2017', 'stage':23, 'date':'42778', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Swansea City', 'away_team':'Leicester City'},

{'match_id':6249, 'season':'2016/2017', 'stage':23, 'date':'42779', 'home_team_goal':0, 'away_team_goal':2, 'home_team':'Bournemouth', 'away_team':'Manchester City'},

{'match_id':6250, 'season':'2016/2017', 'stage':24, 'date':'42791', 'home_team_goal':3, 'away_team_goal':1, 'home_team':'Chelsea', 'away_team':'Swansea City'},

{'match_id':6251, 'season':'2016/2017', 'stage':24, 'date':'42791', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Crystal Palace', 'away_team':'Middlesbrough'},

{'match_id':6252, 'season':'2016/2017', 'stage':24, 'date':'42791', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Everton', 'away_team':'Sunderland'},

{'match_id':6253, 'season':'2016/2017', 'stage':24, 'date':'42791', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Hull City', 'away_team':'Burnley'},

{'match_id':6254, 'season':'2016/2017', 'stage':24, 'date':'42791', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'West Bromwich Albion', 'away_team':'Bournemouth'},

{'match_id':6255, 'season':'2016/2017', 'stage':24, 'date':'42791', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Watford', 'away_team':'West Ham United'},

{'match_id':6256, 'season':'2016/2017', 'stage':24, 'date':'42792', 'home_team_goal':4, 'away_team_goal':0, 'home_team':'Tottenham Hotspur', 'away_team':'Stoke City'},

{'match_id':6257, 'season':'2016/2017', 'stage':24, 'date':'42793', 'home_team_goal':3, 'away_team_goal':1, 'home_team':'Leicester City', 'away_team':'Liverpool'},

{'match_id':6258, 'season':'2016/2017', 'stage':25, 'date':'42798', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Manchester United', 'away_team':'Bournemouth'},

{'match_id':6259, 'season':'2016/2017', 'stage':25, 'date':'42798', 'home_team_goal':3, 'away_team_goal':1, 'home_team':'Leicester City', 'away_team':'Hull City'},

{'match_id':6260, 'season':'2016/2017', 'stage':25, 'date':'42798', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Stoke City', 'away_team':'Middlesbrough'},

{'match_id':6261, 'season':'2016/2017', 'stage':25, 'date':'42798', 'home_team_goal':3, 'away_team_goal':2, 'home_team':'Swansea City', 'away_team':'Burnley'},

{'match_id':6262, 'season':'2016/2017', 'stage':25, 'date':'42798', 'home_team_goal':3, 'away_team_goal':4, 'home_team':'Watford', 'away_team':'Southampton'},

{'match_id':6263, 'season':'2016/2017', 'stage':25, 'date':'42798', 'home_team_goal':0, 'away_team_goal':2, 'home_team':'West Bromwich Albion', 'away_team':'Crystal Palace'},

{'match_id':6264, 'season':'2016/2017', 'stage':25, 'date':'42798', 'home_team_goal':3, 'away_team_goal':1, 'home_team':'Liverpool', 'away_team':'Arsenal'},

{'match_id':6265, 'season':'2016/2017', 'stage':25, 'date':'42799', 'home_team_goal':3, 'away_team_goal':2, 'home_team':'Tottenham Hotspur', 'away_team':'Everton'},

{'match_id':6266, 'season':'2016/2017', 'stage':25, 'date':'42799', 'home_team_goal':0, 'away_team_goal':2, 'home_team':'Sunderland', 'away_team':'Manchester City'},

{'match_id':6267, 'season':'2016/2017', 'stage':25, 'date':'42800', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'West Ham United', 'away_team':'Chelsea'},

{'match_id':6268, 'season':'2016/2017', 'stage':25, 'date':'42802', 'home_team_goal':3, 'away_team_goal':0, 'home_team':'Manchester City', 'away_team':'Stoke City'},

{'match_id':6269, 'season':'2016/2017', 'stage':26, 'date':'42805', 'home_team_goal':3, 'away_team_goal':2, 'home_team':'Bournemouth', 'away_team':'West Ham United'},

{'match_id':6270, 'season':'2016/2017', 'stage':26, 'date':'42805', 'home_team_goal':3, 'away_team_goal':0, 'home_team':'Everton', 'away_team':'West Bromwich Albion'},

{'match_id':6271, 'season':'2016/2017', 'stage':26, 'date':'42805', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Hull City', 'away_team':'Swansea City'},

{'match_id':6272, 'season':'2016/2017', 'stage':26, 'date':'42806', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Liverpool', 'away_team':'Burnley'},

{'match_id':6273, 'season':'2016/2017', 'stage':27, 'date':'42812', 'home_team_goal':3, 'away_team_goal':3, 'home_team':'West Bromwich Albion', 'away_team':'Arsenal'},

{'match_id':6274, 'season':'2016/2017', 'stage':27, 'date':'42812', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Crystal Palace', 'away_team':'Watford'},

{'match_id':6275, 'season':'2016/2017', 'stage':27, 'date':'42812', 'home_team_goal':4, 'away_team_goal':0, 'home_team':'Everton', 'away_team':'Hull City'},

{'match_id':6276, 'season':'2016/2017', 'stage':27, 'date':'42812', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Stoke City', 'away_team':'Chelsea'},

{'match_id':6277, 'season':'2016/2017', 'stage':27, 'date':'42812', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Sunderland', 'away_team':'Burnley'},

{'match_id':6278, 'season':'2016/2017', 'stage':27, 'date':'42812', 'home_team_goal':2, 'away_team_goal':3, 'home_team':'West Ham United', 'away_team':'Leicester City'},

{'match_id':6279, 'season':'2016/2017', 'stage':27, 'date':'42812', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Bournemouth', 'away_team':'Swansea City'},

{'match_id':6280, 'season':'2016/2017', 'stage':27, 'date':'42813', 'home_team_goal':1, 'away_team_goal':3, 'home_team':'Middlesbrough', 'away_team':'Manchester United'},

{'match_id':6281, 'season':'2016/2017', 'stage':27, 'date':'42813', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Tottenham Hotspur', 'away_team':'Southampton'},

{'match_id':6282, 'season':'2016/2017', 'stage':27, 'date':'42813', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Manchester City', 'away_team':'Liverpool'},

{'match_id':6283, 'season':'2016/2017', 'stage':28, 'date':'42826', 'home_team_goal':3, 'away_team_goal':1, 'home_team':'Liverpool', 'away_team':'Everton'},

{'match_id':6284, 'season':'2016/2017', 'stage':28, 'date':'42826', 'home_team_goal':0, 'away_team_goal':2, 'home_team':'Burnley', 'away_team':'Tottenham Hotspur'},

{'match_id':6285, 'season':'2016/2017', 'stage':28, 'date':'42826', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Chelsea', 'away_team':'Crystal Palace'},

{'match_id':6286, 'season':'2016/2017', 'stage':28, 'date':'42826', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Hull City', 'away_team':'West Ham United'},

{'match_id':6287, 'season':'2016/2017', 'stage':28, 'date':'42826', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Leicester City', 'away_team':'Stoke City'},

{'match_id':6288, 'season':'2016/2017', 'stage':28, 'date':'42826', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Manchester United', 'away_team':'West Bromwich Albion'},

{'match_id':6289, 'season':'2016/2017', 'stage':28, 'date':'42826', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Watford', 'away_team':'Sunderland'},

{'match_id':6290, 'season':'2016/2017', 'stage':28, 'date':'42826', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Southampton', 'away_team':'Bournemouth'},

{'match_id':6291, 'season':'2016/2017', 'stage':28, 'date':'42826', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Swansea City', 'away_team':'Middlesbrough'},

{'match_id':6292, 'season':'2016/2017', 'stage':28, 'date':'42826', 'home_team_goal':2, 'away_team_goal':2, 'home_team':'Arsenal', 'away_team':'Manchester City'},

{'match_id':6293, 'season':'2016/2017', 'stage':28, 'date':'42830', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Burnley', 'away_team':'Stoke City'},

{'match_id':6294, 'season':'2016/2017', 'stage':28, 'date':'42830', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Leicester City', 'away_team':'Sunderland'},

{'match_id':6295, 'season':'2016/2017', 'stage':28, 'date':'42830', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Watford', 'away_team':'West Bromwich Albion'},

{'match_id':6296, 'season':'2016/2017', 'stage':28, 'date':'42830', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Manchester United', 'away_team':'Everton'},

{'match_id':6297, 'season':'2016/2017', 'stage':28, 'date':'42830', 'home_team_goal':3, 'away_team_goal':0, 'home_team':'Arsenal', 'away_team':'West Ham United'},

{'match_id':6298, 'season':'2016/2017', 'stage':28, 'date':'42830', 'home_team_goal':4, 'away_team_goal':2, 'home_team':'Hull City', 'away_team':'Middlesbrough'},

{'match_id':6299, 'season':'2016/2017', 'stage':28, 'date':'42830', 'home_team_goal':3, 'away_team_goal':1, 'home_team':'Southampton', 'away_team':'Crystal Palace'},

{'match_id':6300, 'season':'2016/2017', 'stage':28, 'date':'42830', 'home_team_goal':1, 'away_team_goal':3, 'home_team':'Swansea City', 'away_team':'Tottenham Hotspur'},

{'match_id':6301, 'season':'2016/2017', 'stage':28, 'date':'42830', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Chelsea', 'away_team':'Manchester City'},

{'match_id':6302, 'season':'2016/2017', 'stage':28, 'date':'42830', 'home_team_goal':2, 'away_team_goal':2, 'home_team':'Liverpool', 'away_team':'Bournemouth'},

{'match_id':6303, 'season':'2016/2017', 'stage':29, 'date':'42833', 'home_team_goal':4, 'away_team_goal':0, 'home_team':'Tottenham Hotspur', 'away_team':'Watford'},

{'match_id':6304, 'season':'2016/2017', 'stage':29, 'date':'42833', 'home_team_goal':3, 'away_team_goal':1, 'home_team':'Manchester City', 'away_team':'Hull City'},

{'match_id':6305, 'season':'2016/2017', 'stage':29, 'date':'42833', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Middlesbrough', 'away_team':'Burnley'},

{'match_id':6306, 'season':'2016/2017', 'stage':29, 'date':'42833', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Stoke City', 'away_team':'Liverpool'},

{'match_id':6307, 'season':'2016/2017', 'stage':29, 'date':'42833', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'West Bromwich Albion', 'away_team':'Southampton'},

{'match_id':6308, 'season':'2016/2017', 'stage':29, 'date':'42833', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'West Ham United', 'away_team':'Swansea City'},

{'match_id':6309, 'season':'2016/2017', 'stage':29, 'date':'42833', 'home_team_goal':1, 'away_team_goal':3, 'home_team':'Bournemouth', 'away_team':'Chelsea'},

{'match_id':6310, 'season':'2016/2017', 'stage':29, 'date':'42833', 'home_team_goal':0, 'away_team_goal':3, 'home_team':'Sunderland', 'away_team':'Manchester United'},

{'match_id':6311, 'season':'2016/2017', 'stage':29, 'date':'42833', 'home_team_goal':4, 'away_team_goal':2, 'home_team':'Everton', 'away_team':'Leicester City'},

{'match_id':6312, 'season':'2016/2017', 'stage':29, 'date':'42833', 'home_team_goal':3, 'away_team_goal':0, 'home_team':'Crystal Palace', 'away_team':'Arsenal'},

{'match_id':6313, 'season':'2016/2017', 'stage':30, 'date':'42840', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Tottenham Hotspur', 'away_team':'Bournemouth'},

{'match_id':6314, 'season':'2016/2017', 'stage':30, 'date':'42840', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Crystal Palace', 'away_team':'Leicester City'},

{'match_id':6315, 'season':'2016/2017', 'stage':30, 'date':'42840', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Everton', 'away_team':'Burnley'},

{'match_id':6316, 'season':'2016/2017', 'stage':30, 'date':'42840', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Stoke City', 'away_team':'Hull City'},

{'match_id':6317, 'season':'2016/2017', 'stage':30, 'date':'42840', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Sunderland', 'away_team':'West Ham United'},

{'match_id':6318, 'season':'2016/2017', 'stage':30, 'date':'42840', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Watford', 'away_team':'Swansea City'},

{'match_id':6319, 'season':'2016/2017', 'stage':30, 'date':'42840', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Southampton', 'away_team':'Manchester City'},

{'match_id':6320, 'season':'2016/2017', 'stage':30, 'date':'42840', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'West Bromwich Albion', 'away_team':'Liverpool'},

{'match_id':6321, 'season':'2016/2017', 'stage':30, 'date':'42840', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Manchester United', 'away_team':'Chelsea'},

{'match_id':6322, 'season':'2016/2017', 'stage':30, 'date':'42840', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Middlesbrough', 'away_team':'Arsenal'},

{'match_id':6323, 'season':'2016/2017', 'stage':31, 'date':'42847', 'home_team_goal':4, 'away_team_goal':0, 'home_team':'Bournemouth', 'away_team':'Middlesbrough'},

{'match_id':6324, 'season':'2016/2017', 'stage':31, 'date':'42847', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Hull City', 'away_team':'Watford'},

{'match_id':6325, 'season':'2016/2017', 'stage':31, 'date':'42847', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Swansea City', 'away_team':'Stoke City'},

{'match_id':6326, 'season':'2016/2017', 'stage':31, 'date':'42847', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'West Ham United', 'away_team':'Everton'},

{'match_id':6327, 'season':'2016/2017', 'stage':31, 'date':'42847', 'home_team_goal':0, 'away_team_goal':2, 'home_team':'Burnley', 'away_team':'Manchester United'},

{'match_id':6328, 'season':'2016/2017', 'stage':31, 'date':'42847', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Liverpool', 'away_team':'Crystal Palace'},

{'match_id':6329, 'season':'2016/2017', 'stage':31, 'date':'42847', 'home_team_goal':4, 'away_team_goal':2, 'home_team':'Chelsea', 'away_team':'Southampton'},

{'match_id':6330, 'season':'2016/2017', 'stage':31, 'date':'42847', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Arsenal', 'away_team':'Leicester City'},

{'match_id':6331, 'season':'2016/2017', 'stage':31, 'date':'42847', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Middlesbrough', 'away_team':'Sunderland'},

{'match_id':6332, 'season':'2016/2017', 'stage':31, 'date':'42847', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Crystal Palace', 'away_team':'Tottenham Hotspur'},

{'match_id':6333, 'season':'2016/2017', 'stage':31, 'date':'42847', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Manchester City', 'away_team':'Manchester United'},

{'match_id':6334, 'season':'2016/2017', 'stage':32, 'date':'42854', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Southampton', 'away_team':'Hull City'},

{'match_id':6335, 'season':'2016/2017', 'stage':32, 'date':'42854', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Stoke City', 'away_team':'West Ham United'},

{'match_id':6336, 'season':'2016/2017', 'stage':32, 'date':'42854', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Sunderland', 'away_team':'Bournemouth'},

{'match_id':6337, 'season':'2016/2017', 'stage':32, 'date':'42854', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'West Bromwich Albion', 'away_team':'Leicester City'},

{'match_id':6338, 'season':'2016/2017', 'stage':32, 'date':'42854', 'home_team_goal':0, 'away_team_goal':2, 'home_team':'Crystal Palace', 'away_team':'Burnley'},

{'match_id':6339, 'season':'2016/2017', 'stage':32, 'date':'42854', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Manchester United', 'away_team':'Swansea City'},

{'match_id':6340, 'season':'2016/2017', 'stage':32, 'date':'42854', 'home_team_goal':0, 'away_team_goal':3, 'home_team':'Everton', 'away_team':'Chelsea'},

{'match_id':6341, 'season':'2016/2017', 'stage':32, 'date':'42854', 'home_team_goal':2, 'away_team_goal':2, 'home_team':'Middlesbrough', 'away_team':'Manchester City'},

{'match_id':6342, 'season':'2016/2017', 'stage':32, 'date':'42854', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Tottenham Hotspur', 'away_team':'Arsenal'},

{'match_id':6343, 'season':'2016/2017', 'stage':32, 'date':'42854', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Watford', 'away_team':'Liverpool'},

{'match_id':6344, 'season':'2016/2017', 'stage':33, 'date':'42861', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'West Ham United', 'away_team':'Tottenham Hotspur'},

{'match_id':6345, 'season':'2016/2017', 'stage':33, 'date':'42861', 'home_team_goal':5, 'away_team_goal':0, 'home_team':'Manchester City', 'away_team':'Crystal Palace'},

{'match_id':6346, 'season':'2016/2017', 'stage':33, 'date':'42861', 'home_team_goal':2, 'away_team_goal':2, 'home_team':'Bournemouth', 'away_team':'Stoke City'},

{'match_id':6347, 'season':'2016/2017', 'stage':33, 'date':'42861', 'home_team_goal':2, 'away_team_goal':2, 'home_team':'Burnley', 'away_team':'West Bromwich Albion'},

{'match_id':6348, 'season':'2016/2017', 'stage':33, 'date':'42861', 'home_team_goal':0, 'away_team_goal':2, 'home_team':'Hull City', 'away_team':'Sunderland'},

{'match_id':6349, 'season':'2016/2017', 'stage':33, 'date':'42861', 'home_team_goal':3, 'away_team_goal':0, 'home_team':'Leicester City', 'away_team':'Watford'},

{'match_id':6350, 'season':'2016/2017', 'stage':33, 'date':'42861', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Swansea City', 'away_team':'Everton'},

{'match_id':6351, 'season':'2016/2017', 'stage':33, 'date':'42861', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Liverpool', 'away_team':'Southampton'},

{'match_id':6352, 'season':'2016/2017', 'stage':33, 'date':'42861', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Arsenal', 'away_team':'Manchester United'},

{'match_id':6353, 'season':'2016/2017', 'stage':33, 'date':'42861', 'home_team_goal':3, 'away_team_goal':0, 'home_team':'Chelsea', 'away_team':'Middlesbrough'},

{'match_id':6354, 'season':'2016/2017', 'stage':33, 'date':'42861', 'home_team_goal':0, 'away_team_goal':2, 'home_team':'Southampton', 'away_team':'Arsenal'},

{'match_id':6355, 'season':'2016/2017', 'stage':34, 'date':'42868', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Everton', 'away_team':'Watford'},

{'match_id':6356, 'season':'2016/2017', 'stage':34, 'date':'42868', 'home_team_goal':0, 'away_team_goal':2, 'home_team':'West Bromwich Albion', 'away_team':'Chelsea'},

{'match_id':6357, 'season':'2016/2017', 'stage':34, 'date':'42868', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Manchester City', 'away_team':'Leicester City'},

{'match_id':6358, 'season':'2016/2017', 'stage':34, 'date':'42868', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Bournemouth', 'away_team':'Burnley'},

{'match_id':6359, 'season':'2016/2017', 'stage':34, 'date':'42868', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Middlesbrough', 'away_team':'Southampton'},

{'match_id':6360, 'season':'2016/2017', 'stage':34, 'date':'42868', 'home_team_goal':0, 'away_team_goal':2, 'home_team':'Sunderland', 'away_team':'Swansea City'},

{'match_id':6361, 'season':'2016/2017', 'stage':34, 'date':'42868', 'home_team_goal':1, 'away_team_goal':4, 'home_team':'Stoke City', 'away_team':'Arsenal'},

{'match_id':6362, 'season':'2016/2017', 'stage':34, 'date':'42868', 'home_team_goal':4, 'away_team_goal':0, 'home_team':'Crystal Palace', 'away_team':'Hull City'},

{'match_id':6363, 'season':'2016/2017', 'stage':34, 'date':'42868', 'home_team_goal':0, 'away_team_goal':4, 'home_team':'West Ham United', 'away_team':'Liverpool'},

{'match_id':6364, 'season':'2016/2017', 'stage':34, 'date':'42868', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Tottenham Hotspur', 'away_team':'Manchester United'},

{'match_id':6365, 'season':'2016/2017', 'stage':34, 'date':'42870', 'home_team_goal':4, 'away_team_goal':3, 'home_team':'Chelsea', 'away_team':'Watford'},

{'match_id':6366, 'season':'2016/2017', 'stage':34, 'date':'42870', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Arsenal', 'away_team':'Sunderland'},

{'match_id':6367, 'season':'2016/2017', 'stage':34, 'date':'42870', 'home_team_goal':3, 'away_team_goal':1, 'home_team':'Manchester City', 'away_team':'West Bromwich Albion'},

{'match_id':6368, 'season':'2016/2017', 'stage':34, 'date':'42870', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Southampton', 'away_team':'Manchester United'},

{'match_id':6369, 'season':'2016/2017', 'stage':34, 'date':'42870', 'home_team_goal':1, 'away_team_goal':6, 'home_team':'Leicester City', 'away_team':'Tottenham Hotspur'},

{'match_id':6370, 'season':'2016/2017', 'stage':35, 'date':'42876', 'home_team_goal':3, 'away_team_goal':1, 'home_team':'Arsenal', 'away_team':'Everton'},

{'match_id':6371, 'season':'2016/2017', 'stage':35, 'date':'42876', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Burnley', 'away_team':'West Ham United'},

{'match_id':6372, 'season':'2016/2017', 'stage':35, 'date':'42876', 'home_team_goal':5, 'away_team_goal':1, 'home_team':'Chelsea', 'away_team':'Sunderland'},

{'match_id':6373, 'season':'2016/2017', 'stage':35, 'date':'42876', 'home_team_goal':1, 'away_team_goal':7, 'home_team':'Hull City', 'away_team':'Tottenham Hotspur'},

{'match_id':6374, 'season':'2016/2017', 'stage':35, 'date':'42876', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Leicester City', 'away_team':'Bournemouth'},

{'match_id':6375, 'season':'2016/2017', 'stage':35, 'date':'42876', 'home_team_goal':3, 'away_team_goal':0, 'home_team':'Liverpool', 'away_team':'Middlesbrough'},

{'match_id':6376, 'season':'2016/2017', 'stage':35, 'date':'42876', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Manchester United', 'away_team':'Crystal Palace'},

{'match_id':6377, 'season':'2016/2017', 'stage':35, 'date':'42876', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Southampton', 'away_team':'Stoke City'},

{'match_id':6378, 'season':'2016/2017', 'stage':35, 'date':'42876', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Swansea City', 'away_team':'West Bromwich Albion'},

{'match_id':6379, 'season':'2016/2017', 'stage':35, 'date':'42876', 'home_team_goal':0, 'away_team_goal':5, 'home_team':'Watford', 'away_team':'Manchester City'},

{'match_id':7000, 'season':'2017/2018', 'stage':1, 'date':'42959', 'home_team_goal':4, 'away_team_goal':3, 'home_team':'Arsenal', 'away_team':'Leicester City'},

{'match_id':7001, 'season':'2017/2018', 'stage':1, 'date':'42959', 'home_team_goal':3, 'away_team_goal':3, 'home_team':'Watford', 'away_team':'Liverpool'},

{'match_id':7002, 'season':'2017/2018', 'stage':1, 'date':'42959', 'home_team_goal':0, 'away_team_goal':3, 'home_team':'Crystal Palace', 'away_team':'Huddersfield'},

{'match_id':7003, 'season':'2017/2018', 'stage':1, 'date':'42959', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Everton', 'away_team':'Stoke City'},

{'match_id':7004, 'season':'2017/2018', 'stage':1, 'date':'42959', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Southampton', 'away_team':'Swansea City'},

{'match_id':7005, 'season':'2017/2018', 'stage':1, 'date':'42959', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'West Bromwich Albion', 'away_team':'Bournemouth'},

{'match_id':7006, 'season':'2017/2018', 'stage':1, 'date':'42959', 'home_team_goal':0, 'away_team_goal':2, 'home_team':'Brighton', 'away_team':'Manchester City'},

{'match_id':7007, 'season':'2017/2018', 'stage':1, 'date':'42959', 'home_team_goal':0, 'away_team_goal':2, 'home_team':'Newcastle United', 'away_team':'Tottenham Hotspur'},

{'match_id':7008, 'season':'2017/2018', 'stage':1, 'date':'42959', 'home_team_goal':4, 'away_team_goal':0, 'home_team':'Manchester United', 'away_team':'West Ham United'},

{'match_id':7009, 'season':'2017/2018', 'stage':1, 'date':'42959', 'home_team_goal':2, 'away_team_goal':3, 'home_team':'Chelsea', 'away_team':'Burnley'},

{'match_id':7010, 'season':'2017/2018', 'stage':2, 'date':'42966', 'home_team_goal':0, 'away_team_goal':4, 'home_team':'Swansea City', 'away_team':'Manchester United'},

{'match_id':7011, 'season':'2017/2018', 'stage':2, 'date':'42966', 'home_team_goal':0, 'away_team_goal':2, 'home_team':'Bournemouth', 'away_team':'Watford'},

{'match_id':7012, 'season':'2017/2018', 'stage':2, 'date':'42966', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Burnley', 'away_team':'West Bromwich Albion'},

{'match_id':7013, 'season':'2017/2018', 'stage':2, 'date':'42966', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Leicester City', 'away_team':'Brighton'},

{'match_id':7014, 'season':'2017/2018', 'stage':2, 'date':'42966', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Liverpool', 'away_team':'Crystal Palace'},

{'match_id':7015, 'season':'2017/2018', 'stage':2, 'date':'42966', 'home_team_goal':3, 'away_team_goal':2, 'home_team':'Southampton', 'away_team':'West Ham United'},

{'match_id':7016, 'season':'2017/2018', 'stage':2, 'date':'42966', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Stoke City', 'away_team':'Arsenal'},

{'match_id':7017, 'season':'2017/2018', 'stage':2, 'date':'42966', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Huddersfield', 'away_team':'Newcastle United'},

{'match_id':7018, 'season':'2017/2018', 'stage':2, 'date':'42966', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Tottenham Hotspur', 'away_team':'Chelsea'},

{'match_id':7019, 'season':'2017/2018', 'stage':2, 'date':'42966', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Manchester City', 'away_team':'Everton'},

{'match_id':7020, 'season':'2017/2018', 'stage':3, 'date':'42973', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Bournemouth', 'away_team':'Manchester City'},

{'match_id':7021, 'season':'2017/2018', 'stage':3, 'date':'42973', 'home_team_goal':0, 'away_team_goal':2, 'home_team':'Crystal Palace', 'away_team':'Swansea City'},

{'match_id':7022, 'season':'2017/2018', 'stage':3, 'date':'42973', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Huddersfield', 'away_team':'Southampton'},

{'match_id':7023, 'season':'2017/2018', 'stage':3, 'date':'42973', 'home_team_goal':3, 'away_team_goal':0, 'home_team':'Newcastle United', 'away_team':'West Ham United'},

{'match_id':7024, 'season':'2017/2018', 'stage':3, 'date':'42973', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Watford', 'away_team':'Brighton'},

{'match_id':7025, 'season':'2017/2018', 'stage':3, 'date':'42973', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Manchester United', 'away_team':'Leicester City'},

{'match_id':7026, 'season':'2017/2018', 'stage':3, 'date':'42973', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Chelsea', 'away_team':'Everton'},

{'match_id':7027, 'season':'2017/2018', 'stage':3, 'date':'42973', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'West Bromwich Albion', 'away_team':'Stoke City'},

{'match_id':7028, 'season':'2017/2018', 'stage':3, 'date':'42973', 'home_team_goal':4, 'away_team_goal':0, 'home_team':'Liverpool', 'away_team':'Arsenal'},

{'match_id':7029, 'season':'2017/2018', 'stage':3, 'date':'42973', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Tottenham Hotspur', 'away_team':'Burnley'},

{'match_id':7030, 'season':'2017/2018', 'stage':4, 'date':'42987', 'home_team_goal':5, 'away_team_goal':0, 'home_team':'Manchester City', 'away_team':'Liverpool'},

{'match_id':7031, 'season':'2017/2018', 'stage':4, 'date':'42987', 'home_team_goal':3, 'away_team_goal':0, 'home_team':'Arsenal', 'away_team':'Bournemouth'},

{'match_id':7032, 'season':'2017/2018', 'stage':4, 'date':'42987', 'home_team_goal':3, 'away_team_goal':1, 'home_team':'Brighton', 'away_team':'West Bromwich Albion'},

{'match_id':7033, 'season':'2017/2018', 'stage':4, 'date':'42987', 'home_team_goal':0, 'away_team_goal':3, 'home_team':'Everton', 'away_team':'Tottenham Hotspur'},

{'match_id':7034, 'season':'2017/2018', 'stage':4, 'date':'42987', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Leicester City', 'away_team':'Chelsea'},

{'match_id':7035, 'season':'2017/2018', 'stage':4, 'date':'42987', 'home_team_goal':0, 'away_team_goal':2, 'home_team':'Southampton', 'away_team':'Watford'},

{'match_id':7036, 'season':'2017/2018', 'stage':4, 'date':'42987', 'home_team_goal':2, 'away_team_goal':2, 'home_team':'Stoke City', 'away_team':'Manchester United'},

{'match_id':7037, 'season':'2017/2018', 'stage':4, 'date':'42987', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Burnley', 'away_team':'Crystal Palace'},

{'match_id':7038, 'season':'2017/2018', 'stage':4, 'date':'42987', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Swansea City', 'away_team':'Newcastle United'},

{'match_id':7039, 'season':'2017/2018', 'stage':4, 'date':'42987', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'West Ham United', 'away_team':'Huddersfield'},

{'match_id':7040, 'season':'2017/2018', 'stage':5, 'date':'42994', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Bournemouth', 'away_team':'Brighton'},

{'match_id':7041, 'season':'2017/2018', 'stage':5, 'date':'42994', 'home_team_goal':0, 'away_team_goal':2, 'home_team':'Crystal Palace', 'away_team':'Southampton'},

{'match_id':7042, 'season':'2017/2018', 'stage':5, 'date':'42994', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Huddersfield', 'away_team':'Leicester City'},

{'match_id':7043, 'season':'2017/2018', 'stage':5, 'date':'42994', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Liverpool', 'away_team':'Burnley'},

{'match_id':7044, 'season':'2017/2018', 'stage':5, 'date':'42994', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Newcastle United', 'away_team':'Stoke City'},

{'match_id':7045, 'season':'2017/2018', 'stage':5, 'date':'42994', 'home_team_goal':0, 'away_team_goal':6, 'home_team':'Watford', 'away_team':'Manchester City'},

{'match_id':7046, 'season':'2017/2018', 'stage':5, 'date':'42994', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'West Bromwich Albion', 'away_team':'West Ham United'},

{'match_id':7047, 'season':'2017/2018', 'stage':5, 'date':'42994', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Tottenham Hotspur', 'away_team':'Swansea City'},

{'match_id':7048, 'season':'2017/2018', 'stage':5, 'date':'42994', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Chelsea', 'away_team':'Arsenal'},

{'match_id':7049, 'season':'2017/2018', 'stage':5, 'date':'42994', 'home_team_goal':4, 'away_team_goal':0, 'home_team':'Manchester United', 'away_team':'Everton'},

{'match_id':7050, 'season':'2017/2018', 'stage':6, 'date':'43001', 'home_team_goal':2, 'away_team_goal':3, 'home_team':'West Ham United', 'away_team':'Tottenham Hotspur'},

{'match_id':7051, 'season':'2017/2018', 'stage':6, 'date':'43001', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Burnley', 'away_team':'Huddersfield'},

{'match_id':7052, 'season':'2017/2018', 'stage':6, 'date':'43001', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Everton', 'away_team':'Bournemouth'},

{'match_id':7053, 'season':'2017/2018', 'stage':6, 'date':'43001', 'home_team_goal':5, 'away_team_goal':0, 'home_team':'Manchester City', 'away_team':'Crystal Palace'},

{'match_id':7054, 'season':'2017/2018', 'stage':6, 'date':'43001', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Southampton', 'away_team':'Manchester United'},

{'match_id':7055, 'season':'2017/2018', 'stage':6, 'date':'43001', 'home_team_goal':0, 'away_team_goal':4, 'home_team':'Stoke City', 'away_team':'Chelsea'},

{'match_id':7056, 'season':'2017/2018', 'stage':6, 'date':'43001', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Swansea City', 'away_team':'Watford'},

{'match_id':7057, 'season':'2017/2018', 'stage':6, 'date':'43001', 'home_team_goal':2, 'away_team_goal':3, 'home_team':'Leicester City', 'away_team':'Liverpool'},

{'match_id':7058, 'season':'2017/2018', 'stage':6, 'date':'43001', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Brighton', 'away_team':'Newcastle United'},

{'match_id':7059, 'season':'2017/2018', 'stage':6, 'date':'43001', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Arsenal', 'away_team':'West Bromwich Albion'},

{'match_id':7060, 'season':'2017/2018', 'stage':7, 'date':'43008', 'home_team_goal':0, 'away_team_goal':4, 'home_team':'Huddersfield', 'away_team':'Tottenham Hotspur'},

{'match_id':7061, 'season':'2017/2018', 'stage':7, 'date':'43008', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Bournemouth', 'away_team':'Leicester City'},

{'match_id':7062, 'season':'2017/2018', 'stage':7, 'date':'43008', 'home_team_goal':4, 'away_team_goal':0, 'home_team':'Manchester United', 'away_team':'Crystal Palace'},

{'match_id':7063, 'season':'2017/2018', 'stage':7, 'date':'43008', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Stoke City', 'away_team':'Southampton'},

{'match_id':7064, 'season':'2017/2018', 'stage':7, 'date':'43008', 'home_team_goal':2, 'away_team_goal':2, 'home_team':'West Bromwich Albion', 'away_team':'Watford'},

{'match_id':7065, 'season':'2017/2018', 'stage':7, 'date':'43008', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'West Ham United', 'away_team':'Swansea City'},

{'match_id':7066, 'season':'2017/2018', 'stage':7, 'date':'43008', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Chelsea', 'away_team':'Manchester City'},

{'match_id':7067, 'season':'2017/2018', 'stage':7, 'date':'43008', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Arsenal', 'away_team':'Brighton'},

{'match_id':7068, 'season':'2017/2018', 'stage':7, 'date':'43008', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Everton', 'away_team':'Burnley'},

{'match_id':7069, 'season':'2017/2018', 'stage':7, 'date':'43008', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Newcastle United', 'away_team':'Liverpool'},

{'match_id':7070, 'season':'2017/2018', 'stage':8, 'date':'43022', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Liverpool', 'away_team':'Manchester United'},

{'match_id':7071, 'season':'2017/2018', 'stage':8, 'date':'43022', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Burnley', 'away_team':'West Ham United'},

{'match_id':7072, 'season':'2017/2018', 'stage':8, 'date':'43022', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Crystal Palace', 'away_team':'Chelsea'},

{'match_id':7073, 'season':'2017/2018', 'stage':8, 'date':'43022', 'home_team_goal':7, 'away_team_goal':2, 'home_team':'Manchester City', 'away_team':'Stoke City'},

{'match_id':7074, 'season':'2017/2018', 'stage':8, 'date':'43022', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Swansea City', 'away_team':'Huddersfield'},

{'match_id':7075, 'season':'2017/2018', 'stage':8, 'date':'43022', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Tottenham Hotspur', 'away_team':'Bournemouth'},

{'match_id':7076, 'season':'2017/2018', 'stage':8, 'date':'43022', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Watford', 'away_team':'Arsenal'},

{'match_id':7077, 'season':'2017/2018', 'stage':8, 'date':'43022', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Brighton', 'away_team':'Everton'},

{'match_id':7078, 'season':'2017/2018', 'stage':8, 'date':'43022', 'home_team_goal':2, 'away_team_goal':2, 'home_team':'Southampton', 'away_team':'Newcastle United'},

{'match_id':7079, 'season':'2017/2018', 'stage':8, 'date':'43022', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Leicester City', 'away_team':'West Bromwich Albion'},

{'match_id':7080, 'season':'2017/2018', 'stage':9, 'date':'43029', 'home_team_goal':0, 'away_team_goal':3, 'home_team':'West Ham United', 'away_team':'Brighton'},

{'match_id':7081, 'season':'2017/2018', 'stage':9, 'date':'43029', 'home_team_goal':4, 'away_team_goal':2, 'home_team':'Chelsea', 'away_team':'Watford'},

{'match_id':7082, 'season':'2017/2018', 'stage':9, 'date':'43029', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Huddersfield', 'away_team':'Manchester United'},

{'match_id':7083, 'season':'2017/2018', 'stage':9, 'date':'43029', 'home_team_goal':3, 'away_team_goal':0, 'home_team':'Manchester City', 'away_team':'Burnley'},

{'match_id':7084, 'season':'2017/2018', 'stage':9, 'date':'43029', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Newcastle United', 'away_team':'Crystal Palace'},

{'match_id':7085, 'season':'2017/2018', 'stage':9, 'date':'43029', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Stoke City', 'away_team':'Bournemouth'},

{'match_id':7086, 'season':'2017/2018', 'stage':9, 'date':'43029', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Swansea City', 'away_team':'Leicester City'},

{'match_id':7087, 'season':'2017/2018', 'stage':9, 'date':'43029', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Southampton', 'away_team':'West Bromwich Albion'},

{'match_id':7088, 'season':'2017/2018', 'stage':9, 'date':'43029', 'home_team_goal':2, 'away_team_goal':5, 'home_team':'Everton', 'away_team':'Arsenal'},

{'match_id':7089, 'season':'2017/2018', 'stage':9, 'date':'43029', 'home_team_goal':4, 'away_team_goal':1, 'home_team':'Tottenham Hotspur', 'away_team':'Liverpool'},

{'match_id':7090, 'season':'2017/2018', 'stage':10, 'date':'43036', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Manchester United', 'away_team':'Tottenham Hotspur'},

{'match_id':7091, 'season':'2017/2018', 'stage':10, 'date':'43036', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Arsenal', 'away_team':'Swansea City'},

{'match_id':7092, 'season':'2017/2018', 'stage':10, 'date':'43036', 'home_team_goal':2, 'away_team_goal':2, 'home_team':'Crystal Palace', 'away_team':'West Ham United'},

{'match_id':7093, 'season':'2017/2018', 'stage':10, 'date':'43036', 'home_team_goal':3, 'away_team_goal':0, 'home_team':'Liverpool', 'away_team':'Huddersfield'},

{'match_id':7094, 'season':'2017/2018', 'stage':10, 'date':'43036', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Watford', 'away_team':'Stoke City'},

{'match_id':7095, 'season':'2017/2018', 'stage':10, 'date':'43036', 'home_team_goal':2, 'away_team_goal':3, 'home_team':'West Bromwich Albion', 'away_team':'Manchester City'},

{'match_id':7096, 'season':'2017/2018', 'stage':10, 'date':'43036', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Bournemouth', 'away_team':'Chelsea'},

{'match_id':7097, 'season':'2017/2018', 'stage':10, 'date':'43036', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Brighton', 'away_team':'Southampton'},

{'match_id':7098, 'season':'2017/2018', 'stage':10, 'date':'43036', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Leicester City', 'away_team':'Everton'},

{'match_id':7099, 'season':'2017/2018', 'stage':10, 'date':'43036', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Burnley', 'away_team':'Newcastle United'},

{'match_id':7100, 'season':'2017/2018', 'stage':11, 'date':'43043', 'home_team_goal':2, 'away_team_goal':2, 'home_team':'Stoke City', 'away_team':'Leicester City'},

{'match_id':7101, 'season':'2017/2018', 'stage':11, 'date':'43043', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Huddersfield', 'away_team':'West Bromwich Albion'},

{'match_id':7102, 'season':'2017/2018', 'stage':11, 'date':'43043', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Newcastle United', 'away_team':'Bournemouth'},

{'match_id':7103, 'season':'2017/2018', 'stage':11, 'date':'43043', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Southampton', 'away_team':'Burnley'},

{'match_id':7104, 'season':'2017/2018', 'stage':11, 'date':'43043', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Swansea City', 'away_team':'Brighton'},

{'match_id':7105, 'season':'2017/2018', 'stage':11, 'date':'43043', 'home_team_goal':1, 'away_team_goal':4, 'home_team':'West Ham United', 'away_team':'Liverpool'},

{'match_id':7106, 'season':'2017/2018', 'stage':11, 'date':'43043', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Tottenham Hotspur', 'away_team':'Crystal Palace'},

{'match_id':7107, 'season':'2017/2018', 'stage':11, 'date':'43043', 'home_team_goal':3, 'away_team_goal':1, 'home_team':'Manchester City', 'away_team':'Arsenal'},

{'match_id':7108, 'season':'2017/2018', 'stage':11, 'date':'43043', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Chelsea', 'away_team':'Manchester United'},

{'match_id':7109, 'season':'2017/2018', 'stage':11, 'date':'43043', 'home_team_goal':3, 'away_team_goal':2, 'home_team':'Everton', 'away_team':'Watford'},

{'match_id':7110, 'season':'2017/2018', 'stage':12, 'date':'43057', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Arsenal', 'away_team':'Tottenham Hotspur'},

{'match_id':7111, 'season':'2017/2018', 'stage':12, 'date':'43057', 'home_team_goal':4, 'away_team_goal':0, 'home_team':'Bournemouth', 'away_team':'Huddersfield'},

{'match_id':7112, 'season':'2017/2018', 'stage':12, 'date':'43057', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Burnley', 'away_team':'Swansea City'},

{'match_id':7113, 'season':'2017/2018', 'stage':12, 'date':'43057', 'home_team_goal':2, 'away_team_goal':2, 'home_team':'Crystal Palace', 'away_team':'Everton'},

{'match_id':7114, 'season':'2017/2018', 'stage':12, 'date':'43057', 'home_team_goal':0, 'away_team_goal':2, 'home_team':'Leicester City', 'away_team':'Manchester City'},

{'match_id':7115, 'season':'2017/2018', 'stage':12, 'date':'43057', 'home_team_goal':3, 'away_team_goal':0, 'home_team':'Liverpool', 'away_team':'Southampton'},

{'match_id':7116, 'season':'2017/2018', 'stage':12, 'date':'43057', 'home_team_goal':0, 'away_team_goal':4, 'home_team':'West Bromwich Albion', 'away_team':'Chelsea'},

{'match_id':7117, 'season':'2017/2018', 'stage':12, 'date':'43057', 'home_team_goal':4, 'away_team_goal':1, 'home_team':'Manchester United', 'away_team':'Newcastle United'},

{'match_id':7118, 'season':'2017/2018', 'stage':12, 'date':'43057', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Watford', 'away_team':'West Ham United'},

{'match_id':7119, 'season':'2017/2018', 'stage':12, 'date':'43057', 'home_team_goal':2, 'away_team_goal':2, 'home_team':'Brighton', 'away_team':'Stoke City'},

{'match_id':7120, 'season':'2017/2018', 'stage':13, 'date':'43064', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'West Ham United', 'away_team':'Leicester City'},

{'match_id':7121, 'season':'2017/2018', 'stage':13, 'date':'43064', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Crystal Palace', 'away_team':'Stoke City'},

{'match_id':7122, 'season':'2017/2018', 'stage':13, 'date':'43064', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Manchester United', 'away_team':'Brighton'},

{'match_id':7123, 'season':'2017/2018', 'stage':13, 'date':'43064', 'home_team_goal':0, 'away_team_goal':3, 'home_team':'Newcastle United', 'away_team':'Watford'},

{'match_id':7124, 'season':'2017/2018', 'stage':13, 'date':'43064', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Swansea City', 'away_team':'Bournemouth'},

{'match_id':7125, 'season':'2017/2018', 'stage':13, 'date':'43064', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Tottenham Hotspur', 'away_team':'West Bromwich Albion'},

{'match_id':7126, 'season':'2017/2018', 'stage':13, 'date':'43064', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Liverpool', 'away_team':'Chelsea'},

{'match_id':7127, 'season':'2017/2018', 'stage':13, 'date':'43064', 'home_team_goal':4, 'away_team_goal':1, 'home_team':'Southampton', 'away_team':'Everton'},

{'match_id':7128, 'season':'2017/2018', 'stage':13, 'date':'43064', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Burnley', 'away_team':'Arsenal'},

{'match_id':7129, 'season':'2017/2018', 'stage':13, 'date':'43064', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Huddersfield', 'away_team':'Manchester City'},

{'match_id':7130, 'season':'2017/2018', 'stage':14, 'date':'43067', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Brighton', 'away_team':'Crystal Palace'},

{'match_id':7131, 'season':'2017/2018', 'stage':14, 'date':'43067', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Leicester City', 'away_team':'Tottenham Hotspur'},

{'match_id':7132, 'season':'2017/2018', 'stage':14, 'date':'43067', 'home_team_goal':2, 'away_team_goal':4, 'home_team':'Watford', 'away_team':'Manchester United'},

{'match_id':7133, 'season':'2017/2018', 'stage':14, 'date':'43067', 'home_team_goal':2, 'away_team_goal':2, 'home_team':'West Bromwich Albion', 'away_team':'Newcastle United'},

{'match_id':7134, 'season':'2017/2018', 'stage':14, 'date':'43067', 'home_team_goal':5, 'away_team_goal':0, 'home_team':'Arsenal', 'away_team':'Huddersfield'},

{'match_id':7135, 'season':'2017/2018', 'stage':14, 'date':'43067', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Bournemouth', 'away_team':'Burnley'},

{'match_id':7136, 'season':'2017/2018', 'stage':14, 'date':'43067', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Chelsea', 'away_team':'Swansea City'},

{'match_id':7137, 'season':'2017/2018', 'stage':14, 'date':'43067', 'home_team_goal':4, 'away_team_goal':0, 'home_team':'Everton', 'away_team':'West Ham United'},

{'match_id':7138, 'season':'2017/2018', 'stage':14, 'date':'43067', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Manchester City', 'away_team':'Southampton'},

{'match_id':7139, 'season':'2017/2018', 'stage':14, 'date':'43067', 'home_team_goal':0, 'away_team_goal':3, 'home_team':'Stoke City', 'away_team':'Liverpool'},

{'match_id':7140, 'season':'2017/2018', 'stage':15, 'date':'43071', 'home_team_goal':3, 'away_team_goal':1, 'home_team':'Chelsea', 'away_team':'Newcastle United'},

{'match_id':7141, 'season':'2017/2018', 'stage':15, 'date':'43071', 'home_team_goal':1, 'away_team_goal':5, 'home_team':'Brighton', 'away_team':'Liverpool'},

{'match_id':7142, 'season':'2017/2018', 'stage':15, 'date':'43071', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Everton', 'away_team':'Huddersfield'},

{'match_id':7143, 'season':'2017/2018', 'stage':15, 'date':'43071', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Leicester City', 'away_team':'Burnley'},

{'match_id':7144, 'season':'2017/2018', 'stage':15, 'date':'43071', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Stoke City', 'away_team':'Swansea City'},

{'match_id':7145, 'season':'2017/2018', 'stage':15, 'date':'43071', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Watford', 'away_team':'Tottenham Hotspur'},

{'match_id':7146, 'season':'2017/2018', 'stage':15, 'date':'43071', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'West Bromwich Albion', 'away_team':'Crystal Palace'},

{'match_id':7147, 'season':'2017/2018', 'stage':15, 'date':'43071', 'home_team_goal':1, 'away_team_goal':3, 'home_team':'Arsenal', 'away_team':'Manchester United'},

{'match_id':7148, 'season':'2017/2018', 'stage':15, 'date':'43071', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Bournemouth', 'away_team':'Southampton'},

{'match_id':7149, 'season':'2017/2018', 'stage':15, 'date':'43071', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Manchester City', 'away_team':'West Ham United'},

{'match_id':7150, 'season':'2017/2018', 'stage':16, 'date':'43078', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'West Ham United', 'away_team':'Chelsea'},

{'match_id':7151, 'season':'2017/2018', 'stage':16, 'date':'43078', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Burnley', 'away_team':'Watford'},

{'match_id':7152, 'season':'2017/2018', 'stage':16, 'date':'43078', 'home_team_goal':2, 'away_team_goal':2, 'home_team':'Crystal Palace', 'away_team':'Bournemouth'},

{'match_id':7153, 'season':'2017/2018', 'stage':16, 'date':'43078', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Huddersfield', 'away_team':'Brighton'},

{'match_id':7154, 'season':'2017/2018', 'stage':16, 'date':'43078', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Swansea City', 'away_team':'West Bromwich Albion'},

{'match_id':7155, 'season':'2017/2018', 'stage':16, 'date':'43078', 'home_team_goal':5, 'away_team_goal':1, 'home_team':'Tottenham Hotspur', 'away_team':'Stoke City'},

{'match_id':7156, 'season':'2017/2018', 'stage':16, 'date':'43078', 'home_team_goal':2, 'away_team_goal':3, 'home_team':'Newcastle United', 'away_team':'Leicester City'},

{'match_id':7157, 'season':'2017/2018', 'stage':16, 'date':'43078', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Southampton', 'away_team':'Arsenal'},

{'match_id':7158, 'season':'2017/2018', 'stage':16, 'date':'43078', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Liverpool', 'away_team':'Everton'},

{'match_id':7159, 'season':'2017/2018', 'stage':16, 'date':'43078', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Manchester United', 'away_team':'Manchester City'},

{'match_id':7160, 'season':'2017/2018', 'stage':17, 'date':'43081', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Burnley', 'away_team':'Stoke City'},

{'match_id':7161, 'season':'2017/2018', 'stage':17, 'date':'43081', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Crystal Palace', 'away_team':'Watford'},

{'match_id':7162, 'season':'2017/2018', 'stage':17, 'date':'43081', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Huddersfield', 'away_team':'Chelsea'},

{'match_id':7163, 'season':'2017/2018', 'stage':17, 'date':'43081', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Newcastle United', 'away_team':'Everton'},

{'match_id':7164, 'season':'2017/2018', 'stage':17, 'date':'43081', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Southampton', 'away_team':'Leicester City'},

{'match_id':7165, 'season':'2017/2018', 'stage':17, 'date':'43081', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Swansea City', 'away_team':'Manchester City'},

{'match_id':7166, 'season':'2017/2018', 'stage':17, 'date':'43081', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Liverpool', 'away_team':'West Bromwich Albion'},

{'match_id':7167, 'season':'2017/2018', 'stage':17, 'date':'43081', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Manchester United', 'away_team':'Bournemouth'},

{'match_id':7168, 'season':'2017/2018', 'stage':17, 'date':'43081', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Tottenham Hotspur', 'away_team':'Brighton'},

{'match_id':7169, 'season':'2017/2018', 'stage':17, 'date':'43081', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'West Ham United', 'away_team':'Arsenal'}

]

latest_matches = pd.DataFrame(latest_match_data, columns=['match_id', 'season', 'stage', 'date',

                                                          'home_team_goal', 'away_team_goal',

                                                          'home_team','away_team'])

latest_matches.head(20)



# Add to full training data to predict current season

matches = pd.concat([matches, latest_matches])

matches = matches.reset_index(drop=True)



# Create a full set of match data that can be used with feature engineering later

full_matches = matches.copy()

full_matches.tail(10)
# Only use the seasons we require - Optimum appears to be for season 12/13 onwards



unique_seasons = pd.Series(matches['season'].unique())

exclude_seasons = pd.Series(['2008/2009', '2009/2010', '2010/2011', '2011/2012'])

include_seasons = unique_seasons[~unique_seasons.isin(exclude_seasons)]

full_matches = full_matches.loc[full_matches['season'].isin(include_seasons)]

#full_matches.reset_index(drop=True, inplace=True)

full_matches
# State the result as home or away win/draw/lose - 6 possibilites

# Create a binary result

def determine_result(match_list):

    match_list['home_win'] = np.where(match_list['home_team_goal'] > match_list['away_team_goal'], 1, 0)

    match_list['home_draw'] = np.where(match_list['home_team_goal'] == match_list['away_team_goal'], 1, 0)

    match_list['home_lose'] = np.where(match_list['home_team_goal'] < match_list['away_team_goal'], 1, 0)

    match_list['away_win'] = np.where(match_list['home_team_goal'] < match_list['away_team_goal'], 1, 0)

    match_list['away_draw'] = np.where(match_list['home_team_goal'] == match_list['away_team_goal'], 1, 0)

    match_list['away_lose'] = np.where(match_list['home_team_goal'] > match_list['away_team_goal'], 1, 0)



# Set up the matches data how I need it



# Add binary feature for W/D/L home and away

determine_result(full_matches)



# Sort in date order

full_matches.sort_values(by='date', inplace=True)



full_matches.head()
# Cope with newly promoted teams with limited or no stats

team_data = {'team':['West Bromwich Albion', 'Stoke City', 'Hull City',

                     'Wolverhampton Wanderers', 'Birmingham City', 'Burnley',

                     'Newcastle United', 'West Bromwich Albion', 'Blackpool',

                     'Queens Park Rangers', 'Norwich City', 'Swansea City',

                     'Reading', 'Southampton', 'West Ham United',

                     'Cardiff City', 'Crystal Palace', 'Hull City',

                     'Leicester City', 'Burnley', 'Queens Park Rangers',

                     'Bournemouth', 'Watford', 'Norwich City',

                     'Burnley', 'Middlesbrough', 'Hull City',

                     'Brighton', 'Newcastle United', 'Huddersfield'

                    ],

             'season':["2008/2009", "2008/2009", "2008/2009",

                       "2009/2010", "2009/2010", "2009/2010",

                       "2010/2011", "2010/2011", "2010/2011",

                       "2011/2012", "2011/2012", "2011/2012",

                       "2012/2013", "2012/2013", "2012/2013",

                       "2013/2014", "2013/2014", "2013/2014",

                       "2014/2015", "2014/2015", "2014/2015",

                       "2015/2016", "2015/2016", "2015/2016",

                       "2016/2017", "2016/2017", "2016/2017",

                       "2017/2018", "2017/2018", "2017/2018"

                      ]

            }

new_teams = pd.DataFrame(team_data, columns=['team', 'season'])

new_teams.head()

# Cater for new teams by setting the new team for that season to a generic name



for index, row in new_teams.iterrows():

    for index1, row1 in full_matches.iterrows():

        if (row1['home_team'] == row['team']) & (row1['season'] == row['season']):

            full_matches.loc[index1, 'home_team'] = 'Promoted'

        if (row1['away_team'] == row['team']) & (row1['season'] == row['season']):

            full_matches.loc[index1, 'away_team'] = 'Promoted'



full_matches.head(20)
full_match_features = pd.DataFrame(full_matches[['season', 'stage']])

                                   #,

                                   #columns=[['season', 'stage']])

full_match_features
# Convert home & team into a binary feature, ie Arsenal_h or Arsenal_a

# Need all seasons data for team binary feature

full_match_features = pd.DataFrame(full_matches[['season', 'stage']])

                                   #columns=[['season', 'stage']])



full_match_features = pd.concat([full_match_features, pd.get_dummies(full_matches['home_team']).rename(columns=lambda x: str(x) + '_h')],

                                axis=1)

full_match_features = pd.concat([full_match_features, pd.get_dummies(full_matches['away_team']).rename(columns=lambda x: str(x) + '_a')],

                         axis=1)



full_match_features.head()
# To predict this season (1, this week only, remove this week from training set

train_match_features = full_match_features.loc[(full_match_features['season'] != this_season) | 

                                              (full_match_features['season'] == this_season) &

                                                  (full_match_features['stage'] < this_week)].copy()



train_match_features.drop(['season'], axis=1, inplace=True)

train_match_features.tail()
# Get the target results for training



# Function to determine whether result is a win/draw/lose

def determine_home_result(match):

    if match['home_team_goal'] > match['away_team_goal']:

        return 'win'

    elif match['home_team_goal'] < match['away_team_goal']:

        return 'lose'

    else:

        return 'draw'





#  Add the home team result column to the matches dataframe

full_matches['home_team_result'] = full_matches.apply(determine_home_result, axis=1)



# To predict this season, this week, remove latest week from training results

train_matches = full_matches.loc[(full_matches['season'] != this_season) | 

                                              (full_matches['season'] == this_season) &

                                                  (full_matches['stage'] < this_week)].copy()



targets = train_matches['home_team_result'].values

train_matches.tail()
# Get the test matches in correct format:



# Predict this season this week

test_match_features = full_match_features.loc[(full_match_features['season'] == this_season) &

                                              (full_match_features['stage'] == this_week)].copy()



test_match_features.drop(['season'], axis=1, inplace=True)

test_match_features
# Don't have the target results yet but I have entered dummy data of 0-0 draws. 

# Then re-run with actual scores after they are played for comparison

model_test_matches = full_matches.loc[(full_matches['season'] == this_season) & 

                                     (full_matches['stage'] == this_week)].copy()



model_test_matches = model_test_matches.reset_index(drop=True)

model_test_matches
# Function to determine whether the highest prediction is for win/draw/lose



def predict_home_result(match):

    if (match['win'] >= match['draw']) & (match['win'] >= match['lose']):

        return 'win' # Favour a home win if probability equal

    elif (match['lose'] > match['win']) & (match['lose'] > match['draw']):

        return 'lose'

    else:

        return 'draw'
# Train, then predict 

model = MultinomialNB()



model.fit(train_match_features.values, targets)

predicted = model.predict_proba(test_match_features.values)



# Format the output into a DF with columns

predicted_table = pd.DataFrame(predicted,columns=['draw', 'lose', 'win'])



# Compare predicted with test actual results 

predicted_table['predict_res'] = predicted_table.apply(predict_home_result, axis=1)

predicted_table['actual_res'] = model_test_matches['home_team_result']



# Straight comparison - count of equal / total to get %

(predicted_table[predicted_table['predict_res'] 

                 == model_test_matches['home_team_result']].count()) / model_test_matches['home_team_result'].count()
compare_results = model_test_matches[['match_id', 'stage', 'home_team_goal', 

                                    'away_team_goal', 'home_team', 'away_team']].copy()

compare_results.rename(columns={'home_team_goal':'h_goal', 'away_team_goal':'a_goal'}, inplace=True)

compare_results = pd.concat([compare_results, predicted_table], axis=1)

compare_results