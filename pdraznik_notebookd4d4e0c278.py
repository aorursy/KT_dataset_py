# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import math

import sqlite3

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import datetime as dt

import datetime

import sqlalchemy

from numpy.random import random

from sqlalchemy import create_engine

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
conn = sqlite3.connect('../input/database.sqlite')

c = conn.cursor()

print(c)
import datetime

def get_zodiac_of_date(date):

    zodiacs = [(120, 'Cap'), (218, 'Aqu'), (320, 'Pis'), (420, 'Ari'), (521, 'Tau'),

           (621, 'Gem'), (722, 'Can'), (823, 'Leo'), (923, 'Vir'), (1023, 'Lib'),

           (1122, 'Sco'), (1222, 'Sag'), (1231, 'Cap')]

    date_number = int("".join((str(date.month), '%02d' % date.day)))

    for z in zodiacs:

        if date_number <= z[0]:

            return z[1]

def get_zodiac_for_football_players(x):

    date  =  x.split(" ")[0]

    date = datetime.datetime.strptime(date, "%Y-%m-%d").date()

    return get_zodiac_of_date(date)

def get_age_for_football_players(x):

    date  =  x.split(" ")[0]

    today = datetime.datetime.strptime("2015-01-01", "%Y-%m-%d").date()

    born = datetime.datetime.strptime(date, "%Y-%m-%d").date()

    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

def get_overall_rating(x):

    global c

    all_rating = c.execute("""SELECT overall_rating FROM Player_Stats WHERE player_api_id = '%d' """ % (x)).fetchall()

    all_rating = np.array(all_rating,dtype=np.float)[:,0]

    mean_rating = np.nanmean(all_rating)

    return mean_rating

def get_current_team_and_country(x):

    global c

    all_rating = c.execute("""SELECT overall_rating FROM Player_Stats WHERE player_api_id = '%d' """ % (x)).fetchall()

    all_rating = np.array(all_rating,dtype=np.float)[:,0]

    rating = np.nanmean(all_rating)

    if (rating>1): 

        all_football_nums = reversed(range(1,12))

        for num in all_football_nums:

            all_team_id = c.execute("""SELECT home_team_api_id, country_id FROM Match WHERE home_player_%d = '%d'""" % (num,x)).fetchall()

            if len(all_team_id) > 0:

                number_unique_teams = len(np.unique(np.array(all_team_id)[:,0]))

                last_team_id = all_team_id[-1][0]

                last_country_id = all_team_id[-1][1]

                last_country = c.execute("""SELECT name FROM Country WHERE id = '%d'""" % (last_country_id)).fetchall()[0][0]

                last_team = c.execute("""SELECT team_long_name FROM Team WHERE team_api_id = '%d'""" % (last_team_id)).fetchall()[0][0]

                return last_team, last_country, number_unique_teams

    return None, None, 0

def get_position(x):

    global c

    all_rating = c.execute("""SELECT overall_rating FROM Player_Stats WHERE player_api_id = '%d' """ % (x)).fetchall()

    all_rating = np.array(all_rating,dtype=np.float)[:,0]

    rating = np.nanmean(all_rating)

    if (rating>1): 

        all_football_nums = reversed(range(1,12))

        for num in all_football_nums:

            all_y_coord = c.execute("""SELECT home_player_Y%d FROM Match WHERE home_player_%d = '%d'""" % (num,num,x)).fetchall()

            if len(all_y_coord) > 0:

                Y = np.array(all_y_coord,dtype=np.float)

                mean_y = np.nanmean(Y)

                if (mean_y >= 10.0):

                    return "for"

                elif (mean_y > 5):

                    return "mid"

                elif (mean_y > 1):

                    return "def"

                elif (mean_y == 1.0):

                    return "gk"

    return None
with sqlite3.connect('../input/database.sqlite') as con:

    sql = "SELECT * FROM `Player_Stats`"

    max_players_to_analyze = 1000

    

    players_data = pd.read_sql_query(sql, con)

    players_data = players_data.iloc[0:max_players_to_analyze]

    print( players_data )

    

    #players_data["zodiac"] = np.vectorize(get_zodiac_for_football_players)(players_data["birthday"])

    #print( players_data["zodiac"] )

    #players_data["rating"] = np.vectorize(get_overall_rating)(players_data["player_api_id"])

    '''

    players_data = players_data.iloc[0:max_players_to_analyze]

    players_data["zodiac"] = np.vectorize(get_zodiac_for_football_players)(players_data["birthday"])

    players_data["rating"] = np.vectorize(get_overall_rating)(players_data["player_api_id"])

    '''

    '''

    players_data["age"] = np.vectorize(get_age_for_football_players)(players_data["birthday"])

    players_data["team"], players_data["country"], players_data["num_uniq_team"] = np.vectorize(get_current_team_and_country)(players_data["player_api_id"])

    players_data["position"] = np.vectorize(get_position)(players_data["player_api_id"])

players_data.head()

'''