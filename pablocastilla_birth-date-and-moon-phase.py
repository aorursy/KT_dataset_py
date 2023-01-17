import numpy as np

import pandas as pd 

import sqlite3

from datetime import datetime,date

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



LUNAR_CICLE = 29.53

FULL_MOON = date(1967, 1, 26) #first full moon in the dataset min year birth date

#calculate moon fase,

#it divides the moon phase in 30, 0 closer to full moon

def date_to_moon_fase(date_to_convert):

    delta=(date_to_convert-FULL_MOON).days

    days_in_cicle = delta%LUNAR_CICLE

    cicle_status = days_in_cicle/LUNAR_CICLE

    if cicle_status>0.5:

        cicle_status = 1 - cicle_status

    

    return int(30*cicle_status*2)

#full moons

print(date_to_moon_fase(date(2020,2,9)))

print(date_to_moon_fase(date(2020, 1, 10)))



#Jul 20 cycle

print(date_to_moon_fase(date(2020,7,18)))

print(date_to_moon_fase(date(2020,7,19)))

print(date_to_moon_fase(date(2020,7,20))) #new moon

print(date_to_moon_fase(date(2020,7,21)))

print(date_to_moon_fase(date(2020,7,22)))

print(date_to_moon_fase(date(2020,7,23)))

print(date_to_moon_fase(date(2020,7,24)))

print(date_to_moon_fase(date(2020,7,25)))

print(date_to_moon_fase(date(2020,7,26)))

print(date_to_moon_fase(date(2020,7,27)))

print(date_to_moon_fase(date(2020,7,28)))

print(date_to_moon_fase(date(2020,7,29)))

print(date_to_moon_fase(date(2020,7,30)))

print(date_to_moon_fase(date(2020,7,31)))

print(date_to_moon_fase(date(2020,8,1)))

print(date_to_moon_fase(date(2020,8,2)))

print(date_to_moon_fase(date(2020,8,3))) #full moon

print(date_to_moon_fase(date(2020,8,4)))

print(date_to_moon_fase(date(2020,8,5)))
database = '/kaggle/input/soccer/database.sqlite'

conn = sqlite3.connect(database)



number_of_players = pd.read_sql("""SELECT count(*)

                        FROM Player;""", conn)



print('number of people in the dataset: '+str(number_of_players))



birth_dates = pd.read_sql("""SELECT birthday

                        FROM Player;""", conn)







birth_dates = birth_dates['birthday'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')) 



birth_dates['moon_phase'] = birth_dates.apply(lambda x: date_to_moon_fase(datetime.date(x)))

plt.figure()

plt.title('histogram of moon phase at born date', fontsize=18)

_ = plt.hist(birth_dates['moon_phase'].tolist(), 30, alpha=0.5)