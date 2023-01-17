import pandas as pd

import re

import numpy as np



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





pd.set_option('display.max_columns', 85)

pd.set_option('display.max_rows', 200)



df = pd.read_csv('../input/formmdi/mdi.csv')

df = df.drop('Timestamp', axis = 1)

df = df.replace(np.NaN, '')



print('Welcome to TwinFinder 1.0')

print('Enter details as specified below to find your twin.')

print('To skip filling a particular detail, press Enter.')

print('To enter more than one details for a particular field, separate them with a "|".')

print('Try skipping one or more details to get more matches if no matches are found.\n')



print('Enter background(engg/commerce/other): ')

bg = input()

print('Enter city: ')

city = input()

print('Enter favourite music band/artist: ')

musi = input()

print('Favourite sport/team: ')

spor = input()

print('Favourite movie/genre: ')

movi = input()

print('Favourite TV show: ')

show = input()

print('You like to read: ')

read = input()







filtback = (df['Background'].str.contains(bg, flags = re.IGNORECASE))

filtcity = (df['City'].str.contains(city, flags = re.IGNORECASE))

filtmusi = (df['Name 3 Fav music bands/artists'].str.contains(musi, flags = re.IGNORECASE))

filtmovi = (df['3 Favourite movies'].str.contains(movi, flags = re.IGNORECASE))

filtshow = (df['3 Favourite TV shows'].str.contains(show, flags = re.IGNORECASE))

filtspor = (df['1 sport you follow/play and 1 favorite team/player'].str.contains(spor, flags = re.IGNORECASE))

filtread = (df['3 Favourite things to read'].str.contains(read, flags = re.IGNORECASE))



df[filtback & filtcity & filtmusi & filtmovi & filtshow & filtspor & filtread]