# import required libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/epl-results-19932018/EPL_Set.csv')

data
# Creating the columns for the data that is needed

data2 = pd.DataFrame([], columns = ['MatchPlayed','HomeTeam','AwayTeam', 'PFTHG','PFTAG','PFTG','PFTR','PHTHG','PHTAG','PHTG','PHTR'])

data2
#Populating the HomeTeam

data2['HomeTeam'] = data['HomeTeam']

data2
#Populating the AwayTeam

data2['AwayTeam'] = data['AwayTeam']

data2
#Populating the Previous Full Time Home Goal

# data2['PFTHG'] = [(data['FTHG'][index-1]) if index > 0 else np.NaN for index in range(0,len(data['FTHG'])) ]

# data2



# data[data['HomeTeam'].isin(['Arsenal','Coventry']) & data['AwayTeam'].isin(['Arsenal','Coventry'])].iloc[:]



for index in range(0,50):

    home_team = data.iloc[index]['HomeTeam']

    away_team = data.iloc[index]['AwayTeam']

    i = data['HomeTeam'].isin([home_team, away_team]) & data['AwayTeam'].isin([home_team, away_team])

    print(data[i].iloc[index])

    
#Populating the Previous Full Time Away Goal

data2['PFTAG'] = [(data['FTAG'][index-1]) if index > 0 else np.NaN for index in range(0,len(data['FTAG'])) ]

data2
#Populating the Previous Full Time Goal

data2['PFTG'] = [ (data2['PFTAG'][index] + data2['PFTHG'][index]) for index in range(0,len(data2['PFTAG'])) ]

data2
#Populating the Previous Full Time Results

data2['PFTR'] = [ (data['FTR'][index-1]) if index > 0 else '-' for index in range(0,len(data['FTR'])) ]

data2
#Populating the Previous Half Time Home Goal

data2['PHTHG'] = [ (data['HTHG'][index-1]) if index > 0 else np.NaN for index in range(0,len(data['HTHG'])) ]

data2
#Populating the Previous Half Time Away Goal

data2['PHTAG'] = [ (data['HTAG'][index-1]) if index > 0 else np.NaN for index in range(0,len(data['HTAG'])) ]

data2
#Populating the Previous Half Time Goal

data2['PHTG'] = [ (data2['PHTAG'][index] + data2['PHTHG'][index]) for index in range(0,len(data2['PHTAG'])) ]

data2
#Populating the Previous Half Time Result

data2['PHTR'] = [ (data['HTR'][index-1]) if index > 0 else np.NaN for index in range(0,len(data['HTR'])) ]

data2