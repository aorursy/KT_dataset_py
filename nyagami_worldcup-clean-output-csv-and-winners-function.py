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
import pandas as pd

import numpy as np
matches = pd.read_csv('/kaggle/input/fifa-world-cup/WorldCupMatches.csv')

#A little cleaning of the data

matches.replace('NaN',np.nan, inplace=True)

matches.dropna(axis='index', how='all', inplace=True)

matches['Datetime'] = pd.to_datetime(matches['Datetime'])

matches.set_index('Year', inplace=True)

matches = matches.drop_duplicates(subset='MatchID')

matches.to_csv('WorldCupMatchesClean.csv')
finals = matches[matches['Stage']=='Final']

def winners(year):

    try:

        "This function returns the worldcup winner for whichever year is entered"

        if finals.loc[year,'Home Team Goals'] > finals.loc[year,'Away Team Goals']:

            winner = finals.loc[year,'Home Team Name'] + ' ' + 'won' + " " +str(int(finals.loc[year,'Home Team Goals'])) + ' : '+ str(int(finals.loc[year,'Away Team Goals'])) + ' '+finals.loc[year,'Away Team Name']



        elif finals.loc[year,'Home Team Goals'] < finals.loc[year,'Away Team Goals']:

            winner = finals.loc[year,'Away Team Name'] + ' ' + 'won' + " "  +str(int(finals.loc[year,'Away Team Goals']))+ ' : ' +str(int(finals.loc[year,'Home Team Goals'])) +' '+ finals.loc[year,'Home Team Name']



        else:

            winner = finals.loc[year,'Win conditions']



        print(year,'Worldcup winner :')

        print('\033[1m' + winner.upper() + '\033[1m')   

    except:

        print('Please enter a valid year')
#Example

print(winners(1998))

print(winners(1986))

print(winners(1996)) #no worldcup on this year