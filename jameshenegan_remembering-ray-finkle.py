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
df = pd.read_csv("/kaggle/input/superbowl-history-1967-2020/superbowl.csv")

df.head()
df['year']= df['Date'].apply(lambda x: x.split()[2])

df.head()
df.dtypes
df = df.astype({'year': 'int'})

df.dtypes
import matplotlib.pyplot as plt

style = dict(size = 10, color = 'gray')

plt.style.use('seaborn-whitegrid')

plt.rcParams["figure.figsize"] = (20,6)

plt.plot(df['year'], df['Loser Pts'], marker = "o")

plt.plot(df['year'], df['Winner Pts'], marker = "o")

plt.title("Number of Points Scored by the Winners and Losers of Each SuperBowl Game")

plt.xlabel("Year")

plt.ylabel("Number of Points")

plt.legend()

plt.annotate('Very close!', xy = (1991,18), xytext=(1992,5), arrowprops = dict(facecolor = 'green', shrink = 0.05))

plt.annotate('Blowout!', xy = (1990,55), xytext=(1992,55), arrowprops = dict(facecolor = 'green', shrink = 0.05))

plt.annotate('Wide Margins of Victory for a few years', xy = (1984,40), xytext=(1970,50), arrowprops = dict(facecolor = 'green', shrink = 0.05))

plt.annotate("Fairly Tight Margins of Victory Since the Mid 2000's", xy = (2004,35), xytext=(2010,50), ha = "center", arrowprops = dict(facecolor = 'green', shrink = 0.15))

plt.show()
df[df['year'] == 1991]
df['Margin'] = df['Winner Pts'] - df['Loser Pts']

df['Margin'].describe()
from scipy.interpolate import InterpolatedUnivariateSpline

df = df.sort_values('year')

ius = InterpolatedUnivariateSpline(df['year'],df['Margin'])

xi = np.linspace(1967,2020, 1000)

yi = ius(xi)



# Mean Margin

plt.plot(df['year'], np.ones(df['year'].shape[0])*13.907407)

# One Standard Deviation Above

plt.plot(df['year'], np.ones(df['year'].shape[0])*13.907407 + 10.314431)

# One Standard Deviation Below

plt.plot(df['year'], np.ones(df['year'].shape[0])*13.907407 - 10.314431)



plt.scatter(df['year'], df['Margin'])

plt.plot(xi, yi)



plt.title("Is there any pattern in 'Margin of Victory' over the years?")

plt.xlabel("Year")

plt.ylabel("Margin of Victory")

plt.show()
# entered by hand from wikipedia

winningConferenceEncoded = [0,0,1,1,1,0,1,1,1,1,1,0,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,1,1,1,1,0,1,0,0,0,1,0,1,1,1,0,1,1]

past70 = df[df['year'] >= 1970].copy()

past70['WinningConferenceEncoded'] = pd.Series(winningConferenceEncoded)

past70['LosingConferenceEncoded'] = 1 - past70['WinningConferenceEncoded']

past70['WinningConference'] = past70['WinningConferenceEncoded'].map({0:'NFC', 1:'AFC'})

past70['LosingConference'] = past70['LosingConferenceEncoded'].map({0:'NFC', 1:'AFC'})

past70.drop('WinningConferenceEncoded', axis = 1, inplace = True)

past70.drop('LosingConferenceEncoded', axis = 1, inplace = True)

past70.head()
winCols = ['WinningConference', 'Winner Pts' , 'year']

loseCols = ['LosingConference', 'Loser Pts' , 'year']





win_df = past70[winCols].copy()

win_df = win_df.rename(columns = {'WinningConference':'Conference', 'Winner Pts' :'Points', 'year':"Year"})

win_df['Win'] = 1



lose_df = past70[loseCols].copy()

lose_df = lose_df.rename(columns = {'LosingConference':'Conference', 'Loser Pts' :'Points', 'year':"Year"})

lose_df['Win'] = 0



conference_info = win_df.append(lose_df, ignore_index = True, sort = False)

conference_info.head()
nfc_info = conference_info[conference_info['Conference']  == 'NFC'].copy()

nfc_info.rename(columns = {'Points':'NFC'}, inplace = True)

nfc_info.sort_values('Year', inplace= True)



afc_info = conference_info[conference_info['Conference']  == 'AFC'].copy()

afc_info.rename(columns = {'Points':'AFC'}, inplace = True)

afc_info.sort_values('Year', inplace= True)


nfc_info['NFC Cumulative Wins'] = nfc_info['Win'].cumsum()

afc_info['AFC Cumulative Wins'] = afc_info['Win'].cumsum()
import matplotlib.pyplot as plt

style = dict(size = 10, color = 'gray')

plt.style.use('seaborn-whitegrid')

plt.rcParams["figure.figsize"] = (20,6)

plt.plot(nfc_info['Year'], nfc_info['NFC'], marker = "o")

plt.plot(afc_info['Year'], afc_info['AFC'], marker = "o")

plt.title("Number of Points Scored by the AFC and NFC of Each SuperBowl Game since 1970")

plt.xlabel("Year")

plt.ylabel("Number of Points")

plt.legend()

plt.annotate('The NFC Reigned for Several Years Here', xy = (1996,40), xytext=(1998,52), ha = "left", arrowprops = dict(facecolor = 'green', shrink = 0.05))

plt.show()

import matplotlib.pyplot as plt

style = dict(size = 10, color = 'gray')

plt.style.use('seaborn-whitegrid')

plt.rcParams["figure.figsize"] = (20,6)

plt.title("Number of Superbowl Wins Accumulated by the AFC and NFC since 1970")

plt.plot(nfc_info['Year'], nfc_info['NFC Cumulative Wins'], marker = "o")

plt.plot(afc_info['Year'], afc_info['AFC Cumulative Wins'], marker = "o")

plt.xlabel("Year")

plt.ylabel("Number of Wins")

plt.legend()

plt.show()

conference_info['Tally'] = 1

print(conference_info.groupby(['Conference', 'Win']).count()['Tally'])


