# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #for visualisation

import seaborn as sns #for visualisation

%matplotlib inline
deliveries=pd.read_csv("../input/deliveries.csv")

matches=pd.read_csv("../input/matches.csv")
deliveries.head(6)

#deliveries.describe()
matches.head(6)
#matches.info()

#matches.describe()



matches['season'].unique()

matches['team1'].unique()



def shorten_team_names(team):

    tokens = team.split(' ')

    name = ''

    for item in tokens:

        name += item[0]

    return name



team_df = pd.DataFrame(matches['team1'].unique(), columns=['teams'])



team_df['abbrv'] = team_df['teams'].apply(shorten_team_names)
matches['team1'] = matches['team1'].apply(shorten_team_names)

matches['team2'] = matches['team2'].apply(shorten_team_names)
#matches['winner'] = matches['winner'].apply(shorten_team_names)

def replace_na(winner):

    if winner != winner:

        return "nil"

    else:

        return winner



matches['winner'] = matches['winner'].apply(replace_na)
matches['winner'] = matches['winner'].apply(shorten_team_names)

matches.head()
matches['toss_winner'] = matches['toss_winner'].apply(shorten_team_names)

matches.head()
player_of_the_match= pd.pivot_table(matches,values=['player_of_match'],index=['season'],columns=['city'],aggfunc='count',margins=False)



plt.figure(figsize=(10,10))

sns.heatmap(player_of_the_match['player_of_match'],linewidths=.5,annot=True,vmin=0.01,cmap='YlGnBu')

plt.title('Number of player of the match in cities for particular year')