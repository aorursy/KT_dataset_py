

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import sqlite3

from scipy.stats import norm

import matplotlib



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
connection = sqlite3.connect('../input/database.sqlite')



query = '''

with x as (

SELECT



home_team_api_id,

away_team_api_id,



home_team_goal,

away_team_goal



FROM MATCH

where country_id='1729'

)





Select

b.team_long_name as home_team,

x.home_team_goal as home_score,

c.team_long_name as away_team,

x.away_team_goal as away_score

from x inner join Team b on x.home_team_api_id = b.team_api_id

inner join Team c on x.away_team_api_id = c.team_api_id;

'''



df = pd.read_sql(query,connection)





def winner(x):

    

    if x.home_score>x.away_score:

        return 1

    else:

        return 0



df['winner'] = df.apply( winner, axis = 1 )





r = df.groupby('home_team')





home_wins = r.winner.agg({"Wins":sum, "Games":len})



home_wins['Home_Win_Pct'] = home_wins.apply(lambda x: 100*round(x.Wins/x.Games,4), axis = 1)



def stat_test(x):

    

    

    phat = x.Wins/x.Games

    p0 = 0.5

    n = x.Games

    

    sigma = np.sqrt(p0*(1-p0)/n)

    

    z = (phat - p0)/sigma

    

    p = 1- norm.cdf(z)

    return p

    



home_wins['p'] = home_wins.apply( stat_test, axis = 1)

home_wins['Sig'] = ['yes' if j<0.05  else 'No' for j in home_wins.p]



ax = (-1*np.log(home_wins.p)).plot(kind = 'barh', label = '$-\ln(p)$', figsize = (5,8))

ax.set_ylabel('Team')



ax.axvline(-np.log(0.05), color = 'r')



ax.set_xticks([-np.log(0.05)])

ax.set_xticklabels(['Significant'], color = 'r')

ax.legend()

sns.despine()
home_wins[home_wins.Sig=='yes'].sort_values('Home_Win_Pct', ascending = False)