import seaborn as sns

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
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
players_salary = pd.read_csv('/kaggle/input/social-power-nba/nba_2017_salary.csv')

players_stats = pd.read_csv('/kaggle/input/social-power-nba/nba_2017_players_stats_combined.csv')

players_edsm = pd.read_csv('/kaggle/input/social-power-nba/nba_2017_endorsements.csv')

players_twitter = pd.read_csv('/kaggle/input/social-power-nba/nba_2017_twitter_players.csv')

players_RPM = pd.read_csv('/kaggle/input/social-power-nba/nba_2017_real_plus_minus.csv')

team_value = pd.read_csv('/kaggle/input/social-power-nba/nba_2017_att_val.csv')
team_value
players_salary
players_stats = players_stats.rename(columns={'PLAYER':'NAME'})

players_stats.head(8)
players_RPM = players_stats[['NAME','GP','MPG','ORPM','DRPM','RPM','WINS_RPM']]

players_RPM
players_sal_RPM = pd.merge(players_salary,players_RPM,how='inner',on='NAME')

players_sal_RPM
players_sal_RPM_TV = pd.merge(players_sal_RPM,team_value[['TEAM','VALUE_MILLIONS']],how='inner',on='TEAM')

players_sal_RPM_TV
corr1 = players_sal_RPM_TV.corr()

corr1
sns.heatmap(corr1)
results = smf.ols('SALARY ~MPG',data=players_sal_RPM_TV).fit()

results.summary()
results = smf.ols('SALARY ~WINS_RPM',data=players_sal_RPM_TV).fit()

results.summary()
results = smf.ols('SALARY ~RPM',data=players_sal_RPM_TV).fit()

results.summary()
players_sal_RPM.eval('WRPMpM = WINS_RPM / MPG',inplace = True)

players_sal_RPM.eval('RPMpM = RPM / MPG',inplace = True)

players_sal_RPM
corr = players_sal_RPM.corr()

corr
sns.heatmap(corr)
players_twitter = players_twitter.rename(columns={'PLAYER':'NAME'})
players_sal_twi = pd.merge(players_salary,players_twitter,how='inner',on='NAME')

players_sal_twi
corr2 = players_sal_twi.corr()

corr2
sns.heatmap(corr2)
players_edsm
players_edsm['Endorsement'] = players_edsm['ENDORSEMENT'].str[1:]

players_sal_edsm_twi = pd.merge(players_sal_twi,players_edsm[['NAME','Endorsement']],how='inner',on='NAME')

players_sal_edsm_twi
def convert(item):

    s = ''

    temp = item.strip().split(',')

    for i in range(len(temp)):

        s += temp[i]

    return float(s)

players_sal_edsm_twi['Endorsement'] = players_sal_edsm_twi['Endorsement'].map(convert)
players_sal_edsm_twi.info()
players_sal_edsm_twi
corr2 = players_sal_edsm_twi.corr()

corr2
sns.heatmap(corr2)
result = smf.ols('Endorsement ~TWITTER_FAVORITE_COUNT',data=players_sal_edsm_twi).fit()

result.summary()
players_data = pd.merge(players_sal_edsm_twi,players_sal_RPM[['NAME','MPG','ORPM','WINS_RPM','WRPMpM']],how='inner',on='NAME')

players_data.eval('ORPMpM = ORPM / MPG',inplace = True)

players_data
corr = players_data.corr()

corr
sns.heatmap(corr)