import pandas as pd

import statsmodels.api as sm

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn.cluster import KMeans

color = sns.color_palette()

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))

%matplotlib inline
nba_df = pd.read_csv("../input/nba_2017_players_with_salary_wiki_twitter.csv")
nba_df.head(10)
nba_df.info()
# 3 missing values in TWITTER_FAVORITE_COUNT and TWITTER_RETWEET_COUNT

# calculate the mean, sd and the number of missing value for these two columns

nba_twi_fav_mean = nba_df['TWITTER_FAVORITE_COUNT'].mean()

nba_twi_fav_sd = nba_df['TWITTER_FAVORITE_COUNT'].std()

nba_twi_fav_number = nba_df['TWITTER_FAVORITE_COUNT'].isnull().sum()



nba_twi_re_mean = nba_df['TWITTER_RETWEET_COUNT'].mean()

nba_twi_re_sd = nba_df['TWITTER_RETWEET_COUNT'].std()

nba_twi_re_number = nba_df['TWITTER_RETWEET_COUNT'].isnull().sum()



# generate random values between (mean - std) and (mean + std)

nba_twi_fav_random = np.random.randint(nba_twi_fav_mean - nba_twi_fav_sd, nba_twi_fav_mean + nba_twi_fav_sd, size = nba_twi_fav_number)

nba_twi_re_random = np.random.randint(nba_twi_re_mean - nba_twi_re_sd, nba_twi_re_mean + nba_twi_re_sd, size = nba_twi_re_number)



# drop missing values and fill them with random values

nba_df['TWITTER_FAVORITE_COUNT'].dropna()

nba_df['TWITTER_FAVORITE_COUNT'][np.isnan(nba_df['TWITTER_FAVORITE_COUNT'])] = nba_twi_fav_random

nba_df['TWITTER_RETWEET_COUNT'].dropna()

nba_df['TWITTER_RETWEET_COUNT'][np.isnan(nba_df['TWITTER_RETWEET_COUNT'])] = nba_twi_re_random



nba_corr = nba_df.corr()

columns = nba_corr.nlargest(10, 'WINS_RPM')['WINS_RPM'].index 

coff = np.corrcoef(nba_corr[columns].values.T) 

heatmap = sns.heatmap(coff, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, cmap = 'YlGnBu', yticklabels=columns.values, xticklabels=columns.values)
sns.lmplot(x="SALARY_MILLIONS", y="WINS_RPM", data=nba_df)
sns.lmplot(x="TWITTER_FAVORITE_COUNT", y="WINS_RPM", data=nba_df)
sns.lmplot(x="TWITTER_RETWEET_COUNT", y="WINS_RPM", data=nba_df)
results_fav = smf.ols('WINS_RPM ~TWITTER_FAVORITE_COUNT', data=nba_df).fit()

print (results_fav.summary())
results_retweet = smf.ols('WINS_RPM ~TWITTER_RETWEET_COUNT', data=nba_df).fit()

print (results_retweet.summary())
results_retweet = smf.ols('WINS_RPM ~SALARY_MILLIONS', data=nba_df).fit()

print (results_retweet.summary())