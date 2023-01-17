import pandas as pd

import numpy as np

import statsmodels.api as sm

import seaborn as sns

color = sns.color_palette()



import matplotlib.pyplot as plt

%matplotlib inline



from ggplot import *
#players with social media influence power, wiki, twitter & PIE, performance

sp_df = pd.read_csv("../input/nba_2017_players_with_salary_wiki_twitter.csv");sp_df.head()
sp_df.drop(['Unnamed: 0'],axis=1, inplace = True)
sp_df.describe(include = "all").transpose()
# fillin NAN

sp_df['TWITTER_FAVORITE_COUNT'] = sp_df['TWITTER_FAVORITE_COUNT'].fillna(np.mean(sp_df['TWITTER_FAVORITE_COUNT'])).astype(int)

sp_df['TWITTER_RETWEET_COUNT'] = sp_df['TWITTER_RETWEET_COUNT'].fillna(np.mean(sp_df['TWITTER_RETWEET_COUNT'])).astype(int)
#correlation heatmap, selecting performance from 40 columns

fig, axis1 = plt.subplots(1,1,figsize=(20,15))

plt.title('Social Media Influence vs. Performance')

correlation = sp_df.corr()

sns.heatmap(correlation, xticklabels = correlation.columns.values, yticklabels = correlation.columns.values, cmap="jet")
# pairpolt based on corr heatmap > 0.4

subcol1 = ['TEAM','FG','FT','TOV','ORPM','PAGEVIEWS','TWITTER_FAVORITE_COUNT','TWITTER_RETWEET_COUNT']

sub_df = sp_df[subcol1]

sns.pairplot(sub_df,hue="TEAM")
subcol2= ['TEAM','RPM','WINS_RPM','PIE','POINTS','PAGEVIEWS','TWITTER_FAVORITE_COUNT','TWITTER_RETWEET_COUNT']

sub_df2 = sp_df[subcol2]

sns.pairplot(sub_df2,hue="TEAM")
#narrow down

subcol3= ['POSITION','PIE','POINTS','PAGEVIEWS','TWITTER_FAVORITE_COUNT','TWITTER_RETWEET_COUNT']

sub_df3 = sp_df[subcol3]

sns.pairplot(sub_df3,hue="POSITION")
# players social influence - pageviews

p = ggplot(sp_df,aes(x="POINTS", y="TWITTER_FAVORITE_COUNT", color="POSITION",size="PAGEVIEWS")) + geom_point()

p + xlab("POINTS") + ylab("TWITTER_FAV") + ggtitle("Performance and Social Influence")
# distribution of social media influence

fig, axis1 = plt.subplots(1,1,figsize=(20,15))

sns.distplot(sp_df['PAGEVIEWS'])

sns.distplot(sp_df['TWITTER_FAVORITE_COUNT'])

sns.distplot(sp_df['TWITTER_RETWEET_COUNT']);
# avg social influence for different PIE level 

sp_df['SOCIAL_INF']= sp_df['PAGEVIEWS'] + sp_df['TWITTER_FAVORITE_COUNT']+ sp_df['TWITTER_RETWEET_COUNT']

sp_df['PIE_ROUND']= sp_df['PIE'].round()

fig, axis1 = plt.subplots(1,1,figsize=(25,15))

avg_socialinf = sp_df[["SOCIAL_INF","PIE_ROUND"]].groupby(['PIE_ROUND'],as_index=False).mean()

sns.barplot( x="PIE_ROUND", y= "SOCIAL_INF", data= avg_socialinf )
