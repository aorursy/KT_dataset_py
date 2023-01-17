import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import datetime
import seaborn as sns
color = sns.color_palette()
%matplotlib inline
df = pd.read_csv("../input/nba_2017_players_with_salary_wiki_twitter.csv")
df.head()
#plot the correlation 
plt.subplots(figsize=(12,10))
ax = plt.axes()
ax.set_title("NBA Player Correlation Heatmap:  nba_2017_players_with_salary_wiki_twitter")
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
corr.style.background_gradient().set_precision(2)
#Plot a simple linear relationship between salary and wikipedia pageviews
sns.lmplot(x="SALARY_MILLIONS", y="PAGEVIEWS", data=df)
#plot a simple linear relationship between salary and the number of times a Tweet by this player gets favorited, on average.
sns.lmplot(x="SALARY_MILLIONS", y="TWITTER_FAVORITE_COUNT", data=df)
#plot a simple linear relationship between salary and the number of times a Tweet by this player gets retweeted, on average.
sns.lmplot(x="SALARY_MILLIONS", y="TWITTER_RETWEET_COUNT", data=df)
#plot the correlation only between salary, pageviews, twitter favorate count, and twitter retweet count
df_four_elements = df[['SALARY_MILLIONS','PAGEVIEWS','TWITTER_FAVORITE_COUNT','TWITTER_RETWEET_COUNT']]
df_four_elements.head()
plt.subplots(figsize=(6,4))
ax = plt.axes()
ax.set_title("NBA Player Correlation Heatmap:  2017 NBA players (Wiki & Twitter & Salary)")
corr = df_four_elements.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            annot=True)
#sort the raw data by salary
df_sort = df.sort_values(by = ['SALARY_MILLIONS'],ascending=False)
df_bar = df_sort.head(10)
df_bar_chart = df_bar[['PLAYER','PAGEVIEWS','TWITTER_FAVORITE_COUNT','TWITTER_RETWEET_COUNT']]
df_bar_chart.head()
#plot side-by-side bar chart to show the top ten salary players' wiki pageviews, twitter favorite count, and twitter retweet count.
fig, ax1 = plt.subplots(figsize=(16, 10))
tidy = (
    df_bar_chart.set_index('PLAYER')
                .stack() 
                .reset_index() 
                .rename(columns={'level_1': 'Variable', 0: 'Value'})
)
sns.barplot(x='PLAYER', y='Value', hue='Variable', data=tidy, ax=ax1)
sns.despine(fig)