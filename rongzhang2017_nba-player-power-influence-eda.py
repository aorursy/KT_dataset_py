import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
color = sns.color_palette()
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
%matplotlib inline
from mpl_toolkits import mplot3d
import numpy as np
#import twitter dataset
twitter= pd.read_csv("../input/nba_2017_twitter_players.csv");
twitter.head()
#Import plear behavior dataset
plus = pd.read_csv("../input/nba_2017_real_plus_minus.csv");
plus.head()

#Import salary dataset
salary= pd.read_csv("../input/nba_2017_salary.csv");
salary.head()
#Manage salary dataset
salary = salary.rename(columns={'NAME' : 'PLAYER'})
salary["SALARY_MILLIONS"] = round(salary["SALARY"]/1000000, 2)
salary.drop(["SALARY","TEAM"], inplace=True, axis=1)
salary.head()
#manage plus dataset
plus.rename(columns={"NAME":"PLAYER", "WINS": "WINS_RPM"}, inplace=True)
players = []
for player in plus["PLAYER"]:
    plyr, _ = player.split(",")
    players.append(plyr)
plus.drop(["PLAYER"], inplace=True, axis=1)
plus["PLAYER"] = players
plus.head()
#Merger salary,plus and twitter dataset
total=twitter.merge(salary)
total=total.merge(plus,how="inner", on="PLAYER"); 
total.head(5)
total.info()
#1.plot the salary and wins_rpm to see if the higher salary, the higher wins_rpm
sns.lmplot(x="SALARY_MILLIONS", y="WINS_RPM", data=total)
#2.heatmap
plt.subplots(figsize=(7,5))
ax = plt.axes()
ax.set_title("NBA Player Correlation Heatmap:STATS & SALARY")
corr = total.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,cmap="BuPu")
#3 barplot
plt.figure(figsize = (10,4))
sns.barplot(x='POSITION', y='SALARY_MILLIONS', data=salary).set_title("Position Vs. Saraly")
plt.show()
#4 3D
fig = plt.figure()
ax = plt.axes(projection='3d')
# Generate the values
x_vals = total['TWITTER_FAVORITE_COUNT']
y_vals = total['TWITTER_RETWEET_COUNT']
z_vals = total['SALARY_MILLIONS']
# Plot the values
ax.scatter(x_vals, y_vals, z_vals, c = 'r', marker='o')
ax.set_xlabel('TWITTER_FAVORITE_COUNT')
ax.set_ylabel('TWITTER_RETWEET_COUNT')
ax.set_zlabel('SALARY_MILLIONS')
plt.show()
