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
all_play = pd.read_csv("../input/nba_2017_players_with_salary_wiki_twitter.csv", index_col=0)
all_play.head()
def plotCorrelationMatrix(df, graphWidth):

    filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()

plt.subplots(figsize=(20,20))

ax = plt.axes()

ax.set_title("NBA Player Correlation Heatmap: Stats")

corr = all_play.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values, cmap="Purples")
#Looking for relationship between Wins_RPM and other variables

sns.lmplot(x="SALARY_MILLIONS", y="WINS_RPM", data=all_play); 

sns.lmplot(x="PACE", y="WINS_RPM", data=all_play);

sns.lmplot(x="TWITTER_FAVORITE_COUNT", y="WINS_RPM", data=all_play);

sns.lmplot(x="TWITTER_RETWEET_COUNT", y="WINS_RPM", data=all_play);

sns.lmplot(x="PAGEVIEWS", y="WINS_RPM", data=all_play)
#Salary and Wins are related. Check if Points & Position has an effect

sns.swarmplot(x="POINTS", y="WINS_RPM", hue="POSITION" ,data=all_play)
#For specific Positions, does Pace of the player affect Wins

sns.lmplot(x="PACE", y="WINS_RPM", col="POSITION", hue="POSITION",data=all_play, col_wrap=2, size=3)
nRowsRead = 1000 

df3 = pd.read_csv("../input/nba_2017_players_with_salary_wiki_twitter.csv", delimiter=',', nrows = nRowsRead)

df3.dataframeName = 'Player Stats.csv'

nRow, nCol = df3.shape

print(f'There are {nRow} rows and {nCol} columns')
#Check the effect of Age

df3.head(5)



#Running a regression to understand significant effect of variables on WINS_RPM

r=smf.ols(formula='WINS_RPM ~ AGE + POINTS + PACE +POSITION', data=all_play).fit()

print(r.summary())