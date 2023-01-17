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
import warnings
warnings.filterwarnings('ignore')
attendance_valuation_elo_df = pd.read_csv("../input/nba_2017_att_val_elo.csv")
attendance_valuation_elo_df.drop(['Unnamed: 0', "GMS", "AVG",], inplace=True, axis=1)
attendance_valuation_elo_df.head()
salary_df = pd.read_csv("../input/nba_2017_salary.csv")
salary_df.rename(columns={"NAME": "PLAYER"}, inplace=True)
salary_df["SALARY_MILLIONS"] = round(salary_df["SALARY"]/1000000, 2)
salary_df.drop(["POSITION","SALARY"], inplace=True, axis=1)
salary_df.head()
wiki_df = pd.read_csv("../input/nba_2017_player_wikipedia.csv")
wiki_df.rename(columns={'names': 'PLAYER', "pageviews": "PAGEVIEWS"}, inplace=True)
wiki_df.drop(["timestamps"],inplace=True, axis=1)
mean_wiki_df = wiki_df.groupby("PLAYER").mean()
mean_wiki_df.head()
twitter_df = pd.read_csv("../input/nba_2017_twitter_players.csv")
twitter_df.head()
player_with_salary_twitter_df = salary_df.merge(twitter_df)
player_with_salary_wiki_twitter = pd.merge(player_with_salary_twitter_df, mean_wiki_df, 
                                           how='left',left_on='PLAYER', right_on='PLAYER')
player_with_salary_wiki_twitter.drop(['Unnamed: 0'], inplace=True, axis=1)
player_with_salary_wiki_twitter.head()
team_with_salary_wiki_twitter_df = player_with_salary_wiki_twitter.groupby("TEAM").mean()
team_with_salary_wiki_twitter_df.head()
nba_team_df = pd.merge(attendance_valuation_elo_df, team_with_salary_wiki_twitter_df, 
                       how='left',left_on='TEAM', right_on='TEAM')
nba_team_df.head()
f, ax = plt.subplots(figsize = (10,20))
sns.barplot('VALUE_MILLIONS', 'TEAM', data = nba_team_df, color="b")
plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("NBA Team Value Correlation Heatmap:  2016-2017 Season")
corr = nba_team_df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
           center=1)
valuations = nba_team_df.pivot("TEAM", "TOTAL", "VALUE_MILLIONS")
plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("NBA Team Total Attendance vs Valuation in Millions:  2016-2017 Season")
sns.heatmap(valuations,linewidths=.5, annot=True, fmt='g', cmap="YlGnBu")
ax=sns.lmplot(x="PCT", y="VALUE_MILLIONS", data=nba_team_df, hue="CONF", size=8)
ax.set(xlabel='WIN PERCENTAGE', ylabel='VALUATION IN MILLIONS', title="NBA Team Valuation vs Win Percentage:  2016-2017 Season")
results1 = smf.ols('VALUE_MILLIONS ~ TWITTER_FAVORITE_COUNT',
                  data = nba_team_df).fit()
results1.summary()
results2 = smf.ols('VALUE_MILLIONS ~ TWITTER_RETWEET_COUNT',
                  data = nba_team_df).fit()
results2.summary()
results1 = smf.ols('VALUE_MILLIONS ~ PAGEVIEWS',
                  data = nba_team_df).fit()
results1.summary()
nba_team_df.info()
nba_team_df['TWITTER_FAVORITE_COUNT'] = nba_team_df['TWITTER_FAVORITE_COUNT'].fillna(nba_team_df['TWITTER_FAVORITE_COUNT'].median())
nba_team_df['TWITTER_RETWEET_COUNT'] = nba_team_df['TWITTER_RETWEET_COUNT'].fillna(nba_team_df['TWITTER_RETWEET_COUNT'].median())
nba_team_df['PAGEVIEWS'] = nba_team_df['PAGEVIEWS'].fillna(nba_team_df['PAGEVIEWS'].median())
numerical_df = nba_team_df.loc[:,["TOTAL", "PCT", "VALUE_MILLIONS", "TWITTER_FAVORITE_COUNT", "TWITTER_RETWEET_COUNT", "PAGEVIEWS"]]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
print(scaler.fit(numerical_df))
print(scaler.transform(numerical_df))
from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3)
kmeans = k_means.fit(scaler.transform(numerical_df))
nba_team_df['cluster'] = kmeans.labels_
nba_team_df
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=0)
    km.fit(scaler.transform(numerical_df))
    distortions.append(km.inertia_)
    
plt.plot(range(1,11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.title("Team Valuation Elbow Method Cluster Analysis")
plt.show()