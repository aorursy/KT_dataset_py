import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
color = sns.color_palette()
from IPython.core.display import display, HTML
from plotnine import *
display(HTML("<style>.container { width:100% !important; }</style>"))
%matplotlib inline
players_with_salary_df = pd.read_csv("../input/nba_2017_nba_players_with_salary.csv")
pie_df = pd.read_csv("../input/nba_2017_pie.csv")
players_with_salary_df = players_with_salary_df[['PLAYER','POSITION','MP','POINTS','SALARY_MILLIONS']]
pie_df = pie_df[['PLAYER','TEAM','AGE','GP','W','L','NETRTG','OFFRTG','DEFRTG','PIE']]
salary_df = pie_df.merge(players_with_salary_df, how="inner", on="PLAYER")
twitter_df =  pd.read_csv("../input/nba_2017_twitter_players.csv")
salary_df = salary_df.merge(twitter_df, how="inner", on="PLAYER")
salary_df.info()
plt.subplots(figsize=(15,10))
ax = plt.axes()
ax.set_title("NBA Player Correlation Heatmap:  2017 Season (STATS & SALARY)")
corr = salary_df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
results = smf.ols('POINTS ~SALARY_MILLIONS', data=salary_df).fit()
print(results.summary())
sns.lmplot(x="POINTS", y="SALARY_MILLIONS", data=salary_df)
import warnings
warnings.filterwarnings('ignore')
(
    ggplot(data=salary_df)
      + aes(y='POINTS', x='SALARY_MILLIONS')
      + aes(color='TEAM', shape='POSITION')
      + geom_point(alpha=0.5)
      + scale_x_log10()
      + facet_wrap('~type', nrow=2, ncol=1)
      + theme_classic()
)
