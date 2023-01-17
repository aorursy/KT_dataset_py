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
completedata = pd.read_csv("../input/nba_2016_2017_100.csv");completedata.head()
#Enlarge the value of PIE
completedata['PIE_HUNDREDS']=completedata['PIE']*100;completedata.head()
#Plot independents with dependents to see the relationship
#It makes sense to differentiate whether the player is avtive in Twitter using different color
sns.lmplot(x="PIE_HUNDREDS", y="TWITTER_FOLLOWER_COUNT_MILLIONS", data = completedata, hue = 'ACTIVE_TWITTER_LAST_YEAR',palette="husl")

sns.lmplot(x="PIE_HUNDREDS", y="TWITTER_FOLLOWER_COUNT_MILLIONS", data = completedata, hue = 'ACTIVE_TWITTER_LAST_YEAR',palette="husl")
sns.lmplot(x="SALARY_MILLIONS", y="TWITTER_FOLLOWER_COUNT_MILLIONS", data = completedata, hue = 'ACTIVE_TWITTER_LAST_YEAR',palette="husl")
sns.lmplot(x="W", y="TWITTER_FOLLOWER_COUNT_MILLIONS", data = completedata, hue = 'ACTIVE_TWITTER_LAST_YEAR',palette="husl")
sns.lmplot(x="PTS", y="TWITTER_FOLLOWER_COUNT_MILLIONS", data = completedata, hue = 'ACTIVE_TWITTER_LAST_YEAR',palette="husl")
#Run multi-variable linear regression
results = smf.ols('TWITTER_FOLLOWER_COUNT_MILLIONS ~ PIE_HUNDREDS + SALARY_MILLIONS + W + PTS', data=completedata).fit()

print(results.summary())

