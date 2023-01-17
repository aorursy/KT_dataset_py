import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from scipy import stats

from mlxtend.preprocessing import minmax_scaling
kickstarters_data = pd.read_csv("../input/ks-projects-201801.csv")

kickstarters_data.sample(5)
kickstarters_data.info()
usd_goal = kickstarters_data.usd_goal_real

scaled_data = minmax_scaling(usd_goal, columns = [0])

fig, ax=plt.subplots(1,2)

sns.distplot(kickstarters_data.usd_goal_real, ax=ax[0])

ax[0].set_title("Original Data")

sns.distplot(scaled_data, ax=ax[1])

ax[1].set_title("Scaled data")
index_of_positive_pledges = kickstarters_data.usd_pledged_real > 0

positive_pledges = kickstarters_data.usd_pledged_real.loc[index_of_positive_pledges]

normalized_pledges = stats.boxcox(positive_pledges)[0]

# plot both together to compare

fig, ax=plt.subplots(1,2)

sns.distplot(positive_pledges, ax=ax[0])

ax[0].set_title("Original Data")

sns.distplot(normalized_pledges, ax=ax[1])

ax[1].set_title("Normalized data")
from matplotlib.pyplot import xticks, pie

sns.countplot(x="main_category", data= kickstarters_data)

xticks(rotation=90)
kickstarters_data.groupby('main_category')['usd pledged',"usd_pledged_real"].mean().plot.bar()

print (kickstarters_data.groupby('main_category')['usd pledged', "usd_pledged_real"].mean())
kickstarters_data.groupby('main_category')['usd_goal_real',"usd_pledged_real"].mean().plot.bar()

plt.title('usd_goal_real vs usd_pledged_real')
kickstarters_data["usd pledged"]  = kickstarters_data["usd pledged"].fillna(kickstarters_data["usd_goal_real"])

kickstarters_data.groupby('main_category')['usd_goal_real',"usd_pledged_real"].mean().plot.line()

plt.xlabel('main_category')

plt.title('usd_goal_real vs usd_pledged_real')
figsize = (16,8)

sns.countplot(x="currency", data= kickstarters_data)

plt.title("different currrencies")
kickstarters_data['country'].value_counts()[:10].plot(kind='barh', figsize=(14,6) ,  title='Top 10 countries')               
sns.countplot(x = 'state', data = kickstarters_data)
(kickstarters_data.backers >= 1).value_counts().plot.pie(autopct='%0.0f%%', labels=None,explode = [0,.1], shadow=True,colors=['brown', 'green'])

plt.title('Kickstarter Backers')

plt.legend(['backers', 'no backers'], loc =1)
kickstarters_data = kickstarters_data.dropna()

kickstarters_data.isnull().sum()