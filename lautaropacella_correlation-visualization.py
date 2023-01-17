# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import norm

from plotly.subplots import make_subplots

plt.style.use("ggplot")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/youtube-new/USvideos.csv")
df.head()
df.describe()
df.dtypes
w_ratings = df[df["ratings_disabled"]==False].copy()
corr = w_ratings[["views", "likes", "dislikes"]].corr(method = "spearman")

corr.style.background_gradient(cmap='coolwarm').set_precision(2)
fig, axes = plt.subplots(ncols=3,figsize=(30,5))

ax1 = sns.scatterplot(x =w_ratings["views"],y= w_ratings["likes"], ax = axes[0]).set_title("Views - Like Correlation")

ax2 = sns.scatterplot(x =w_ratings["views"],y=w_ratings["dislikes"], ax = axes[1]).set_title("Views - Dislikes Correlation")

ax3 = sns.scatterplot(x= w_ratings["likes"], y= w_ratings["dislikes"], ax = axes[2]).set_title("Likes - Dislikes Correlation")

fig.tight_layout()

plt.show()
fig, axes = plt.subplots(ncols=3,figsize=(20,8))

ax1 = sns.distplot(w_ratings["views"], ax = axes[0]).set_title("Views Distribution")

ax2 = sns.distplot(w_ratings["likes"], ax = axes[1]).set_title("Likes Distribution")

ax3 = sns.distplot(w_ratings["dislikes"], ax = axes[2]).set_title("Dislikes Distribution")

fig.tight_layout()

plt.show()
views_log = np.log(w_ratings.views+0.01)



fig = plt.figure(figsize=(15,5))



sns.distplot(views_log)

rv = norm(loc = views_log.mean(), scale = views_log.std())

x = np.arange(views_log.min(), views_log.max(), .1)



plt.plot(x, rv.pdf(x))



plt.show()


likes_log = np.log(w_ratings.likes+0.01)



fig = plt.figure(figsize=(15,5))



sns.distplot(likes_log)

rv = norm(loc = likes_log.mean(), scale = likes_log.std())

x = np.arange(likes_log.min(), likes_log.max(), .1)



plt.plot(x, rv.pdf(x))

plt.show()
dislikes_log = np.log(w_ratings.dislikes+0.01)



fig = plt.figure(figsize=(15,5))



sns.distplot(dislikes_log)

rv = norm(loc = dislikes_log.mean(), scale = dislikes_log.std())

x = np.arange(dislikes_log.min(), dislikes_log.max(), .1)



plt.plot(x, rv.pdf(x))

plt.show()
fig = px.scatter(x= views_log, y = likes_log, trendline ="ols", trendline_color_override="red")

fig.update_layout(title_text="Correlation between Logs of Views & Likes", xaxis_title = "Views Log", yaxis_title = "Likes Log")

fig.show()
fig = px.scatter(x= views_log, y = dislikes_log, trendline ="ols", trendline_color_override="red")

fig.update_layout(title_text="Correlation between Logs of Views & Dislikes", xaxis_title = "Views Log", yaxis_title = "Dislike Log")

fig.show()