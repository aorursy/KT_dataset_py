# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
us_videos = pd.read_csv("/kaggle/input/youtube-new/USvideos.csv")

us_videos
us_videos.columns
us_categories = pd.read_json("/kaggle/input/youtube-new/US_category_id.json")

us_categories = us_categories.drop(columns=['kind', 'etag'])

us_categories['kind']=us_categories['items'].map(lambda x: dict(x)['kind'])

us_categories['etag']=us_categories['items'].map(lambda x: dict(x)['etag'])

us_categories['category_id']=us_categories['items'].map(lambda x: dict(x)['id'])

us_categories['channelId']=us_categories['items'].map(lambda x: dict(x)['snippet']['channelId'])

us_categories['title']=us_categories['items'].map(lambda x: dict(x)['snippet']['title'])

us_categories['assignable']=us_categories['items'].map(lambda x: dict(x)['snippet']['assignable'])

us_categories=us_categories.drop(columns=['items'])

us_categories
us_vid_stats = us_videos.join(us_categories, on=['category_id'], how="inner", lsuffix="", rsuffix="_category")

us_vid_stats.head()
us_vid_stats = us_vid_stats[['views', 'likes', 'dislikes', 'comment_count']]

us_vid_stats.head()
us_vid_stats.describe()
us_vid_stats.corr()
import seaborn as sns
sns.scatterplot(data=us_vid_stats, x='views', y='likes')
sns.scatterplot(data=us_vid_stats, x='dislikes', y='likes')
sns.scatterplot(data=us_vid_stats, x='comment_count', y='likes')
X = us_vid_stats[["views", "comment_count"]].values

Y = us_vid_stats["likes"].values

print(X[:5])

print(Y[:5])
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=123)
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
lr = LinearRegression(normalize=True)

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("MSR: {}".format(mean_squared_error(y_pred, y_test)))

print("R2 Score: {}".format(r2_score(y_pred, y_test)))
regr_1 = DecisionTreeRegressor(max_depth=4)



regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),

                          n_estimators=300, random_state=123)



regr_1.fit(X_train, y_train)

regr_2.fit(X_train, y_train)



# Predict

y_1 = regr_1.predict(X_test)

y_2 = regr_2.predict(X_test)
print("Decision Tree MSR: {}".format(mean_squared_error(y_1, y_test)))

print("Decision Tree R2_Score: {}".format(r2_score(y_1, y_test)))

print("AdaBoostRegress MSR: {}".format(mean_squared_error(y_2, y_test)))

print("AdaBoostRegress R2_Score: {}".format(r2_score(y_2, y_test)))