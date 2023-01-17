# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

color = sns.color_palette()

%matplotlib inline



import plotly.graph_objs as go

import plotly.offline as py



from IPython.display import HTML

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
qt100_at = pd.read_csv("../input/queryTimes_100_prod_AT.log",sep='|',header=None,usecols=[1,2,3,4],index_col=0,names=['datetime','query','duration','results'],parse_dates=True)

qt100_at.head(15)
qt100_bc = pd.read_csv("../input/queryTimes_100_prod_BC.log",sep='|',header=None,usecols=[1,2,3,4],index_col=0,names=['datetime','query','duration','results'],parse_dates=True)

qt100_bc.head(10)
qt100_nb = pd.read_csv("../input/queryTimes_100_prod_NB.log",sep='|',header=None,usecols=[1,2,3,4],index_col=0,names=['datetime','query','duration','results'],parse_dates=True)

qt100_nb = qt100_nb.loc[qt100_nb['duration'] > 0]

qt100_nb.head(15)
qt20_at = pd.read_csv("../input/queryTimes_20_prod_AT.log",sep='|',header=None,usecols=[1,2,3,4],index_col=0,names=['datetime','query','duration','results'],parse_dates=True)

qt20_at.head(15)
grouped_qt100_at = qt100_at.groupby('query')

grouped_qt100_at.describe()
grouped_qt100_bc = qt100_bc.groupby('query')

grouped_qt100_bc.describe()
grouped_qt100_nb = qt100_nb.groupby('query')

grouped_qt100_nb.describe()
# set up groups for plotting by query

qt100_at_GetData = qt100_at.groupby('query').get_group('GetData(User)')

qt100_at_GetAllUsers = qt100_at.groupby('query').get_group('GetAllUsers')

qt100_at_QueryUserPosts = qt100_at.groupby('query').get_group('QueryUserPosts')



qt100_at_DBConnection = qt100_at.groupby('query').get_group('DBConnection')

qt100_at_FollowingPostsQuery = qt100_at.groupby('query').get_group('FollowingPostsQuery')

qt100_at_GroupFeedQuery = qt100_at.groupby('query').get_group('GroupFeedQuery')

qt100_at_NotificationTabsQuery = qt100_at.groupby('query').get_group('NotificationTabsQuery')

qt100_at_NotificationsQuery = qt100_at.groupby('query').get_group('NotificationsQuery(isMe)')

qt100_at_QueryChallengePosts = qt100_at.groupby('query').get_group('QueryChallengePosts')

qt100_at_QueryChallengeProgress = qt100_at.groupby('query').get_group('QueryChallengeProgress')

qt100_at_QueryGroupMembers = qt100_at.groupby('query').get_group('QueryGroupMembers')

qt100_at_QueryRelatedPosts = qt100_at.groupby('query').get_group('QueryRelatedPosts')

qt100_at_QueryUserGroups = qt100_at.groupby('query').get_group('QueryUserGroups')

qt100_at_RecentPostsQuery = qt100_at.groupby('query').get_group('RecentPostsQuery')

qt100_at_SearchQuery = qt100_at.groupby('query').get_group('SearchQuery')



qt100_at_QueryUserPosts.head()
# set up groups for plotting by query

qt100_bc_GetData = qt100_bc.groupby('query').get_group('GetData(User)')

qt100_bc_GetAllUsers = qt100_bc.groupby('query').get_group('GetAllUsers')

qt100_bc_QueryUserPosts = qt100_bc.groupby('query').get_group('QueryUserPosts')



qt100_bc_DBConnection = qt100_bc.groupby('query').get_group('DBConnection')

qt100_bc_FollowingPostsQuery = qt100_bc.groupby('query').get_group('FollowingPostsQuery')

qt100_bc_GroupFeedQuery = qt100_bc.groupby('query').get_group('GroupFeedQuery')

qt100_bc_NotificationTabsQuery = qt100_bc.groupby('query').get_group('NotificationTabsQuery')

qt100_bc_NotificationsQuery = qt100_bc.groupby('query').get_group('NotificationsQuery(isMe)')

qt100_bc_QueryChallengePosts = qt100_bc.groupby('query').get_group('QueryChallengePosts')

qt100_bc_QueryChallengeProgress = qt100_bc.groupby('query').get_group('QueryChallengeProgress')

qt100_bc_QueryGroupMembers = qt100_bc.groupby('query').get_group('QueryGroupMembers')

qt100_bc_QueryRelatedPosts = qt100_bc.groupby('query').get_group('QueryRelatedPosts')

qt100_bc_QueryUserGroups = qt100_bc.groupby('query').get_group('QueryUserGroups')

qt100_bc_RecentPostsQuery = qt100_bc.groupby('query').get_group('RecentPostsQuery')

qt100_bc_SearchQuery = qt100_bc.groupby('query').get_group('SearchQuery')



qt100_bc_QueryUserPosts.head()
# set up groups for plotting by query

qt100_nb_GetData = qt100_nb.groupby('query').get_group('GetData(User)')

qt100_nb_GetAllUsers = qt100_nb.groupby('query').get_group('GetAllUsers')

qt100_nb_QueryUserPosts = qt100_nb.groupby('query').get_group('QueryUserPosts')



qt100_nb_DBConnection = qt100_nb.groupby('query').get_group('DBConnection')

qt100_nb_FollowingPostsQuery = qt100_nb.groupby('query').get_group('FollowingPostsQuery')

qt100_nb_GroupFeedQuery = qt100_nb.groupby('query').get_group('GroupFeedQuery')

qt100_nb_NotificationTabsQuery = qt100_nb.groupby('query').get_group('NotificationTabsQuery')

qt100_nb_NotificationsQuery = qt100_nb.groupby('query').get_group('NotificationsQuery(isMe)')

qt100_nb_QueryChallengePosts = qt100_nb.groupby('query').get_group('QueryChallengePosts')

qt100_nb_QueryChallengeProgress = qt100_nb.groupby('query').get_group('QueryChallengeProgress')

qt100_nb_QueryGroupMembers = qt100_nb.groupby('query').get_group('QueryGroupMembers')

qt100_nb_QueryRelatedPosts = qt100_nb.groupby('query').get_group('QueryRelatedPosts')

qt100_nb_QueryUserGroups = qt100_nb.groupby('query').get_group('QueryUserGroups')

qt100_nb_RecentPostsQuery = qt100_nb.groupby('query').get_group('RecentPostsQuery')

qt100_nb_SearchQuery = qt100_nb.groupby('query').get_group('SearchQuery')



qt100_nb_QueryUserPosts.head()
plt.figure(figsize=(12,9))

qt100_at_QueryUserPosts.duration.plot(label='QueryUserPosts')

qt100_at_GetAllUsers.duration.plot(label='GetAllUsers')

qt100_at_GetData.duration.plot(label='GetData')

plt.legend()

plt.show()
plt.figure(figsize=(12,9))

qt100_bc_QueryUserPosts.duration.plot(label='QueryUserPosts')

qt100_bc_GetAllUsers.duration.plot(label='GetAllUsers')

qt100_bc_GetData.duration.plot(label='GetData')

plt.legend()

plt.show()
plt.figure(figsize=(12,9))

qt100_nb_QueryUserPosts.duration.plot(label='QueryUserPosts')

qt100_nb_GetAllUsers.duration.plot(label='GetAllUsers')

qt100_nb_GetData.duration.plot(label='GetData')

plt.legend()

plt.show()
trace0 = go.Scatter(

    x = qt100_at_RecentPostsQuery.index,

    y = qt100_at_RecentPostsQuery.duration,

    name = 'AT RecentPostsQuery'

)



trace1 = go.Scatter(

    x = qt100_at_QueryChallengePosts.index,

    y = qt100_at_QueryChallengePosts.duration,

    name = 'AT QueryChallengePosts'

)



trace2 = go.Scatter(

    x = qt100_at_DBConnection.index,

    y = qt100_at_DBConnection.duration,

    name = 'AT DBConnection'

)



data = [trace0, trace1, trace2]

py.iplot(data, filename='at-query-times')
trace0 = go.Scatter(

    x = qt100_bc_RecentPostsQuery.index,

    y = qt100_bc_RecentPostsQuery.duration,

    name = 'BC RecentPostsQuery'

)



trace1 = go.Scatter(

    x = qt100_bc_QueryChallengePosts.index,

    y = qt100_bc_QueryChallengePosts.duration,

    name = 'BC QueryChallengePosts'

)



trace2 = go.Scatter(

    x = qt100_bc_DBConnection.index,

    y = qt100_bc_DBConnection.duration,

    name = 'BC DBConnection'

)



data = [trace0, trace1, trace2]

py.iplot(data, filename='bc-query-times')
trace0 = go.Scatter(

    x = qt100_nb_RecentPostsQuery.index,

    y = qt100_nb_RecentPostsQuery.duration,

    name = 'NB RecentPostsQuery'

)



trace1 = go.Scatter(

    x = qt100_nb_QueryChallengePosts.index,

    y = qt100_nb_QueryChallengePosts.duration,

    name = 'NB QueryChallengePosts'

)



trace2 = go.Scatter(

    x = qt100_nb_DBConnection.index,

    y = qt100_nb_DBConnection.duration,

    name = 'NB DBConnection'

)



data = [trace0, trace1, trace2]

py.iplot(data, filename='nb-query-times')
hourly_at_GetData = qt100_at_GetData.resample('H').mean()



hourly_at_GetAllUsers = qt100_at_GetAllUsers.resample('H').mean()

hourly_at_QueryUserPosts = qt100_at_QueryUserPosts.resample('H').mean()

hourly_at_DBConnection = qt100_at_DBConnection.resample('H').mean()

hourly_at_FollowingPostsQuery = qt100_at_FollowingPostsQuery.resample('H').mean()

hourly_at_GroupFeedQuery = qt100_at_GroupFeedQuery.resample('H').mean()

hourly_at_NotificationTabsQuery = qt100_at_NotificationTabsQuery.resample('H').mean()

hourly_at_NotificationsQuery = qt100_at_NotificationsQuery.resample('H').mean()

hourly_at_QueryChallengePosts = qt100_at_QueryChallengePosts.resample('H').mean()

hourly_at_QueryChallengeProgress = qt100_at_QueryChallengeProgress.resample('H').mean()

hourly_at_QueryGroupMembers = qt100_at_QueryGroupMembers.resample('H').mean()

hourly_at_QueryRelatedPosts = qt100_at_QueryRelatedPosts.resample('H').mean()

hourly_at_QueryUserGroups = qt100_at_QueryUserGroups.resample('H').mean()

hourly_at_RecentPostsQuery = qt100_at_RecentPostsQuery.resample('H').mean()

hourly_at_SearchQuery = qt100_at_SearchQuery.resample('H').mean()



hourly_at_SearchQuery.head()
plt.figure(figsize=(12,9))

hourly_at_DBConnection.duration.plot(label='DBConnection')

hourly_at_QueryChallengePosts.duration.plot(label='QueryChallengePosts')

hourly_at_RecentPostsQuery.duration.plot(label='RecentPostsQuery')

hourly_at_QueryUserPosts.duration.plot(label='QueryUserPosts')

hourly_at_GetAllUsers.duration.plot(label='GetAllUsers')

hourly_at_GetData.duration.plot(label='GetData')

plt.legend()

plt.show()
# resample hourly

hourly_bc = qt100_bc.resample('H').mean()

hourly_at = qt100_at.resample('H').mean()

hourly_nb = qt100_nb.resample('H').mean()

hourly_bc.head()
plt.figure(figsize=(12,9))

hourly_bc.duration.plot(label='BC')

hourly_at.duration.plot(label='AT')

hourly_nb.duration.plot(label='NB')

plt.legend()

plt.show()
# combine all results into single dataframe

frames = [qt100_nb,qt100_nb,qt100_nb]

qt_all = pd.concat(frames)

qt_all.describe()
qt_all_small = qt_all.loc[qt_all['results'] < 10]

qt_all_small.describe()
qt_all_medium = qt_all.loc[(qt_all['results'] >= 10) & (qt_all['results'] < 100)]

qt_all_medium.describe()
qt_all_100 = qt_all.loc[(qt_all['results'] >= 100) & (qt_all['results'] <= 105)]

qt_all_100.describe()
qt_all_large = qt_all.loc[(qt_all['results'] > 105)]

qt_all_large.describe()
hourly_small = qt_all_small.resample('H').mean()

hourly_medium = qt_all_medium.resample('H').mean()

hourly_100 = qt_all_100.resample('H').mean()

hourly_large = qt_all_large.resample('H').mean()
trace0 = go.Scatter(

    x = hourly_small.index,

    y = hourly_small.duration,

    name = '<10 Results'

)



trace1 = go.Scatter(

    x = hourly_medium.index,

    y = hourly_medium.duration,

    name = '10-40 Results'

)



trace2 = go.Scatter(

    x = hourly_100.index,

    y = hourly_100.duration,

    name = '100-105 Results'

)



trace3 = go.Scatter(

    x = hourly_large.index,

    y = hourly_large.duration,

    name = 'up to 4300 Results'

)



data = [trace0,trace1,trace2,trace3]

py.iplot(data, filename='nb-query-times')
trace0 = go.Scatter(

    x = qt100_nb_QueryUserPosts.index,

    y = qt100_nb_QueryUserPosts.duration,

    name = 'NB UserPosts'

)



trace1 = go.Scatter(

    x = qt100_at_QueryUserPosts.index,

    y = qt100_at_QueryUserPosts.duration,

    name = 'AT UserPosts'

)



trace2 = go.Scatter(

    x = qt100_bc_QueryUserPosts.index,

    y = qt100_bc_QueryUserPosts.duration,

    name = 'BC UserPosts'

)



data = [trace0, trace1, trace2]

py.iplot(data, filename='queryuserposts-times')
trace0 = go.Scatter(

    x = qt100_at_QueryChallengePosts.index,

    y = qt100_at_QueryChallengePosts.duration,

    name = 'AT ChallengePosts'

)



trace1 = go.Scatter(

    x = qt100_nb_QueryChallengePosts.index,

    y = qt100_nb_QueryChallengePosts.duration,

    name = 'NB ChallengePosts'

)



trace2 = go.Scatter(

    x = qt100_bc_QueryChallengePosts.index,

    y = qt100_bc_QueryChallengePosts.duration,

    name = 'BC ChallengePosts'

)



trace3 = go.Scatter(

    x = qt100_at_RecentPostsQuery.index,

    y = qt100_at_RecentPostsQuery.duration,

    name = 'AT RecentPosts'

)



trace4 = go.Scatter(

    x = qt100_nb_RecentPostsQuery.index,

    y = qt100_nb_RecentPostsQuery.duration,

    name = 'NB RecentPosts'

)



trace5 = go.Scatter(

    x = qt100_bc_RecentPostsQuery.index,

    y = qt100_bc_RecentPostsQuery.duration,

    name = 'BC RecentPosts'

)



data = [trace0, trace1, trace2, trace3, trace4, trace5]

py.iplot(data, filename='posts-times')
sns.distplot(qt100_bc_RecentPostsQuery.duration, kde=False);
sns.distplot(qt100_at_QueryUserPosts.duration, kde=False);
sns.distplot(qt100_bc_QueryChallengePosts.duration, kde=False);
sns.distplot(qt100_at_QueryChallengePosts.duration, kde=False);
sns.distplot(qt100_bc_GetData.duration, kde=False);
grouped_qt20_at = qt20_at.groupby('query')

grouped_qt20_at.describe()
# set up groups for plotting by query

qt20_at_GetData = qt20_at.groupby('query').get_group('GetData(User)')

qt20_at_GetAllUsers = qt20_at.groupby('query').get_group('GetAllUsers')

qt20_at_QueryUserPosts = qt20_at.groupby('query').get_group('QueryUserPosts')



qt20_at_DBConnection = qt20_at.groupby('query').get_group('DBConnection')

qt20_at_FollowingPostsQuery = qt20_at.groupby('query').get_group('FollowingPostsQuery')

qt20_at_GroupFeedQuery = qt20_at.groupby('query').get_group('GroupFeedQuery')

qt20_at_NotificationTabsQuery = qt20_at.groupby('query').get_group('NotificationTabsQuery')

qt20_at_NotificationsQuery = qt20_at.groupby('query').get_group('NotificationsQuery(isMe)')

qt20_at_QueryChallengePosts = qt20_at.groupby('query').get_group('QueryChallengePosts')

qt20_at_QueryChallengeProgress = qt20_at.groupby('query').get_group('QueryChallengeProgress')

qt20_at_QueryGroupMembers = qt20_at.groupby('query').get_group('QueryGroupMembers')

qt20_at_QueryRelatedPosts = qt20_at.groupby('query').get_group('QueryRelatedPosts')

qt20_at_QueryUserGroups = qt20_at.groupby('query').get_group('QueryUserGroups')

qt20_at_RecentPostsQuery = qt20_at.groupby('query').get_group('RecentPostsQuery')

qt20_at_SearchQuery = qt20_at.groupby('query').get_group('SearchQuery')



qt100_at_QueryUserPosts.head()
plt.figure(figsize=(12,9))

qt20_at_QueryUserPosts.duration.plot(label='QueryUserPosts')

qt20_at_GetAllUsers.duration.plot(label='GetAllUsers')

qt20_at_GetData.duration.plot(label='GetData')

plt.legend()

plt.show()
trace0 = go.Scatter(

    x = qt20_at_RecentPostsQuery.index,

    y = qt20_at_RecentPostsQuery.duration,

    name = 'AT RecentPostsQuery'

)



trace1 = go.Scatter(

    x = qt20_at_QueryChallengePosts.index,

    y = qt20_at_QueryChallengePosts.duration,

    name = 'AT QueryChallengePosts'

)



trace2 = go.Scatter(

    x = qt20_at_DBConnection.index,

    y = qt20_at_DBConnection.duration,

    name = 'AT DBConnection'

)



data = [trace0, trace1, trace2]

py.iplot(data, filename='at-query-times')
qt_20_small = qt20_at.loc[qt20_at['results'] < 10]

qt_20_small.describe()
qt_20_medium = qt20_at.loc[(qt20_at['results'] >= 10) & (qt20_at['results'] < 100)]

qt_20_medium.describe()
qt20_hourly_small = qt_20_small.resample('H').mean()

qt20_hourly_medium = qt_20_medium.resample('H').mean()



trace0 = go.Scatter(

    x = qt20_hourly_small.index,

    y = qt20_hourly_small.duration,

    name = '<10 Results'

)



trace1 = go.Scatter(

    x = qt20_hourly_medium.index,

    y = qt20_hourly_medium.duration,

    name = '10-40 Results'

)



data = [trace0,trace1]

py.iplot(data, filename='nb-query-times')
sns.distplot(qt20_hourly_small.duration, kde=False);
sns.distplot(qt20_hourly_medium.duration, kde=False);