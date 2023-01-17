%matplotlib inline
import os
import pandas as pd
import datetime as dt
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 14
import seaborn as sns
sns.set_palette(sns.color_palette('tab20', 20))
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from datetime import date, timedelta
class MetaData():
    def __init__(self, path='/kaggle/input/meta-kaggle'):
        self.path = path

    def ForumMessages(self, usecols, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'ForumMessages.csv'), nrows=nrows, usecols=usecols)
        df['PostDate'] = pd.to_datetime(df['PostDate'])
        return df.rename(columns={'Id': 'ForumMessageId'})

    def ForumMessageVotes(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'ForumMessageVotes.csv'), nrows=nrows)
        df['VoteDate'] = pd.to_datetime(df['VoteDate'])
        return df

    def Forums(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'Forums.csv'), nrows=nrows).rename(columns={'Id': 'ForumId'})

    def ForumTopics(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'ForumTopics.csv'), nrows=nrows)
        return df.rename(columns={'Id': 'ForumTopicId'})

    def Users(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'Users.csv'), nrows=nrows)
        return df.rename(columns={'Id': 'UserId'})
    
    def PerformanceTiers(self):
        df = pd.DataFrame([
            [0, 'Novice', '#5ac995'],
            [1, 'Contributor', '#00BBFF'],
            [2, 'Expert', '#95628f'],
            [3, 'Master', '#f96517'],
            [4, 'GrandMaster', '#dca917'],
            [5, 'KaggleTeam', '#008abb'],
        ], columns=['PerformanceTier', 'PerformanceTierName', 'PerformanceTierColor'])
        return df
    
    def UserAchievements(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'UserAchievements.csv'), nrows=nrows)
    
    def Users(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'Users.csv'), nrows=nrows)
        return df.rename(columns={'Id': 'UserId'})

start = dt.datetime.now()

START_DATE = '2016-01-01'
md = MetaData('/kaggle/input/meta-kaggle')
fmv = md.ForumMessageVotes()
fm = md.ForumMessages(usecols=['Id', 'ForumTopicId', 'PostUserId', 'PostDate'])
ft = md.ForumTopics()

message_upvotes = fmv.groupby(['ForumMessageId', 'VoteDate']).size().reset_index()
message_upvotes.columns = ['ForumMessageId', 'VoteDate', 'Upvotes']
messages = pd.merge(fm, message_upvotes, on='ForumMessageId')
messages.head(2)

daily_topic_votes = messages.groupby(['ForumTopicId', 'VoteDate'])[['Upvotes']].sum().reset_index()
daily_topic_votes['TopicRank'] = daily_topic_votes.groupby('VoteDate')['Upvotes'].rank(ascending=False, method='first')
daily_topic_votes = daily_topic_votes.merge(ft[['ForumTopicId', 'Title']], on='ForumTopicId')
daily_topic_votes = daily_topic_votes.sort_values(by='Upvotes', ascending=False)
daily_topic_votes.head(5)


daily_top_topics = daily_topic_votes[daily_topic_votes.TopicRank == 1]
daily_top_topics = daily_top_topics.sort_values(by='VoteDate', ascending=False)
daily_top_topics.head()
daily_top_topics.Upvotes.sum()
daily_top_topics.shape
daily_top_topics = daily_top_topics[daily_top_topics.VoteDate > START_DATE]
data = [
    go.Scatter(
        y=daily_top_topics['Upvotes'].values,
        x=daily_top_topics.VoteDate.astype(str),
        mode='markers',
        marker=dict(sizemode='diameter',
                    sizeref=1,
                    size=np.sqrt(daily_top_topics['Upvotes'].values),
                    color=daily_top_topics['Upvotes'].values,
                    colorscale='Viridis',
                    showscale=True
                    ),
        text=daily_top_topics.Title.values,
    )
]
layout = go.Layout(
    autosize=True,
    title='Daily Hottest Forum Threads',
    hovermode='closest',
    xaxis=dict(title='VoteDate', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Number of votes (daily)', ticklen=5, gridwidth=2),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='dailyTopTopics')
daily_top_topics = daily_top_topics[daily_top_topics.VoteDate >= '2020-01-01']
data = [
    go.Scatter(
        y=daily_top_topics['Upvotes'].values,
        x=daily_top_topics.VoteDate.astype(str),
        mode='markers',
        marker=dict(sizemode='diameter',
                    sizeref=1,
                    size=np.sqrt(daily_top_topics['Upvotes'].values),
                    color=daily_top_topics['Upvotes'].values,
                    colorscale='Reds',
                    showscale=True
                    ),
        text=daily_top_topics.Title.values,
    )
]
layout = go.Layout(
    autosize=True,
    title='Daily Hottest Forum Threads (2020)',
    hovermode='closest',
    xaxis=dict(title='VoteDate', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Number of votes (daily)', ticklen=5, gridwidth=2),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='dailyTopTopics')
users = md.Users().merge(md.PerformanceTiers(), on='PerformanceTier')

upvotes = fmv.groupby(['ForumMessageId']).size().reset_index()
upvotes.columns = ['ForumMessageId', 'Upvotes']
messages = pd.merge(fm, upvotes, on='ForumMessageId')
messages = messages.merge(users[['UserId', 'DisplayName', 'PerformanceTierColor']], left_on='PostUserId', right_on='UserId')
dfdc = messages[messages.ForumTopicId == 157983].sort_values(by='PostDate')
dfdc['n'] = np.arange(len(dfdc))
zillow = messages[messages.ForumTopicId == 45770].sort_values(by='PostDate')
zillow['n'] = np.arange(len(zillow))
dfdc.head()
zillow.head()
zillow.shape, dfdc.shape
data = [
    go.Scatter(
        y=dfdc['n'].values,
        x=dfdc.PostDate,
        mode='markers',
        marker=dict(sizemode='diameter',
                    sizeref=0.4,
                    size=np.sqrt(dfdc['Upvotes'].values),
                    color=dfdc.PerformanceTierColor.values,
                    ),
        text=dfdc.DisplayName.values,
    ),
    go.Scatter(
        y=dfdc['n'].values,
        x=dfdc.PostDate,
        mode='lines',
    )
]
layout = go.Layout(
    autosize=True,
    title='Deepfake Detection Challenge - Disqualification Thread',
    hovermode='closest',
    xaxis=dict(title='Time', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Cumulative number of messages', ticklen=5, gridwidth=2, range=[-10, 280]),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='dailyTopTopics')
data = [
    go.Scatter(
        y=zillow['n'].values,
        x=zillow.PostDate,
        mode='markers',
        marker=dict(sizemode='diameter',
                    sizeref=0.4,
                    size=np.sqrt(zillow['Upvotes'].values),
                    color=zillow.PerformanceTierColor.values,
                    ),
        text=zillow.DisplayName.values,
    ),
    go.Scatter(
        y=zillow['n'].values,
        x=zillow.PostDate,
        mode='lines',
    )
]
layout = go.Layout(
    autosize=True,
    title='Zillow Prize - Disqualification Thread',
    hovermode='closest',
    xaxis=dict(title='Time', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Cumulative number of messages', ticklen=5, gridwidth=2, range=[-10, 280]),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='dailyTopTopics')
end = dt.datetime.now()
print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))