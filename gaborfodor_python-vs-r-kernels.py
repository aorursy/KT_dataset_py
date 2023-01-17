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
    def __init__(self, path='../input'):
        self.path = path

    def Competitions(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'Competitions.csv'), nrows=nrows)
        return df.rename(columns={'Id': 'CompetitionId'})

    def CompetitionTags(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'CompetitionTags.csv'), nrows=nrows)

    def Datasets(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'Datasets.csv'), nrows=nrows)

    def DatasetTags(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'DatasetTags.csv'), nrows=nrows)

    def DatasetVersions(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'DatasetVersions.csv'), nrows=nrows)

    def DatasetVotes(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'DatasetVotes.csv'), nrows=nrows)

    def DatasourceObjects(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'DatasourceObjects.csv'), nrows=nrows)

    def Datasources(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'Datasources.csv'), nrows=nrows)

    def DatasourceVersionObjectTables(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'DatasourceVersionObjectTables.csv'), nrows=nrows)

    def ForumMessages(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'ForumMessages.csv'), nrows=nrows)
        df['PostDate'] = pd.to_datetime(df['PostDate'])
        df['PostWeek'] = [date_to_first_day_of_week(pd.Timestamp(d).date()) for d in df.PostDate]
        return df.rename(columns={'Id': 'ForumMessageId'})

    def ForumMessageVotes(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'ForumMessageVotes.csv'), nrows=nrows)
        df['VoteDate'] = pd.to_datetime(df['VoteDate'])
        df['VoteWeek'] = [date_to_first_day_of_week(pd.Timestamp(d).date()) for d in df.VoteDate]
        return df

    def Forums(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'Forums.csv'), nrows=nrows).rename(columns={'Id': 'ForumId'})

    def ForumTopics(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'ForumTopics.csv'), nrows=nrows)
        return df.rename(columns={'Id': 'ForumTopicId'})

    def KernelLanguages(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'KernelLanguages.csv'), nrows=nrows)
        return df.rename(columns={'Id': 'KernelLanguageId', 'DisplayName': 'KernelLanguageName'})

    def Kernels(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'Kernels.csv'), nrows=nrows)
        return df.rename(columns={'Id': 'KernelId'})

    def KernelTags(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'KernelTags.csv'), nrows=nrows)

    def KernelVersionCompetitionSources(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'KernelVersionCompetitionSources.csv'), nrows=nrows)

    def KernelVersionDatasetSources(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'KernelVersionDatasetSources.csv'), nrows=nrows)

    def KernelVersionKernelSources(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'KernelVersionKernelSources.csv'), nrows=nrows)

    def KernelVersionOutputFiles(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'KernelVersionOutputFiles.csv'), nrows=nrows)

    def KernelVersions(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'KernelVersions.csv'), nrows=nrows)
        df['CreationDate'] = pd.to_datetime(df['CreationDate'])
        df['CreationWeek'] = [date_to_first_day_of_week(pd.Timestamp(d).date()) for d in df.CreationDate]
        return df.rename(columns={'Id': 'KernelVersionId'})

    def KernelVotes(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'KernelVotes.csv'), nrows=nrows)
        df['VoteDate'] = pd.to_datetime(df['VoteDate'])
        df['VoteWeek'] = [date_to_first_day_of_week(pd.Timestamp(d).date()) for d in df.VoteDate]
        return df

    def Medals(self):
        df = pd.DataFrame([
            [1, 'Gold', '#FFCE3F', '#A46A15'],
            [2, 'Silver', '#E6E6E6', '#787775'],
            [3, 'Bronze', '#EEB171', '#835036'],
        ], columns=['Medal', 'MedalName', 'MedalBody', 'MedalBorder'])
        return df

    def Organizations(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'Organizations.csv'), nrows=nrows)

    def Submissions(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'Submissions.csv'), nrows=nrows,
                         usecols=['SubmittedUserId', 'TeamId', 'SubmissionDate'])
        df['SubmissionDate'] = pd.to_datetime(df['SubmissionDate'])
        df['SubmissionWeek'] = [date_to_first_day_of_week(pd.Timestamp(d).date()) for d in df.SubmissionDate]
        return df

    def Tags(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'Tags.csv'), nrows=nrows)

    def TeamMemberships(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'TeamMemberships.csv'), nrows=nrows)

    def Teams(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'Teams.csv'), nrows=nrows)
        return df.rename(columns={'Id': 'TeamId'})

    def UserAchievements(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'UserAchievements.csv'), nrows=nrows)

    def UserFollowers(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'UserFollowers.csv'), nrows=nrows)

    def UserOrganizations(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'UserOrganizations.csv'), nrows=nrows)

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

    def get_weekly_forum_votes(self):
        fmv = self.ForumMessageVotes()
        fm = self.ForumMessages()
        ft = self.ForumTopics()
        f = self.Forums()
        fmv['cnt'] = 1
        weekly_message_votes = fmv.groupby(['ForumMessageId', 'VoteWeek'])[['cnt']].sum().reset_index()
        weekly_message_votes = weekly_message_votes.merge(fm[['ForumMessageId', 'ForumTopicId', 'Message']],
                                                          on='ForumMessageId')
        weekly_message_votes = weekly_message_votes.merge(ft[['ForumTopicId', 'ForumId', 'Title']], on='ForumTopicId')
        weekly_message_votes = weekly_message_votes.merge(f[['ForumId', 'Title']], on='ForumId',
                                                          suffixes=['Topic', 'Forum'])
        return weekly_message_votes

    def get_weekly_kernel_version_votes(self):
        kernel_votes = self.KernelVotes()
        kernel_versions = self.KernelVersions()
        kernels = self.Kernels()
        kernel_votes['cnt'] = 1
        weekly_kernel_votes = kernel_votes.groupby(['KernelVersionId', 'VoteWeek'])[['cnt']].sum().reset_index()
        weekly_kernel_votes = weekly_kernel_votes.merge(kernel_versions[['KernelVersionId', 'KernelId']],
                                                        on='KernelVersionId')
        weekly_kernel_votes = weekly_kernel_votes.merge(kernels[['KernelId', 'CurrentKernelVersionId']],
                                                        on='KernelId')
        weekly_kernel_votes = weekly_kernel_votes.merge(
            kernel_versions[['KernelVersionId', 'KernelLanguageId', 'AuthorUserId', 'Title']],
            left_on='CurrentKernelVersionId', right_on='KernelVersionId')
        return weekly_kernel_votes

    def get_kernel_vote_info(self):
        kernel_votes = self.KernelVotes()
        kernel_versions = self.KernelVersions()
        kernels = self.Kernels()[[
            'KernelId', 'CurrentKernelVersionId', 'AuthorUserId', 'Medal', 'CurrentUrlSlug', 'TotalVotes']]
        users = self.Users()[['UserId', 'PerformanceTier', 'DisplayName']]

        df = pd.merge(kernel_votes, users, on='UserId')
        df = df.merge(kernel_versions[['KernelVersionId', 'KernelId']], on='KernelVersionId')
        df = df.merge(kernels, on='KernelId')
        df = df.merge(kernel_versions[['KernelVersionId', 'KernelLanguageId', 'Title']],
                      left_on='CurrentKernelVersionId', right_on='KernelVersionId')
        df = df.drop(['KernelVersionId_x', 'CurrentKernelVersionId', 'KernelVersionId_y'], axis=1)
        df = df.merge(users, left_on='AuthorUserId', right_on='UserId', suffixes=['Voter', 'Author'])
        df = df.drop(['AuthorUserId', 'Id'], axis=1)
        df = df[df.UserIdVoter != df.UserIdAuthor]
        return df


def date_to_first_day_of_week(day: date) -> date:
    return day - timedelta(days=day.weekday())

start = dt.datetime.now()
START_DATE = dt.date(2016, 1, 1)
md = MetaData('../input/meta-kaggle')
kernel_versions = md.KernelVersions()
kernel_versions = kernel_versions.merge(md.KernelLanguages(), on='KernelLanguageId')
kernel_versions.shape
kernel_versions.tail()

weekly_kernels = kernel_versions.groupby(['CreationWeek', 'KernelLanguageName'])[['KernelId']].nunique().reset_index()
weekly_kernels.shape
weekly_kernels.tail()
python = weekly_kernels[weekly_kernels.KernelLanguageName == 'Python']
r = weekly_kernels[weekly_kernels.KernelLanguageName == 'R']
data = [
    go.Scatter(
        x=python.CreationWeek.values,
        y=python.KernelId.values,
        mode='lines',
        name='Python',
        line=dict(width=4, color='#5ac995')
    ),
    go.Scatter(
        x=r.CreationWeek.values,
        y=r.KernelId.values,
        mode='lines',
        name='R',
        line=dict(width=4, color='#007FB4')
    ),
]
layout = go.Layout(
    title='Weekly Modified Kernels',
    xaxis=dict(title='WeekStart', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Number of kernels (weekly)', ticklen=5, gridwidth=2),
    showlegend=True
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='weekly_kernels')
end = dt.datetime.now()
print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))