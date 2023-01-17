import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def run_print(df):

    print(df.shape)

    print(df.tail())
competitions_df = pd.read_csv('/kaggle/input/meta-kaggle/Competitions.csv')

forum_message_df = pd.read_csv('/kaggle/input/meta-kaggle/ForumMessages.csv')

forum_df = pd.read_csv('/kaggle/input/meta-kaggle/Forums.csv')

forum_topic_df = pd.read_csv('/kaggle/input/meta-kaggle/ForumTopics.csv')

forum_message_votes_df = pd.read_csv('/kaggle/input/meta-kaggle/ForumMessageVotes.csv')

# users_df = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv')
competitions_df = competitions_df.rename(columns={'Id': 'CompetitionsId', 'Title': 'CompetitionsTitle'})

forum_message_df = forum_message_df.rename(columns={'Id': 'ForumMessageId'})

forum_df = forum_df.rename(columns={'Id': 'ForumId', 'Title': 'ForumTitle'})

forum_topic_df = forum_topic_df.rename(columns={'Id': 'ForumTopicId', 'Title': 'ForumTopicTitle'})

forum_message_votes_df = forum_message_votes_df.rename(columns={'Id': 'ForumMessageVotesId'})
use_featured_competitions_columns = ['CompetitionsId', 'CompetitionsTitle', 'ForumId', 'CompetitionTypeId', 'HostSegmentTitle', 

                                     'OnlyAllowKernelSubmissions', 'EvaluationAlgorithmAbbreviation',

                                     'TotalTeams', 'TotalCompetitors', 'TotalSubmissions']
featured_competitions_df = competitions_df[competitions_df['HostSegmentTitle']=='Featured']

featured_competitions_df = featured_competitions_df[use_featured_competitions_columns]

# run_print(featured_competitions_df)
forum_message_votes_df.head()
# merge_df = pd.merge(forum_message_df, forum_message_votes_df, on='ForumMessageId')

merge_df = pd.merge(forum_message_df, forum_topic_df, on='ForumTopicId')

merge_df = pd.merge(merge_df, forum_df, on='ForumId')

merge_df = pd.merge(merge_df, featured_competitions_df, on='ForumId')

print(merge_df.shape)

merge_df.tail()
RSNA_discussion_df = merge_df[merge_df['CompetitionsTitle']=='RSNA Intracranial Hemorrhage Detection']

print(RSNA_discussion_df.columns)

RSNA_discussion_df = RSNA_discussion_df[['PostUserId', 'Message', 'Medal', 'ForumTopicTitle', 'ForumTitle',

       'TotalViews', 'Score', 'CompetitionsTitle', 'CompetitionTypeId',]]

print(RSNA_discussion_df.shape)

RSNA_discussion_df.head()
kaggle_discussion_df = merge_df[['CompetitionsTitle', 'ForumTitle', 'ForumTopicTitle', 'PostUserId', 'ReplyToForumMessageId', 

                                 'Message', 'TotalViews', 'Score', 'Medal', 'TotalMessages', 'TotalReplies', 

                                 'CompetitionTypeId', 'OnlyAllowKernelSubmissions', 'EvaluationAlgorithmAbbreviation', 

                                 'TotalTeams', 'TotalCompetitors', 'TotalSubmissions',

                                ]]



# 'Medal' は 'Score' と相関しているため後ほど除く

print(kaggle_discussion_df.shape)

kaggle_discussion_df.head()
kaggle_discussion_df.to_csv('kaggle_discussion_df.csv', index=False)