import numpy as np
import pandas as pd
import os
import matplotlib.pylab as plt
user = pd.read_csv('../input/Users.csv')
KernelVotes = pd.read_csv('../input/KernelVotes.csv')
Kernels = pd.read_csv('../input/Kernels.csv')
# Our guy
user.loc[user['Id'] == 949803]
his_kernels = Kernels.loc[Kernels['AuthorUserId'] == 949803]
len(his_kernels)
his_kernels.set_index('CurrentUrlSlug')['TotalVotes'].sort_values().plot(kind='barh',
                                                                         figsize=(15, 18),
                                                                         title='Kernels and Votes',
                                                                         color='grey')
plt.show()
his_kernels_list = his_kernels['CurrentKernelVersionId'].tolist()
his_votes = KernelVotes.loc[KernelVotes['KernelVersionId'].isin(his_kernels_list)].copy()
his_votes['count'] = 1
his_votes.groupby('UserId').count()[['count']].sort_values('count',
                                                           ascending=False) \
    .plot(kind='bar', figsize=(15, 5), title='Number of Votes on His Kernel by User')
plt.show()
his_vote_user_count = his_votes.groupby('UserId').count()[['count']].sort_values('count', ascending=False)
his_vote_user_count_more_than_5 = his_vote_user_count.loc[his_vote_user_count['count'] >= 5]
kernel_votes_with_kernel_details = pd.merge(KernelVotes, Kernels, how='left')
kernel_votes_with_kernel_details = pd.merge(kernel_votes_with_kernel_details, user, left_on='AuthorUserId', right_on='Id', how='left')
kernels_voted_on_by_friends = kernel_votes_with_kernel_details.loc[kernel_votes_with_kernel_details['UserId']
                                                                   .isin(his_vote_user_count_more_than_5.index.tolist())]
ax = KernelVotes.loc[KernelVotes['UserId'].isin(his_vote_user_count_more_than_5.index.tolist())] \
    .groupby('KernelVersionId') \
    .count() \
    .sort_values('Id')['Id'].plot(kind='bar', figsize=(15, 5), title='Each bar is a kernel voted on my one of these friends')
x_axis = ax.axes.get_xaxis()
x_axis.set_visible(False)
plt.show()
# What kernels do the friends vote on?
KernelVotes['count'] = 1
kernels_friends_voted_on = KernelVotes.loc[KernelVotes['UserId'].isin(his_vote_user_count_more_than_5.index.tolist())] \
    .groupby('KernelVersionId') \
    .count()[['count']]
kernels_friends_voted_on_more_than_once = kernels_friends_voted_on.loc[kernels_friends_voted_on['count'] > 1]
kernels_with_author_info = pd.merge(Kernels, user, left_on='AuthorUserId', right_on='Id')

kernels_with_author_info.loc[kernels_with_author_info['CurrentKernelVersionId'].isin(kernels_friends_voted_on_more_than_once.index.tolist())] \
    .groupby('UserName') \
    .count()[['Id_x']] \
    .sort_values('Id_x').plot(kind='barh', title='Count of Kernels Friends voted on more than once by Author',
                              figsize=(15, 5),
                              legend=False)
plt.show()
his_friends = his_vote_user_count_more_than_5.reset_index()['UserId'].tolist()
user.loc[user['Id'].isin(his_friends)]
following = pd.read_csv('../input/UserFollowers.csv')
following['count'] = 1
following_counts = following.loc[following['UserId'].isin(his_friends)] \
    .groupby('FollowingUserId') \
    .count() \
    .sort_values('count')[['count']].reset_index()
# All The Users these 'Friends are following'
pd.merge(following_counts.loc[following_counts['count'] > 1],
         user, left_on='FollowingUserId', right_on='Id', how='left')
pd.merge(following_counts.loc[following_counts['count'] > 1],
         user, left_on='FollowingUserId', right_on='Id', how='left') \
    .plot(x='UserName', y='count', kind='barh', title='Users who These Profiles follow (min 1)',
          legend=False, figsize=(15, 5))
plt.show()
