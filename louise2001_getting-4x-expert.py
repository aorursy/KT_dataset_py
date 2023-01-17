import numpy as np

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if 'User' in filename or 'Data' in filename:

            print(os.path.join(dirname, filename))
achievements = pd.read_csv('/kaggle/input/meta-kaggle/UserAchievements.csv')

achievements.head()
np.unique(achievements.AchievementType), sorted(np.unique(achievements.Tier))
from copy import deepcopy as dc

def get_users_all_level(tier=0):

    ach = dc(achievements)

    if tier < 5: #in this case we remove kaggle team

        ach = ach.loc[ach.Tier < 5]

    d = ach.loc[(ach.Tier>=tier) & (ach.AchievementType=='Discussion'), 'UserId'].values.tolist()

    c = ach.loc[(ach.Tier>=tier) & (ach.AchievementType=='Competitions'), 'UserId'].values.tolist()

    s = ach.loc[(ach.Tier>=tier) & (ach.AchievementType=='Scripts'), 'UserId'].values.tolist()

    return [user for user in c if user in d and user in s]
print(f'We have {len(get_users_all_level(tier=2))} experts in 3, {len(get_users_all_level(tier=3))} masters in 3 and only {len(get_users_all_level(tier=4))} grandmasters in 3 !')

interest_users = get_users_all_level(tier=2)
users = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv')

users.columns = ['UserId', 'UserName', 'DisplayName', 'RegisterDate', 'PerformanceTier']

users = users.loc[users.PerformanceTier>0]

users.sample(5)
users.loc[users.UserId.isin(interest_users)].sample(10)
users.loc[users.UserName=='louise2001']
my_userid = users.loc[users.UserName=='louise2001', 'UserId'].values[0]

my_userid in interest_users
total = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv').shape[0]

novices = total - users.shape[0]

contributors = users.loc[users.PerformanceTier==1].shape[0]

experts = users.loc[users.PerformanceTier==2].shape[0]

masters = users.loc[users.PerformanceTier==3].shape[0]

grandmasters  = users.loc[users.PerformanceTier==4].shape[0]

kaggle_team = users.loc[users.PerformanceTier==5].shape[0]

print(f'Out of a total of {total} registered Kaggle users, we have :')

print(f'{novices} novices ({round(novices/total*100,3)}%)')

print(f'{contributors} contributors ({round(contributors/total*100,3)}%)')

print(f'{experts} experts ({round(experts/total*100,3)}%)')

print(f'{masters} masters ({round(masters/total*100,3)}%)')

print(f'{grandmasters} grandmasters ({round(grandmasters/total*100,3)}%)')

print(f'{kaggle_team} Kaggle Team ({round(kaggle_team/total*100,3)}%)')
data_votes = pd.read_csv('/kaggle/input/meta-kaggle/DatasetVotes.csv')

data_votes = data_votes.loc[data_votes.UserId.isin(users.UserId.values)]

data_votes.sample(5)
datasets = pd.read_csv('/kaggle/input/meta-kaggle/Datasets.csv')

datasets.columns = ['DatasetId', 'CreatorUserId', 'OwnerUserId', 'OwnerOrganizationId',

       'CurrentDatasetVersionId', 'CurrentDatasourceVersionId', 'ForumId',

       'Type', 'CreationDate', 'ReviewDate', 'FeatureDate', 'LastActivityDate',

       'TotalViews', 'TotalDownloads', 'TotalVotes', 'TotalKernels']

datasets = datasets.loc[(datasets['TotalVotes']>=5) & (datasets['CreatorUserId'].isin(interest_users)), ['DatasetId', 'CreatorUserId',

       'CurrentDatasetVersionId', 'CurrentDatasourceVersionId', 'CreationDate', 'TotalVotes']]

datasets.sample(5)
interest_users[:] = [user for user in interest_users if datasets.loc[datasets.CreatorUserId==user].shape[0]>=3]

len(interest_users)
datasets = datasets.loc[datasets.CreatorUserId.isin(interest_users)]

datasets
datasets = datasets.merge(data_votes, left_on='CurrentDatasetVersionId', right_on = 'DatasetVersionId')

datasets
datasets = datasets.loc[datasets.CreatorUserId != datasets.UserId]

datasets
size = datasets.groupby(['DatasetId']).size()

size = size[size>=5]

datasets = datasets.loc[datasets['DatasetId'].isin(size.index)]

dataset_experts = np.unique(datasets.CreatorUserId.values)
interest_users[:] = [user for user in interest_users if user in dataset_experts]

len(interest_users)
my_userid in interest_users