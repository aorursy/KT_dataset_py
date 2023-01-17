import numpy as np

import pandas as pd

import os

import datetime

import warnings

import seaborn as sns

from matplotlib import pyplot as plt

from matplotlib.dates import DateFormatter

from tqdm._tqdm_notebook import tqdm_notebook



tqdm_notebook.pandas()

pd.set_option('display.max_columns', 100)

sns.set_style("whitegrid")

warnings.filterwarnings('ignore')



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
PATH = '/kaggle/input/meta-kaggle'
users = pd.read_csv(f'{PATH}/Users.csv')

users_ach = pd.read_csv(f'{PATH}/UserAchievements.csv')
display(users.tail())

display(users_ach.head())

print(f'User Id Duplicated: {users.Id.duplicated().any()}')
def get_resample_data(df, col, offset):

    df[col] = pd.to_datetime(df[col])

    return df.set_index(col).resample(offset).count().reset_index()



def plot_transition_graph(x, y, label=''):

    figure_ = plt.figure(1, figsize=(15,7))

    axes = figure_.add_subplot(111)

    axes.plot(x, y, 'o-', label=label)

    axes.legend(fontsize=18)

    xaxis = axes.xaxis

    xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
users_rd = get_resample_data(users, 'RegisterDate', 'M').rename(columns={'Id':'Cnt'})[:-1]

plot_transition_graph(users_rd.RegisterDate, users_rd.Cnt)
sub = pd.read_csv(f'{PATH}/Submissions.csv')

teams = pd.read_csv(f'{PATH}/Teams.csv')

compe = pd.read_csv(f'{PATH}/Competitions.csv')

display(sub.tail())

display(teams.tail())

display(compe.tail())
# Competitions type

compe = compe[(compe.HostSegmentTitle == 'Featured') |

              (compe.HostSegmentTitle == 'Research') ]

# merge

compe = compe.rename(columns={'Id': 'CompetitionId'})

teams = pd.merge(teams, compe[['CompetitionId']], on='CompetitionId')

teams = teams.rename(columns={'Id': 'TeamId'})

sub = pd.merge(sub, teams[['TeamId']], on='TeamId')
# Success Submission

sub = sub[~sub.PublicScoreLeaderboardDisplay.isnull() & ~sub.IsAfterDeadline]



# merge

sub_userid = sub.drop_duplicates(subset='SubmittedUserId')[['SubmittedUserId']]

sub_users = pd.merge(users, sub_userid, left_on='Id', right_on='SubmittedUserId')

sub_users.head()
sub_users_rd = sub_users.set_index('RegisterDate').resample('M').count().reset_index()[:-1]

sub_users_rd['roll'] = sub_users_rd.rolling(12).mean().Id
figure_ = plt.figure(1, figsize=(15,7))

axes = figure_.add_subplot(111)

axes.plot(sub_users_rd.RegisterDate, sub_users_rd.Id, 'o-', label='')

axes.plot(sub_users_rd.RegisterDate, sub_users_rd.roll, '-', alpha=0.5, label='12-month rolling mean')

axes.legend(fontsize=18)

xaxis = axes.xaxis

xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
compe = pd.read_csv(f'{PATH}/Competitions.csv')

compe_tag = pd.read_csv(f'{PATH}/CompetitionTags.csv')

tag = pd.read_csv(f'{PATH}/Tags.csv')
display(compe.head())

display(compe_tag.head())

display(tag.head())
# Competitions type

compe = compe[(compe.HostSegmentTitle=='Featured') | 

              (compe.HostSegmentTitle=='Research') ]



compe_tag_id = pd.merge(compe_tag, compe, left_on='CompetitionId', right_on='Id')
# top10

conpe_tag_cnt = compe_tag_id.groupby('TagId')[['CompetitionId']].count().rename(columns={'CompetitionId': 'Cnt'})

tag_top10 = conpe_tag_cnt.sort_values('Cnt', ascending=False)[:10].reset_index()

pd.merge(tag_top10, tag[['Id','Name']], how='left', left_on='TagId', right_on='Id').drop('Id', axis=1)
tabular_compe = compe_tag_id[compe_tag_id.TagId == 14101]

image_compe   = compe_tag_id[compe_tag_id.TagId == 14102]

text_compe    = compe_tag_id[compe_tag_id.TagId == 14104]
start_data = pd.DataFrame({'EnabledDate':[pd.to_datetime('2013-06-1')],'cnt':[0]})

end_data = pd.DataFrame({'EnabledDate':[pd.to_datetime('2019-11-05')],'cnt':[0]})



for compe_data, label in zip([tabular_compe, image_compe, text_compe],

                             ['tabular','image','text']):

    compe_data = get_resample_data(compe_data, 'EnabledDate', 'M'

                                  ).rename(columns={'CompetitionId': 'cnt'})

    # Add Start Data

    compe_data = start_data.append(compe_data,sort=False)

    compe_data = compe_data.append(end_data,sort=False)

    compe_data = compe_data.resample('2Q', on='EnabledDate').sum().reset_index()[:-1]

    plot_transition_graph(compe_data.EnabledDate, compe_data.cnt, label)
compe['EnabledDate'] = pd.to_datetime(compe.EnabledDate)

compe_rd = compe.set_index('EnabledDate').resample('2Q').count().reset_index()[:-1]

plot_transition_graph(compe_rd.EnabledDate, compe_rd.Id, 'num of compe')