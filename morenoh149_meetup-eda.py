import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
groups = pd.read_csv('/kaggle/input/meetups-data-from-meetupcom/groups.csv')

members = pd.read_csv('/kaggle/input/meetups-data-from-meetupcom/members.csv', encoding='latin-1')
groups.loc[groups['city'] == 'New York']
ny_groups = groups.loc[groups['city'] == 'New York']
ny_member_groups = pd.merge(ny_groups, members, on='group_id')
ny_member_groups
# find unique by member_id and group_id