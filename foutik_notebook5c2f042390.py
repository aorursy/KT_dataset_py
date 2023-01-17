# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
sessions = pd.read_csv('../input/sessions.csv')
users = pd.read_csv('../input/train_users_2.csv')
device_type_time = pd.pivot_table(sessions, index=['user_id'], columns=['device_type'], values='secs_elapsed', aggfunc=sum, fill_value=0)

device_type_time
device_type_time.reset_index(inplace=True)

device_type_time
device_type_time['total_elapsed_time'] = device_type_time.sum(axis=1)
device_type_time
device_columns = device_type_time.columns[1:-2]
for column in device_columns:

        device_type_time[column + '_pct'] = device_type_time.apply(lambda row: row[column]/row['total_elapsed_time'] if row['total_elapsed_time'] > 0 else 0, axis = 1)
device_type_time
device_type_time.rename(columns={'id' : 'user_id'})

users.rename(columns={'id':'user_id'})
users_combined_df = pd.merge(users, device_type_time, left_on='id', right_on='user_id', how = 'left', left_index = True)
users_combined_df.to_csv('combined.csv', index = False)
users.loc[(users['age'] > 100) | (users['age'] < 14), 'age'] = -1
users['age'].fillna(-1, inplace=True)
bins = [16, 20, 25, 30, 40, 50, 60, 75, 100]

users['age_group'] = np.digitize(users['age'], bins, right=True)
users['age_group'] = users['age_group'].replace(0,-1)

users['age_group']
users['age']