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
!mkdir /kaggle/temp
!unzip ../input/job-recommendation/jobs.zip
!mv  /kaggle/working/jobs.tsv /kaggle/temp/
jobs_df = pd.read_csv('/kaggle/temp/jobs.tsv', sep='\t', error_bad_lines=False, nrows=99999)
jobs_df.head()
users_df = pd.read_csv('../input/job-recommendation/users.tsv', sep='\t', error_bad_lines=False)
users_df.head()
usersHistory_df = pd.read_csv('../input/job-recommendation/user_history.tsv', sep='\t')
usersHistory_df.head()
jobs_df.at[2, 'Requirements']
removePattern = r'(<(.*?)>)|(&\w+)'
addSpacePattern = r'([;:])|(\\r)|(\\n)'
removeExtraSpaces = r'(\s\s+?)(?=\S)'
jobs_df['DescCleaned'] = jobs_df['Description'].str.lower()
jobs_df['DescCleaned'] = jobs_df['DescCleaned'].str.replace(removePattern, "")
jobs_df['DescCleaned'] = jobs_df['DescCleaned'].str.replace(addSpacePattern, " ")
jobs_df['DescCleaned'] = jobs_df['DescCleaned'].str.replace(removeExtraSpaces, " ")
# Same for Requirements
jobs_df['ReqCleaned'] = jobs_df['Requirements'].str.lower()
jobs_df['ReqCleaned'] = jobs_df['ReqCleaned'].str.replace(removePattern, "")
jobs_df['ReqCleaned'] = jobs_df['ReqCleaned'].str.replace(addSpacePattern, " ")
jobs_df['ReqCleaned'] = jobs_df['ReqCleaned'].str.replace(removeExtraSpaces, " ")
jobs_df.at[0, 'DescCleaned']
userHistory_df = pd.read_csv('../input/job-recommendation/user_history.tsv', sep='\t')
userHistory_df.loc[userHistory_df.UserID == 72]
CS = users_df.loc[(users_df.Major == 'Computer Science')]
CS['DegreeType'].value_counts()
CS.loc[(CS.ManagedOthers == 'Yes') & (CS.DegreeType == 'Associate\'s')]
users_df['Major'].value_counts().head(50)
CS = users_df.loc[(users_df.Major == 'Computer Science')|
                  (users_df.Major == 'Information Technology')|
                  (users_df.Major == 'Computer Information Systems')
                 ]
CS['DegreeType'].value_counts()
CS.head()
import seaborn as sns
x = CS['WorkHistoryCount'].loc[CS.WorkHistoryCount < 17]
sns.distplot(x)
jobs_df.loc[(jobs_df.Title.str.contains('Developer'))&
            (jobs_df.State == 'CA')&(
            (jobs_df.DescCleaned.str.contains('python')) |
            (jobs_df.ReqCleaned.str.contains('python'))
            )
           ]
job = jobs_df.loc[13882,['Title', 'City', 'State', 'DescCleaned', 'ReqCleaned']]
job.DescCleaned
job.ReqCleaned
CS.loc[(CS.City == 'San Jose') &
       ((CS.DegreeType == "Bachelor's") |
        (CS.DegreeType == "Master's")) &
       (CS.WorkHistoryCount < 5) &
       (CS.WorkHistoryCount > 1)
      ]
usersHistory_df.loc[usersHistory_df.UserID == 929214]
usersHistory_df.loc[usersHistory_df.UserID == 1216512]
