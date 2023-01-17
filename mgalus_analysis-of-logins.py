# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        if filename == 'mlog_passwords.log':

            print('Lets go!')

            file_path = os.path.join(dirname, filename)

            df = pd.read_csv(file_path, usecols=[0,1,2,3], names=['ip', 'time', 'user', 'pass'])

            print('\nSummary:')

            print(df.describe())

            #df['ip'].plot(kind='hist'); 

            ax = sns.countplot(x="ip",data=df)

            print('unique ips: ', df.ip.nunique())

            print('ips: ', df.ip.unique())



# Any results you write to the current directory are saved as output.
            #bx = sns.countplot(x="time",data=df)

            print('unique times lol: ', df.time.nunique())

            print('all times logged in: ', df.time.count())

            print('Number of logins per minute AKA activity: ', df.time.count()/df.time.nunique())
            cx = sns.countplot(x="user",data=df)

            print('unique users: ', df.user.nunique())

            print('users: ', df.user.unique())
            dx = sns.countplot(x="pass",data=df)

            print('unique passwords: ', df['pass'].nunique())

            print('passwords: ', df['pass'].unique())

            print('Done')