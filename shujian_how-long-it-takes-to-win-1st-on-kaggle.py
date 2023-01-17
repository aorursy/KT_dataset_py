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
%matplotlib inline  



import matplotlib.pyplot as plt

import numpy as np

import plotly.plotly as py

import pandas as pd

import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

sns.set(style="ticks")
t = pd.read_csv('../input/Teams.csv')

# t.head()
tm =  pd.read_csv('../input/TeamMemberships.csv')

# tm.head()
usr =  pd.read_csv('../input/Users.csv')

# usr.head()
cp =  pd.read_csv('../input/Competitions.csv')

# cp.head()
from datetime import datetime



d = dict()



t1 = t[t.Ranking == 1]



for index, row in t1.iterrows():

    cpRow = cp[cp.Id == row.CompetitionId].iloc[0]

    if cpRow.RewardTypeId in [1, 5]:  # Only Reward Type of USD or Jobs

        deadline = cpRow.Deadline 

        if row.TeamLeaderId in d:

            if deadline < d[row.TeamLeaderId]:

                d[row.TeamLeaderId] = deadline

        else:

            d[row.TeamLeaderId] = deadline



df = pd.DataFrame(columns=['Id', 'RegisterDate', 'Duration'])

for k in d:

    reg = datetime.strptime(usr[usr.Id == k].iloc[0].RegisterDate, "%Y-%m-%d %H:%M:%S")

    duration = (datetime.strptime(d[k], "%Y-%m-%d %H:%M:%S") - reg).days

    df = df.append(pd.Series([k, usr[usr.Id == k].iloc[0].RegisterDate, duration], index=['Id', 'RegisterDate', 'Duration']), ignore_index=True)



    

df.describe()
# df

df.head()
plt.hist(df.Duration)

plt.title("Days Needed to Win 1st")

plt.xlabel("Days Needed")

plt.ylabel("Frequency")



fig = plt.gcf()
df["RegisterDate"] = df["RegisterDate"].astype('datetime64[ns]')



ax = df.plot(x = "RegisterDate", y = "Duration", style=['o'], title = "Days Needed to Win 1st")
d = dict()



t1 = t[t.Ranking == 1]



for index, row in t1.iterrows():

    cpRow = cp[cp.Id == row.CompetitionId].iloc[0]

    if cpRow.RewardTypeId in [1, 5]:  # Only Reward Type of USD or Jobs

        deadline = cpRow.Deadline 

        

        all_members = tm[tm.TeamId == row.Id]

        

        for _, row_member in all_members.iterrows():

            if row_member.UserId in d:

                if deadline < d[row_member.UserId]:

                    d[row_member.UserId] = deadline

            else:

                d[row_member.UserId] = deadline



        



df = pd.DataFrame(columns=['Id', 'RegisterDate', 'Duration'])

for k in d:

    try:

        reg = datetime.strptime(usr[usr.Id == k].iloc[0].RegisterDate, "%Y-%m-%d %H:%M:%S")

        duration = (datetime.strptime(d[k], "%Y-%m-%d %H:%M:%S") - reg).days

        df = df.append(pd.Series([k, usr[usr.Id == k].iloc[0].RegisterDate, duration], index=['Id', 'RegisterDate', 'Duration']), ignore_index=True)

    except:

        pass

    

df.describe()
# df

df.head()
plt.hist(df.Duration)

plt.title("Days Needed to Win 1st")

plt.xlabel("Days Needed")

plt.ylabel("Frequency")



fig = plt.gcf()
df["RegisterDate"] = df["RegisterDate"].astype('datetime64[ns]')



ax = df.plot(x = "RegisterDate", y = "Duration", style=['o'], title = "Days Needed to Win 1st")
np.mean(df[df.Duration > 0].Duration)