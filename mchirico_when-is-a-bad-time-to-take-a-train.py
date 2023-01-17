# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sqlalchemy import create_engine

import seaborn as sns

sns.plt.rcParams['figure.figsize'] = (12, 10)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
con = create_engine('sqlite:///../input/database.sqlite')
df = pd.read_sql_table('otp', con)

df.head()
df.describe()
df.status.unique()
df[df.status=="1440 min"]
df.loc[df.status=="1440 min", "status"] = "999 min"

df['status_n'] = df.status.str.replace("On Time", "0").str.replace(" min","").astype("int")

df.head()
df[df.status_n!=999].status_n.hist(bins=100, log=True);
print("Number of suspended trains:", len(df[df.status_n==999]))
# On time trains:

ot = df[df.status_n < 6]

# Late trains:

lt = df[df.status_n >= 6]

print("On time trains:", len(ot), "Late trains:", len(lt), "Percentage on time:", len(ot)/len(df)*100)
df['Day'] = df.timeStamp.dt.dayofweek

df['Hour'] = df.timeStamp.dt.hour

gb = df[df.status_n!=999].groupby(["Hour", "Day"]).aggregate(np.sum).unstack()

gb.head()
sns.heatmap(gb,xticklabels=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]);
lt = df[df.status_n >= 6]

gb2 = lt[df.status_n!=999].groupby(["Hour", "Day"]).aggregate(np.sum).unstack()

gb2.head()
sns.heatmap(gb2,xticklabels=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]);