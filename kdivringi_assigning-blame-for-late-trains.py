import pandas as pd

import numpy as np

import seaborn as sns

%matplotlib inline

sns.plt.rcParams['figure.figsize'] = (12, 10)
from sqlalchemy import create_engine

con = create_engine('sqlite:///../input/database.sqlite')
df = pd.read_sql_table('otp', con)

df.head()
df.info()
df.loc[df.status=="1440 min", "status"] = "999 min"

df['status_n'] = df.status.str.replace("On Time", "0").str.replace(" min","").astype("int")
t = df[df.train_id=="550"].sort_values(by='timeStamp')

t.head()
df[df.train_id=="550"].sort_values(by='timeStamp').iloc[:100].plot(x='date', y='status_n')
t['status_diff']= t.status_n.diff()

t.head()
t.plot(x='date', y='status_diff')
t.status_diff.hist(bins=50, log=True)
tg = t.groupby(['next_station']).mean().sort_values(['status_diff'])

tg
tg.plot(kind="scatter", x='status_n', y='status_diff')
tg.corr()
df.sort_values(by=['train_id', 'timeStamp'], inplace=True)
df['status_diff'] = df.status_n.diff()
df.loc[df.next_station == "None",'status_diff'] = np.NaN

df.head()
diffs = df.dropna().groupby(['next_station']).mean().sort_values(['status_diff'])

diffs
diffs.plot(kind='scatter', x='status_n', y='status_diff')
diffs.corr()