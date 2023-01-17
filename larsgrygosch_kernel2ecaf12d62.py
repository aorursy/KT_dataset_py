import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

with open("/kaggle/input/sleepstudypilot/README.md", 'r') as f:
    for line in f.readlines():
        print(line)
df = pd.read_csv("/kaggle/input/sleepstudypilot/SleepStudyData.csv")
df.head()
df.tail()
df.info()
df.describe()
#df["Hours"]
df.Hours
df.iloc[2][4]
df.loc[2]["Tired"]
df
df[df.Hours < 5]
df.info()
df_enough = df.groupby(["Enough"])
df_enough.describe()
