import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import math
df = pd.read_csv("../input/liverpool-ion-switching/test.csv")
n_groups = 40

df["group"] = 0

for i in range(n_groups):

    ids = np.arange(i*50000, (i+1)*50000)

    df.loc[ids,"group"] = i
for i in range(n_groups):

    sub = df[df.group == i]

    signals = sub.signal.values

    imax, imin = math.floor(np.max(signals)), math.ceil(np.min(signals))

    signals = (signals - np.min(signals))/(np.max(signals) - np.min(signals) + 2)

#     signals = (signals - np.min(signals))/(np.max(signals) - np.min(signals))

    signals = signals*(imax-imin)

    df.loc[sub.index,"open_channels"] = np.array(signals,np.int)
sample_df = pd.read_csv("../input/liverpool-ion-switching/sample_submission.csv", dtype={'time':str})
sample_df.open_channels = np.array(df.open_channels, np.int)

sample_df.to_csv("submission.csv",index=False)
!head submission.csv