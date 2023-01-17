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
df = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')

df
print("Reverse column order")

df.loc[:, ::-1]
print("Reverse row order")

df.loc[::-1]

print("Reverse row order and reset index")

df.loc[::-1].reset_index(drop = True)
print("Add prefix")

df.add_prefix("A_")

print("Add suffix")

df.add_suffix("_Z")
df
df["App"] == df["Category"]

df["App"].equals(df["Category"])





df1 = df.copy(deep = True)

df.equals(df1)





df == df1
df2 = pd.read_csv('/kaggle/input/ipl-data-set/matches.csv')

df2
df2.info(memory_usage = "deep")

df.memory_usage(deep = True)
import pandas_profiling

df2.profile_report()