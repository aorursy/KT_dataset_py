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

df.head()

meta = df.pop("Category").to_frame()

df.head()

meta.head()
df = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')

df.head()
d = {"A":[1, 2, 3, 4,], "B":[1.0, 2.0, 3.0, 4.0], "C":[1.00000, 2.00000, 3.00000, 4.000003], "D":[1.0, 2.0, 3.0, 4.0], "E":[4.0, 2.0, 3.0, 1.0]}

df = pd.DataFrame(d)

df



df["A"].equals(df["B"]) # they requiere identical datatypes
df["B"].equals(df["C"])
df["B"].equals(df["D"])
print(pd.testing.assert_series_equal(df["A"], df["B"], check_names=False, check_dtype=False))
df = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')

df
# Fixing columns

df["App"] = df["App"].str.title()

df["Category"] = df["Category"].str.title()



df["count_by_App"] = df.groupby("App").cumcount() + 1

df["count_by_App"]
df["count_by_Category"] = df.groupby("Category").cumcount() + 1

df["count_by_Category"] 
df["count_by_both"] = df.groupby(["App","Category"]).cumcount() + 1

df["count_by_both"]

df
df = pd.read_csv('/kaggle/input/ipl-data-set/matches.csv')

df
df["running_total"] = df["win_by_runs"].cumsum()

df["running_total_by_person"] = df.groupby("win_by_runs")["win_by_wickets"].cumsum()

df
df = pd.Series(["Geordi La Forge", "Deanna Troi", "Data"]).to_frame()

df.rename({0:"names"}, inplace = True, axis = 1)

df

#split on first space  

df["first_name"] = df["names"].str.split(n = 1).str[0]

df["last_name"] = df["names"].str.split(n = 1).str[1]

df