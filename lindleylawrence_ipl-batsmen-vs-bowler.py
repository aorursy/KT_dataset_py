# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sys



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/ipldata/deliveries.csv")

df
df.info()
df.columns
df_no_wides = df.loc[(df["wide_runs"] == 0)]

df_no_wides
# Create batsmen and bowler lists

batsmen_list = df["batsman"].unique().tolist()

bowlers_list = df["bowler"].unique().tolist()

batsmen_list[0:10]
bowlers_list[0:10]


## This will be a long list as you can imagine

# for bat in batsmen_list:

#     for bowler in bowlers_list:

#         runs = df_no_wides.loc[(df_no_wides["batsman"] == bat) & (df_no_wides["bowler"] == bowler)]

#         balls_faced = runs["batsman_runs"].count()

#         runs_scored = runs["batsman_runs"].sum()

#         strike_rate = round(runs_scored / balls_faced *100,2)

#         print(bat,bowler,runs_scored,balls_faced,strike_rate)





        