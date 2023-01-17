# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
competition_df = pd.read_csv("/kaggle/input/meta-kaggle/Competitions.csv")
competition_df.tail(3)
competition_df.columns
competition_df["HostSegmentTitle"].unique()
mask = competition_df["CanQualifyTiers"] == True

#mask = competition_df["NumPrizes"] > 0

competitions = competition_df[mask].copy()
competitions.head(3)
competitions["EnabledDate"] = pd.to_datetime(competitions["EnabledDate"])

competitions["year"] = competitions["EnabledDate"].dt.year

competitions["month"] = competitions["EnabledDate"].dt.month
mask = (competitions["TotalTeams"] > 2000) & (competitions["year"] >= 2016) & (competitions["year"] <= 2019)

popular_competitions = competitions[mask].copy()



use_col = ["Slug", "month"]

popular_competitions = popular_competitions[use_col]
ax = popular_competitions["month"].plot(kind="hist", bins=12)

ax.set_xlabel("Month")

ax.set_ylabel("Frequency")

ax.set_title("Histogram of popular competitions since 2016")
mask = (competitions["year"] >= 2016) & (competitions["year"] <= 2019)

all_competitions = competitions[mask].copy()
ax = all_competitions["month"].plot(kind="hist", bins=12)

ax.set_xlabel("Month")

ax.set_ylabel("Frequency")

ax.set_title("Histogram of all competitions since 2016")