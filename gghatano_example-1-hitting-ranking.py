# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dat2010 = pd.read_csv("../input/all2010.csv")

dat2010.head()
dat2010 = dat2010[["BAT_ID", "EVENT_CD"]]

dat2010 = dat2010[dat2010["EVENT_CD"].isin([20,21,22,23])]

dat2010.head()
dat2010_hit = dat2010.groupby("BAT_ID")["EVENT_CD"].count().rename("HIT").reset_index()

dat2010_hit.head()
dat2010_hit.sort_values(by = "HIT", ascending = False).head(10)
## Ex2 Home run Ranking in 2019
dat2010_homerun = dat2010[dat2010["EVENT_CD"] == 23]

dat2010_homerun_number = dat2010_homerun.groupby("BAT_ID")["EVENT_CD"].count().rename("HOMERUN").reset_index()

dat2010_homerun_number.sort_values(by = "HOMERUN", ascending=False).head(10)