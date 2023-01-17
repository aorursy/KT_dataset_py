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
df = pd.read_csv("../input/GlobalLandTemperaturesByCity.csv")

df.head()
us_df = df[df.Country=="United States"]

us_df.date = pd.to_datetime(us_df.dt)

us_df["year"] = us_df.date.apply(lambda x: x.year)

us_df["month"] = us_df.date.apply(lambda x: x.month)

us_df["day"] = us_df.date.apply(lambda x: x.day)

us_df.head()
us_df.City.value_counts()