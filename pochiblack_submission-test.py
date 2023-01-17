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
df = pd.read_csv("../input/cities.csv")
df.head()
sample_df = pd.read_csv('../input/sample_submission.csv')
sample_df.head()
for city_id in df["CityId"]:
    print(city_id)
print(df["CityId"].max())
print(df["CityId"].min())
submission_ids = []
for i in range(df["CityId"].min(), df["CityId"].max()+1):
    submission_ids.append(i)
submission_ids.append(0)
submission_df = pd.DataFrame(submission_ids)
submission_df.columns = ["Path"]
submission_df.head()
submission_df.to_csv("submission.csv", index=None)
