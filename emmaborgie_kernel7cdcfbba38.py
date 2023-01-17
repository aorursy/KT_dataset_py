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
df= pd.read_csv("/kaggle/input/us-counties-covid-19-dataset/us-counties.csv")
df
sum_df=df.groupby(["state"]).sum()
sum_df.head
simple_df=sum_df.drop(columns=["fips"])
simple_df.head()
simple_df.plot(kind="bar",figsize=(20,8))
sort_df=simple_df.sort_values("cases",ascending=False)
sort_df.plot(kind ="bar", figsize=(20,8))
df_cal=df.loc[df["state"]=="California"]
df_cal
cal_sum=df_cal.groupby(["county"]).sum()
cal_sum
cal_sum1 =cal_sum.drop(columns=["fips"])
cal_sum1
cal_sum1.plot(kind="bar",figsize=(20,8))