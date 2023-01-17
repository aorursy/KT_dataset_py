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
ROOT_DIR = "/kaggle/input/covid19-global-forecasting-week-1/"
train_df = pd.read_csv(ROOT_DIR+"train.csv")

test_df = pd.read_csv(ROOT_DIR+"test.csv")

sample_sub_df = pd.read_csv(ROOT_DIR+"submission.csv")
train_df.head()
train_df.describe()
print(f"Train:\nStart date: {train_df.Date.min()}\tMax date: {train_df.Date.max()}")

print(f"Test:\nStart date: {test_df.Date.min()}\tMax date: {test_df.Date.max()}")
valid_df = train_df[train_df.Date >= test_df.Date.min()]

train_new = train_df[train_df.Date < test_df.Date.min()]
train_countries = train_df['Country/Region'].unique()

test_countries = test_df['Country/Region'].unique()
grouped = train_df.groupby(["Country/Region", "Date"])
grouped.first()
start_date = {}

for name, group in grouped:

    start_ = group[group["ConfirmedCases"]==1]

    start_date[start_["Country/Region"]] = start_.Date.min()

    print(start_date)
from datetime import datetime
first_date = {}

for tr_c in train_countries:

    #print(tr_c)

    train_ = train_df[train_df["Country/Region"]==tr_c]

    st_date = train_[train_["ConfirmedCases"]>=1].Date.min()

    print(tr_c, st_date, st_date=="nan")

    print(type(st_date))

    if st_date != "nan" and type(st_date) != float:

        print(datetime.strptime(st_date, '%Y-%m-%d').date()-datetime.strptime(st_date, '%Y-%m-%d').date())

        train_["diff"] = train_["Date"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').date()-datetime.strptime(st_date, '%Y-%m-%d').date())

    first_date[tr_c] = st_date

    
first_date