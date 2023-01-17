# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime as dt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import training data as a DataFrame: df



train_df = pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv", parse_dates=["Date"])



# Show first few entries



train_df.head()
# Filter by country to UK



uk_train_df = train_df[train_df["Country/Region"] == "United Kingdom"]



# Show first few UK entries



uk_train_df.head()
# Find unique Province/State values



uk_train_df["Province/State"].unique()
# Filter to United Kingdom Province



uk_train_df = uk_train_df[ uk_train_df["Province/State"] == "United Kingdom" ]



# Show first few entries



uk_train_df.head()
# Set index to Date column

# Make sure to only run this once



uk_train_df.set_index("Date", inplace=True)



# Get rid of unnecessary columns



uk_train_df = uk_train_df[ ["ConfirmedCases", "Fatalities"] ]



uk_train_df.head()
# Get dates later than 10th March



recent = uk_train_df[dt.datetime(2020,3,10):]



recent
# Plot recent data



recent.plot()