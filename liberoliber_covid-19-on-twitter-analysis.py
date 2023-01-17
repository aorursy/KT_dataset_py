!pip install twint



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd

import datetime



import twint

import sys



# Set up TWINT config

c = twint.Config()

from collections import Counter



import nest_asyncio

nest_asyncio.apply()



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import datetime
# Configure

list_ = ["covid19", "covid-19", "covid"]
count=0

base = "2020-01-27 17:00:00"

base1 = datetime.datetime.strptime(base,'%Y-%m-%d %H:%M:%S')

date_list = [base1 - datetime.timedelta(days=x) for x in range(360)]
for i in list_:

    print(i)

    c = twint.Config()

    c.Search = i

    c.Pandas = True

    c.Since = "2020-01-27"

    c.Until = "2020-05-28"

    #c.Location = True

    #c.Limit = 1000

    #c.Near = "Houston "

    c.Custom_csv = ["id", "user_id", "username", "date", "tweet"]

    c.Output = f"hh_{count}.csv"
c.Custom_csv = ["id", "user_id", "username", "date", "tweet"]

c.Output = f"hh_{count}.csv"
c.Output = f"hh_{count}.csv"





twint.run.Search(c);



Tweets_df = twint.storage.panda.Tweets_df

count+=1

Tweets_df.to_csv(f"hh_{count}.csv")
filenames = [f"hh_{i}.csv" for i in range(0,23)]

df = pd.concat( [pd.read_csv(f"{f}") for f in filenames] )
df