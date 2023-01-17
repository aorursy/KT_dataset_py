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
usa_2016_presidential_election_by_county = pd.read_csv('/kaggle/input/us-elections-dataset/usa-2016-presidential-election-by-county.csv', sep=';')
usa_2016_presidential_election_by_county.head()
usa_2016_presidential_election_by_county.shape
usa_2016_presidential_election_by_county.describe()
primary_results = pd.read_csv('/kaggle/input/us-elections-dataset/us-2016-primary-results.csv', sep=';')
us_2016_primary_results.head()
us_2016_primary_results.shape
county_votes = primary_results.loc[primary_results["county"] == "Cabarrus"]

county_votes

import matplotlib.pyplot as plt

county_votes = county_votes[["candidate","votes"]]

candidate_column = county_votes.loc[:,'candidate']

candidate = candidate_column.values

vote_column = county_votes.loc[:,'votes']

votes = vote_column.values



county_votes.plot(x= "candidate" , y= "votes", kind="bar")

county_votes.plot(x= "candidate" , y= "votes", kind="line")

county_votes.plot(x= "candidate" , y= "votes", kind="pie")
