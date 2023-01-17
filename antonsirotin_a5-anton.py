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
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt

#loading the csv file into a var
nfl = pd.read_csv("../input/nfl-passing-stats/NFL Passing Stats - 2019.csv")
nfl

#defining a function that decides whether a QB is a scrub, 'just a guy,' or a franchise QB based on QB rating
def qbQual(rating):
    if rating < 80:
        return 'Scrub'
    elif rating >= 70 and rating < 100:
        return 'JAG'
    else:
        return 'Franchise Quarterback'
    
#applies the qbQual function through the values in the Rate collomn, then puts them into categories, then plots a simple pie graph
nfl['Rate'].apply(qbQual).value_counts().plot(kind='pie')
#extracting only relevant columns from the data and plotting them on a scatter plot
nfl[['Pass Yds', 'Rate']].plot(kind='scatter', x='Pass Yds', y='Rate', alpha=1)
#defining a function that decides whether a qb should be cut, given one more chance to prove himself, or he's a keeper - based on interceptions
def qbINT(INT):
    if INT > 18:
        return 'Cut Him Now'
    elif INT <= 17 and INT > 10:
        return 'Give him a prove-it year'
    else:
        return 'Keep him!'
    
#applies the qbINT function through the values in the INT collumn, then puts them into categories, then plots a simple bar graph
nfl['INT'].apply(qbINT).value_counts().plot(kind='bar')