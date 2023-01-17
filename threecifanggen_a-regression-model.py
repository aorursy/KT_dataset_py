# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bokeh
import nltk

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
open('../input/hashes.txt').read()
import  sqlite3
conn = sqlite3.connect('../input/database.sqlite')
data_all = pd.read_sql('select * from tweets;', conn)
data_all.head()
numeric_columns = ["airline_sentiment_confidence", "negativereason_confidence", "retweet_count"]
factor_columns = ["tweet_location", "name", "negativereason", "airline_sentiment", "airline"]
data_all.loc[:, numeric_columns].describe(include='all')
data_all.loc[:, factor_columns].describe(include='all')
