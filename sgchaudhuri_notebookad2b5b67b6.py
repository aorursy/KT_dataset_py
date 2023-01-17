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
from sklearn.cross_validation import train_test_split

from stop_words import get_stop_words

import matplotlib.pyplot as plt

from __future__ import division

from collections import Counter

import pandas as pd

import numpy as np

import string

import re





tweets_data_path = '../input/Tweets.csv'

tweets = pd.read_csv(tweets_data_path, header=0)

df = tweets.copy()[['tweet_id', 'airline','text' , 'tweet_created','user_timezone','airline_sentiment']]
pd.Series(tweets["negativereason"]).value_counts().plot(kind = "barh",

                            figsize=(10,6),fontsize=12,rot = 0, title = "Negative Reason")
pd.Series(tweets["user_timezone"]).value_counts().head(10).plot(kind = "barh",

                                     figsize=(8,6),title = "User_Timezone")
