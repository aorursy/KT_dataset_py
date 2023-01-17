# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

############### My First Attempt of Kaggle Kernel ###################
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) 

# import the module and create an environment
from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()

############### run code with Ctrl + Enter/ Shift + Enter is within window, not Commit  ##########################
# Any results you write to the current directory are saved as output.

## Returns the training data DataFrames as a tuple of
(market_train_df, news_train_df) = env.get_training_data()
market_train_df.head()
market_train_df.tail()
news_train_df.head()
news_train_df.tail()

