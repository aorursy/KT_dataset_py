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
# official way to get the data
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
print('Done!')
os.listdir("../input")
(market_train_df, news_train_df) = env.get_training_data()
market_train_df.dtypes
market_train_df.head()
news_train_df.head()
import seaborn as sns
sns.pairplot(market_train_df)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_style("white")
sns.distplot(market_train_df.volume)
sns.distplot(market_train_df.open)
sns.distplot(market_train_df.close)
market_train_df.describe()
sns.distplot(market_train_df.returnsClosePrevRaw1)
sns.distplot(market_train_df.returnsOpenPrevRaw1)
sns.distplot(market_train_df.universe)
sns.distplot(market_train_df.returnsOpenNextMktres10)
sns.lineplot(x="time", y="open",
             hue="universe",
             data=market_train_df)
sns.lineplot(x="time", y="returnsOpenNextMktres10",
             hue="universe",
             data=market_train_df)
days = env.get_prediction_days()
print(days)
dir(env)
filtered_ = market_train_df.loc[market_train_df['universe'] == 1.0 ]
filtered_.head()
filtered_.shape