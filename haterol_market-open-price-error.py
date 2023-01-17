# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

(market_train_df, news_train_df) = env.get_training_data()
market_train_df[abs(market_train_df['returnsOpenPrevMktres10'])>=3]
market_train_df[abs(market_train_df['returnsClosePrevMktres10'])>=3]
market_train_df[abs(market_train_df['close']/market_train_df['open'])<=0.5]
market_train_df[market_train_df['assetCode'] == "EXH.N"][['time', 'open','returnsOpenPrevRaw1','returnsOpenPrevRaw10','returnsOpenPrevMktres1', 'returnsOpenPrevMktres10', 'returnsOpenNextMktres10',
                                                         'close','returnsClosePrevRaw1','returnsClosePrevRaw10','returnsClosePrevMktres1', 'returnsClosePrevMktres10']].head(50)