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
history_df = pd.read_csv('../input/historical_transactions.csv')
train_df = pd.read_csv('../input/train.csv')
#I merge dataframes, with only have the same card_id
join_df = pd.merge(history_df, train_df, on=['card_id'], how='inner')
join_df.head()
#My objective is create new dataframe, with column : id_card (uniquie), feature_1, feature_2, feature_3, target
card_target_df = join_df.groupby(['card_id','feature_1','feature_2','feature_3'], as_index=False)['target'].mean()
card_target_df.target.agg(['min','max'])
card_target_df.head()