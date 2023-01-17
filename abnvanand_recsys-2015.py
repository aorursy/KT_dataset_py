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
# Load Datasets

clicks_df=pd.read_csv('../input/yoochoose-data/yoochoose-clicks.dat',

                      names=['session_id','timestamp','item_id','category'],

                      dtype={'category': str})

display("Clicks Data",)

display(clicks_df.head())



buys_df = pd.read_csv('../input/yoochoose-data/yoochoose-buys.dat', names=['session_id', 'timestamp', 'item_id', 'price', 'quantity'])

display("Buys Data",)

display(buys_df.head())
# Explore data



# display(clicks_df.describe())

# display(buys_df.describe())

## The results indicate that buys data session_id and item_id are a subset of clicks data





# (clicks_df.groupby(['session_id', 'item_id']).count()).head(10)

## There can be multiple clicks on the same item in a particular session



(buys_df.groupby(['session_id', 'item_id']).count()).head(10)

## There can be multiple buys of the same item in a particular session.

## Notice this is different from quantity bought.

buys_df[buys_df['session_id']==11]



# Merge clicks and buys data by session and item ids

# merge by left join bcoz clicks_df's session and item ids are a superset of those of buys_df's

df = pd.merge(clicks_df, buys_df, on=['session_id','item_id'], how='left' , suffixes=('_click','_buy'))

print(len(df), len(clicks_df) + len(buys_df))

df.head()
display(clicks_df[clicks_df.session_id==420374])



display(buys_df[buys_df.session_id==420374])



display(df[df.session_id==420374])