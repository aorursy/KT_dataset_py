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
# Read historical transactions data
hist_df1 = pd.read_csv("../input/hist_DecFeb18.csv")
hist_df2 = pd.read_csv("../input/hist_SepNov.csv")
hist_df3 = pd.read_csv("../input/hist_JunAug.csv")
hist_df4 = pd.read_csv("../input/hist_MarMay.csv")
hist_df5 = pd.read_csv("../input/hist_JanFeb.csv")
VarAmt_by_Card_Merch1 = hist_df1.groupby(['card_id'])['purchase_amount'].var().reset_index()
VarAmt_by_Card_Merch2 = hist_df2.groupby(['card_id'])['purchase_amount'].var().reset_index()
VarAmt_by_Card_Merch3 = hist_df3.groupby(['card_id'])['purchase_amount'].var().reset_index()
VarAmt_by_Card_Merch4 = hist_df4.groupby(['card_id'])['purchase_amount'].var().reset_index()
VarAmt_by_Card_Merch5 = hist_df5.groupby(['card_id'])['purchase_amount'].var().reset_index()
VarAmt_by_Card_Merch = pd.concat([VarAmt_by_Card_Merch1, VarAmt_by_Card_Merch2,VarAmt_by_Card_Merch3,VarAmt_by_Card_Merch4,VarAmt_by_Card_Merch5]).groupby('card_id')['purchase_amount'].sum().reset_index()
VarAmt_by_Card_Merch.head()
VarAmt_by_Card_Merch.to_csv('VarAmt_by_Card_Merch.csv')
MedianAmt_by_Card_Merch1 = hist_df1.groupby(['card_id'])['purchase_amount'].median().reset_index()
MedianAmt_by_Card_Merch2 = hist_df2.groupby(['card_id'])['purchase_amount'].median().reset_index()
MedianAmt_by_Card_Merch3 = hist_df3.groupby(['card_id'])['purchase_amount'].median().reset_index()
MedianAmt_by_Card_Merch4 = hist_df4.groupby(['card_id'])['purchase_amount'].median().reset_index()
MedianAmt_by_Card_Merch5 = hist_df5.groupby(['card_id'])['purchase_amount'].median().reset_index()
MedianAmt_by_Card_Merch = pd.concat([MedianAmt_by_Card_Merch1, MedianAmt_by_Card_Merch2,MedianAmt_by_Card_Merch3,MedianAmt_by_Card_Merch4,MedianAmt_by_Card_Merch5]).groupby('card_id')['purchase_amount'].mean().reset_index()
MedianAmt_by_Card_Merch.to_csv('MedianAmt_by_Card_Merch.csv')