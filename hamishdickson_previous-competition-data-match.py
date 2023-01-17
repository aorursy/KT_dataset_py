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
df_comp_1 = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')

df_comp_2 = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
df_comp_1.head(3)
df_comp_2.head(3)
df_both = df_comp_2[df_comp_2['comment_text'].isin(df_comp_1['comment_text'])]

df_both
len(df_both['comment_text'].unique())