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
data_df = pd.read_csv("../input/amazon_alexa.tsv", sep='\t', header=0)
data_df.head()
data_df.describe()
data_df.feedback[data_df.feedback == 1].count()
data_df.feedback[data_df.feedback == 0].count()
pd.isnull(data_df).sum()
min(pd.to_datetime(data_df.date))
min(pd.to_datetime(data_df.date))