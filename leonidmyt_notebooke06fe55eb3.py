# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



df_train = pd.read_csv("../input/train.csv")

df_train['Gender'] = df_train['Sex'].map({'male': 1, 'female': 0}).astype(int)

df_train.hist(by=['Sex'], column=['Survive'])