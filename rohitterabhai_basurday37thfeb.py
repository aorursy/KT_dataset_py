# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv("../input/apndcts/apndcts.csv")

df.head()
from sklearn.model_selection import train_test_split

print(df.shape)

df_train, df_test=train_test_split(df,test_size=0.3, random_state=12)

print(df_train.shape)

print(df_test.shape)

from sklearn.model_selection import KFold

kf=KFold(n_splits=5)

for train_index, test_index in kf.split(df):

    df_train = df.iloc[train_index]

    df_test=df.iloc[test_index]

    print("training data size:", df_train.shape)

    print("test data size:", df_test.shape)