# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from sklearn.utils import shuffle

## READ DATASET

aa0 = pd.read_csv("../input/train.csv", sep=",", header = 0)

aa02 = pd.read_csv("../input/macro.csv", sep=",", header = 0)
## MERGE DATABASE

df = pd.merge(aa0, aa02, how = "left", on="timestamp")

df.columns.get_loc("price_doc")

df = shuffle(df)
## DELETE VARIABLES WITH MORE THAN 15% OF MISSING  VALUES

miss = [df.iloc[:,i].isnull().sum()/df.shape[0] for i in range(0, df.shape[1])]

delete = np.where(np.array(miss)>.15)[0]

df2=df.drop(df.columns[delete], axis=1)
## TURN NOMINAL DATA INTO NUMERICAL VALUES

categoricals = []

for i in range(0, df2.shape[1]):

    if df2.iloc[:,i].dtype=="object":

        categoricals.append(1)

    else:

        categoricals.append(0)

for i in np.where(np.array(categoricals)==1)[0]:

    df2.iloc[:,i] = pd.Categorical(df2.iloc[:,i])

    df2.iloc[:,i] = df2.iloc[:,i].cat.codes
## REPLACE MISSING VALUE BY MEAN

for i in range(0, df2.shape[1]):

    df2.iloc[:,i]=df2.iloc[:,i].fillna(df2.iloc[:,i].mean())
## CHECK CORRELATIONS

ab = df2.corr()



ab.shape
ab[259]