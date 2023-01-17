# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')
columns=df_train.drop(["ID","y"],axis=1).columns.tolist()

sameentries=df_train.groupby(columns).apply(lambda x: list(x.index)).tolist()
ywidth=list()

for i in sameentries:

    if len(i)>1:

        ywidth.append(df_train.iloc[i]['y'].max()-df_train.iloc[i]['y'].min())
np.mean(ywidth)
hits=plt.hist(ywidth,bins=np.mgrid[0:50:2],color='red')