# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/wine-reviews"))

print(os.listdir("../input/most-common-wine-scores"))





# Any results you write to the current directory are saved as output.
test=pd.read_csv("../input/most-common-wine-scores/top-five-wine-score-counts.csv")

test.head()
#train=pd.read_csv("../input/top-five-wine-scores.csv")

#train.head()
train=pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)

train.head()
varieties=train.groupby(['variety', ])['points'].count().nlargest(5).index

varieties
df = train.loc[train['variety'].isin(varieties)]

df.head()
df2=df.groupby(['points', 'variety'])['variety'].count().sort_values(ascending=False)

df2=df2.to_frame().unstack()

df2=df2.reset_index()

df2.head()

test.head()