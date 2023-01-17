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
import pandas as pd

df1=pd.read_csv("../input/winemag-data-130k-v2.csv", index_col=0)

df1.head()
df2=pd.read_csv("../input/winemag-data_first150k.csv", index_col=0)

df2.head()
df3=pd.read_json("../input/winemag-data-130k-v2.json")

df3.head()
final_df=pd.concat([df1, df2, df3], ignore_index=True)

final_df.head()
final_df.tail()
final_df.info()
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(25,7))

plt.xticks(rotation=90)

sns.countplot(final_df.country)
pd.crosstab(final_df.country, final_df.points)
pd.crosstab(final_df.country, final_df.price)