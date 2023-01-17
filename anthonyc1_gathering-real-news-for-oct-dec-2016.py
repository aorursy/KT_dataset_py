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
df1 = pd.read_csv("../input/articles1.csv")
df2 = pd.read_csv("../input/articles2.csv")
df3 = pd.read_csv("../input/articles3.csv")
df1 = df1.loc[df1['year'] == 2016.0]
df1 = df1.loc[df1['month'].isin([10.0,11.0,12.0])]
df1.head(10)
df2 = df2.loc[df2['year'] == 2016.0]
df2 = df2.loc[df2['month'].isin([10.0,11.0,12.0])]
df2.head(10)
df3 = df3.loc[df3['year'] == 2016.0]
df3 = df3.loc[df3['month'].isin([10.0,11.0,12.0])]
df3.head(10)
publications1 = np.unique(df1["publication"].values)
publications2 = np.unique(df2["publication"].values)
publications3 = np.unique(df3["publication"].values)
print(publications1)
print(publications2)
print(publications3)
df = pd.concat([df1,df2,df3])
df = df.loc[df['publication'].isin(['New York Times', 'CNN','Atlantic','Fox News','Guardian','National Review','NPR','Reuters', 'Vox','Washington Post'])]
df
df.to_csv('real_news.csv', encoding='utf-8')