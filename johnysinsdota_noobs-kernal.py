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
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("../input/us-border-crossing-data/Border_Crossing_Entry_Data.csv")
df.State.unique()
import seaborn as sns
sns.catplot(y="State", kind="count", data=df)
df['State'].value_counts().plot(kind="bar")

plt.show()
df.Border.unique()
df.Date=pd.to_datetime(df.Date)

df['Total']=df.groupby(df.Date.dt.year)["Value"].transform('sum')
df['year']=df.Date.dt.year

df["Year"]=df.Date.dt.year
import seaborn as sns
sns.lineplot("Year","Total",data=df)


i=df.groupby(['year','Border'])[["Value"]].sum().reset_index()
sns.lineplot("year","Value",data=i,hue="Border")