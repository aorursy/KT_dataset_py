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
df = pd.read_csv("../input/vgsales.csv")

df #preview data
gc = df.Platform.value_counts().head(10).sort_values(ascending=True)

gc.plot.barh() #bar graph for top 10 platforms
gc.sort_values(ascending=False) #actual values
grp = df.groupby('Genre')

import seaborn as sb

sb.heatmap(grp['NA_Sales','EU_Sales','JP_Sales','Other_Sales', 'Global_Sales'].sum(), annot=True, fmt=".2f")
NA = grp['NA_Sales'].sum()

NA.sort_values(ascending=False).head(3)
EU = grp['EU_Sales'].sum()

EU.sort_values(ascending=False).head(3)
JP = grp['JP_Sales'].sum()

JP.sort_values(ascending=False).head(3)
Others = grp['Other_Sales'].sum()

Others.sort_values(ascending=False).head(3)
Global = grp['Global_Sales'].sum()

Global.sort_values(ascending=False).head(5)
got = df.groupby(['Year','Genre']).size()

got