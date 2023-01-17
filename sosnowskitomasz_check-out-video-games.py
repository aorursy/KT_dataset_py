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
# Quick check of some sample data

vg_data = pd.read_csv("../input/vgsales.csv")

df = pd.DataFrame(vg_data)

df.head()
# Number of Wii 

df['Platform'].value_counts().plot(x="Platform", y="# of entries", kind='bar', title="Total Platform Counts in Provided Data")
# Total Wii Sales in Global Arena

sum(df[df['Platform']=="Wii"]['Global_Sales'])
df.groupby('Platform').sum()['Global_Sales'].plot(x="Platform", y="Sales", kind='bar', title="Global Sales")
df.groupby('Genre').sum()['Global_Sales'].plot(x="Genre", y="Sales", kind='bar', title="Global Sales for different Genre")
df.groupby('Publisher').sum().head(10)['Global_Sales'].plot(x="Publisher", y="Sales", kind='bar', title="Global Sales for top 10 publishers")