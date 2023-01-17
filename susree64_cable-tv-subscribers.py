import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import os
print("The file name that has the data is " ,os.listdir("../input"))

df = pd.read_csv("../input/CableTVSubscribersData.csv")
df.head()
segment = df['Segment'].value_counts(). reset_index()
segment.columns = ['Segment', 'Count'] # Changed the column names
plt.figure(figsize= (20,5)) # Make a plot size
trace = sns.barplot(x = segment['Segment'], y = segment['Count'], data = segment)
# Adding values on the top of the bars
for index, row in segment.iterrows():
    trace.text(x = row.name, y = row.Count+ 2, s = str(row.Count),color='black', ha="center" )
plt.show()
set = df[['Segment', 'subscribe']]
grouped = set.groupby(['Segment', 'subscribe']).size()
plt.figure(figsize= (20,5)) # Make a plot size
#trace = sns.barplot(x = segment['Segment'], y = segment['Count'], data = segment)
trace = grouped.plot(kind = 'bar')
grouped.unstack()
df[['income','ownHome' ]].groupby(['ownHome']).mean()
