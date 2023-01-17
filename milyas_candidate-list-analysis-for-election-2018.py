# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

for i in (os.listdir("../input")):
    print(i)

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/National Assembly Candidates List - 2018 Updated.csv")
df.head()
df['Party']= df['Party'].replace('Pakistan Seriaki Party', 'PS')
for i in df['Party'].unique():
    print(i)
ax = df.groupby(['Party'])['NA#'].count().nlargest(20).plot(kind='bar',
                                    figsize=(18,8),
                                    title="Top 20 Parties Candidate Wise")
ax.set_xlabel("Parties")
ax.set_ylabel("Counts")
plt.show()
"Total Candidate: "+str(df.groupby(['Party'])['NA#'].count().sum())
 
ax = df.groupby(['Province'])['NA#'].count().nlargest(4) .plot(kind='bar',
                                    figsize=(14,8),
                                    title="Total Number of Candidate Province Wise")
ax.set_xlabel("Provinces")
ax.set_ylabel("Counts")
plt.show()
for i in df['Province'].unique():
    print(i+": "+str(len(df[df['Province'] == i]['Party'].unique())))
    #break
df.groupby(['NA#']).size().mean()
#df.groupby(['Province']).df['Party']=="Independent"
df[df['Party']=="Independent"].groupby("Province").size()
#df.groupby(['Province']).df['Party']=="Independent"
ax =df[df['Party']=="Independent"].groupby("Province").size().plot(kind='bar',figsize=(18,8),title="Number of Independent Candidate Per Province")
ax.set_xlabel("Province")
ax.set_ylabel("Counts")
plt.show()
