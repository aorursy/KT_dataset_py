import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import os
print(os.listdir("../input"))


df = pd.read_csv("../input/World_Cups.csv")
df

sns.pairplot(df)
attend = df[['Year', 'Attendance']]
plt.figure(figsize = (15,5))
ax = sns.barplot(data=attend, x='Year', y='Attendance' )
# plot regplot with numbers 0,..,len(a) as x value
sns.regplot(x=np.arange(0,len(attend)), y = 'Attendance', data = attend)
#sns.despine(offset=10, trim=False)
ax.set_ylabel("Attendance")
plt.show()
winners = df[['Year', 'Winner', 'Runners-Up']]
plt.figure(figsize = (15,5))
ax = sns.countplot(x="Winner", data=winners)

stats = df[['Country','Goals Scored', 'Matches Played']]
grouped = stats.groupby('Country', as_index = False).sum().sort_values('Goals Scored', ascending = False)
grouped['Average'] = grouped['Goals Scored']/grouped['Matches Played']
grouped
