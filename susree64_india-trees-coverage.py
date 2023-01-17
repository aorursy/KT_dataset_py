import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import os
print("The file name for the analysis is ", os.listdir("../input"))
# Read the data into dataframe
df = pd.read_csv('../input/StatewiseTreeCover.csv')
df.head(10)
def convert(a):
    return (int(a.replace(',', '')))
df['Geographical - Area'] = df['Geographical - Area'].apply(convert)
df['Tree Cover - Area'] = df['Tree Cover - Area'].apply(convert)
df['Tree Cover - Per cent of GA'] = round(df['Tree Cover - Area']/df['Geographical - Area'] * 100, 2)
df.head()

df[df['Tree Cover - Per cent of GA'] == df['Tree Cover - Per cent of GA']. max()]

df[df['Tree Cover - Per cent of GA'] == df['Tree Cover - Per cent of GA']. min()]
state_cover = df[['State/ Uts','Tree Cover - Per cent of GA']]
state_cover = state_cover.iloc[0:35, :]
state_cover = state_cover.sort_values("Tree Cover - Per cent of GA", ascending=False)

f, ax = plt.subplots(figsize=(15, 10))
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
ax = sns.barplot(x="Tree Cover - Per cent of GA", y="State/ Uts", data=state_cover)
ax.set(ylabel="",
       xlabel="Percentage green cover")
plt.show()





