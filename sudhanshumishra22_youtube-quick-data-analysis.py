import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns 

df=pd.read_csv("../input/data.csv")

df['Subscribers'] = pd.to_numeric(df['Subscribers'], errors='coerce')

df['Video Uploads'] = pd.to_numeric(df['Video Uploads'], errors='coerce')

df['Grade'] = df['Grade'].astype(str)
df.info()
df
df.head(20).plot.bar(x = 'Channel name', y = 'Subscribers')

plt.xlabel('Channel Name')

plt.ylabel('Subscribers')

plt.show()
df.head(20).plot.bar(x = 'Channel name', y = 'Video Uploads')

plt.xlabel('Channel Name')

plt.ylabel("Video Uploads")

plt.show()
df.head(20).plot.bar(x = 'Channel name', y = 'Video views')

plt.xlabel('Channel Name')

plt.ylabel("Video Views")

plt.show()
sns.heatmap(df.corr(), cmap = 'RdGy')

plt.title('Correlation Matrix Plot')