import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
path='../input/health-care-data-set-on-heart-attack-possibility/heart.csv'
df=pd.read_csv('../input/health-care-data-set-on-heart-attack-possibility/heart.csv')
df.head()
plt.figure(figsize=(8,8))

sns.barplot(x='sex',y='age',data=df)
plt.figure(figsize=(16,8))

sns.heatmap(df.corr(),annot=True)