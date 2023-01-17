# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/camera_dataset.csv',low_memory=False)
df.head(1)
df.describe()
df['Release date'].value_counts().sort_index().plot(kind='bar')
df['Range'] = (df['Zoom tele (T)'] - df['Zoom wide (W)']).abs()
g_df = df.groupby('Release date')
plt.figure(figsize=(10,5))

plt.scatter(x=df['Release date'], y=df['Max resolution'],c='green')

plt.scatter(x=df['Release date'], y=df['Low resolution'],c='red',alpha=0.4)

g_df['Max resolution'].max().plot(color='green',legend=True)

g_df['Low resolution'].max().plot(color='red',legend=True)
plt.figure(figsize=(10,5))

plt.scatter(x=df['Release date'], y=df['Zoom tele (T)'],c='green')

plt.scatter(x=df['Release date'], y=df['Zoom wide (W)'],c='red',alpha=0.4)

g_df['Zoom tele (T)'].max().plot(color='green',legend=True)

g_df['Zoom wide (W)'].min().plot(color='red',legend=True)
plt.figure(figsize=(10,5))

plt.scatter(x=df[df['Range'] >25]['Release date'], y=df[df['Range'] >25]['Range'],c='green')

g_df['Range'].max().plot(color='green',legend=True)
plt.figure(figsize=(10,5))

plt.scatter(x=df[(df['Range'] >100) & (df['Range'] < 200)]['Release date'], y=df[(df['Range'] >100) & (df['Range'] < 200)]['Price'],c='green')

plt.scatter(x=df[(df['Range'] >200) & (df['Range'] < 250)]['Release date'], y=df[(df['Range'] >200) & (df['Range'] < 250)]['Price'],c='Yellow')

plt.scatter(x=df[df['Range'] >250]['Release date'], y=df[df['Range'] >250]['Price'],c='Red', alpha=0.2)
plt.figure(figsize=(10,5))

plt.scatter(x=df['Release date'], y=df['Storage included'],c='green',alpha=0.2)

g_df['Storage included'].max().plot(color='green',legend=True)
plt.figure(figsize=(10,5))

plt.scatter(x=df[(df['Range'] >100) & (df['Range'] < 200)]['Release date'], y=df[(df['Range'] >100) & (df['Range'] < 200)]['Dimensions'],c='green')

plt.scatter(x=df[(df['Range'] >200) & (df['Range'] < 250)]['Release date'], y=df[(df['Range'] >200) & (df['Range'] < 250)]['Dimensions'],c='Yellow')

plt.scatter(x=df[df['Range'] >250]['Release date'], y=df[df['Range'] >250]['Dimensions'],c='Red', alpha=0.2)

g_df['Dimensions'].min().plot(color='blue',legend=True)

g_df['Dimensions'].max().plot(color='black',legend=True)
plt.figure(figsize=(10,5))

plt.scatter(x=df[(df['Range'] >100) & (df['Range'] < 200)]['Release date'], y=df[(df['Range'] >100) & (df['Range'] < 200)]['Weight (inc. batteries)'],c='green')

plt.scatter(x=df[(df['Range'] >200) & (df['Range'] < 250)]['Release date'], y=df[(df['Range'] >200) & (df['Range'] < 250)]['Weight (inc. batteries)'],c='Yellow')

plt.scatter(x=df[df['Range'] >250]['Release date'], y=df[df['Range'] >250]['Weight (inc. batteries)'],c='Red', alpha=0.2)

g_df['Weight (inc. batteries)'].min().plot(color='blue',legend=True)

g_df['Weight (inc. batteries)'].max().plot(color='black',legend=True)