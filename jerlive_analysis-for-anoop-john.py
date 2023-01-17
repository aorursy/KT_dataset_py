# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/shield/shield-july-1-13.csv")
df
df.columns

df['createdAt']=pd.to_datetime(df['createdAt'])
df['createdAt']
sliced=df[['numViews','numLikes','createdAt']]
sliced
import matplotlib.pyplot as plt
ax=sliced['numViews'].plot()
ax.set_xlabel("Time")
ax.set_ylabel("Views")
plt.show()
ax=sliced['numLikes'].plot()
ax.set_xlabel("Time")
ax.set_ylabel("Likes")
plt.show()
sliced2=df[['numViews','numLikes','numComments']]
sliced2
sliced2=sliced2.sort_values(by=['numComments'])
sliced2=sliced2.set_index('numComments')
ax=sliced2['numViews'].plot(kind='bar')
ax.set_xlabel("Comments")
ax.set_ylabel("Views")
plt.show()
ax=sliced2['numLikes'].plot(kind='bar')
ax.set_xlabel("Comments")
ax.set_ylabel("Likes")
plt.show()
sliced3=df.sort_values(by=['numEmojies'])
sliced3=sliced3[['numViews','numLikes','numEmojies']]
sliced3=sliced3.set_index('numEmojies')
ax=sliced3['numViews'].plot(kind='bar')
ax.set_xlabel("Emojis")
ax.set_ylabel("Views")
plt.show()
ax=sliced3['numLikes'].plot(kind='bar')
ax.set_xlabel("Emojis")
ax.set_ylabel("Likes")
plt.show()
sliced4=df.sort_values(by=['numHashtags'])
sliced4=sliced4[['numViews','numLikes','numHashtags']]
sliced4=sliced4.set_index('numHashtags')
ax=sliced4['numViews'].plot(kind='bar')
ax.set_xlabel("Number of Hashtags")
ax.set_ylabel("Views")
plt.show()
ax=sliced4['numLikes'].plot(kind='bar')
ax.set_xlabel("Number of Hashtags")
ax.set_ylabel("Likes")
plt.show()
sliced4=df.sort_values(by=['numLikes'])
sliced4=sliced4[['numViews','numLikes']]
sliced4=sliced4.set_index('numLikes')
ax=sliced4['numViews'].plot(kind='bar')
ax.set_xlabel("Likes")
ax.set_ylabel("Views")
plt.show()
