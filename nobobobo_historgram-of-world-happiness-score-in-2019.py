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
df = pd.read_csv('/kaggle/input/world-happiness/2019.csv')
df.head()

print(df['Score'].min())
print(df['Score'].max())
import matplotlib.pyplot as plt

score = df['Score']
result = plt.hist(score, bins=30, color='c', edgecolor='k', alpha=0.65)
plt.axvline(score.mean(), color='k', linestyle='dashed', linewidth=1)
plt.axvline(score.median(), color='b', linestyle='dashed', linewidth=1)

min_ylim, max_ylim = plt.ylim()
plt.text(score.mean()*1.05, max_ylim*0.9, 'Mean: {:.2f}'.format(score.mean()))
plt.text(score.median()*1.05, max_ylim*0.8, 'Median: {:.2f}'.format(score.median()))

plt.title('Histogram of World Happiness Score in 2019')