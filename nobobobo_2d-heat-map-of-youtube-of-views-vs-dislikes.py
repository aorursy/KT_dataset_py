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
import matplotlib.pyplot as plt

from scipy import stats 
# read csv file: USvideos.csv

df = pd.read_csv('/kaggle/input/youtube-new/USvideos.csv')
# show 5 records from the data

df.head()
df[['views','dislikes']].describe()
plt.hist(df['views'],bins=20)
plt.hist(df['dislikes'],bins=20)
plt.hist2d(df['views'], df['dislikes'], bins=(100, 100))

plt.xlabel('# Of Views)')

plt.ylabel('# Of Dislikes)')

plt.show()
plt.hist2d(np.log(df['views']), np.log(df['dislikes']+1), bins=(100, 100))

plt.xlabel('Log(# Of Views)')

plt.ylabel('Log(# Of Dislikes)')

plt.show()