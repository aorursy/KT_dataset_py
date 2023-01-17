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
df = pd.read_csv("/kaggle/input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv")
df.head()
# Select the IMDb column
df["IMDb"]
# Select the Rotten Tomatoes column
df["Rotten Tomatoes"]
df.info()
# Data wrangling: convert the percents to numbers
df['Rotten Tomatoes'] = df['Rotten Tomatoes'].str.rstrip('%').astype('float')
df['Rotten Tomatoes']
# Data wrangling: drop any rows that have missing data
df.dropna(axis=0, inplace=True)
df.shape
# Import library and dataset
import matplotlib.pyplot as plt
import seaborn as sns
 
sns.distplot( df["IMDb"] )
plt.show()
sns.distplot( df["Rotten Tomatoes"] )
plt.show()
# Choose two variables to plot# If you have an independent variable, it goes on the x-axis

sns.regplot(x=df["Rotten Tomatoes"], y=df["IMDb"])
plt.show()
