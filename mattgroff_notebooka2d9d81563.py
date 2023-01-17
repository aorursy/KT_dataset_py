# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
%matplotlib inline 
sns.set(color_codes=True)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/plays.csv')
#display top 5 results
df.head(5)





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#display the bottom 5 results 
df.tail(5)
#return number of rows and columns 
df.shape
#basic stats on each column 
df.describe()
#return the names of each columns
df.columns
#return the number of unique values for each column
df.nunique()
correlation = df.corr()
sns.heatmap(correlation, xticklabels = correlation.columns, yticklabels = correlation.columns)
df_week1 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/week1.csv')
df_week2 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/week2.csv')
df_week3 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/week3.csv')
df_week4 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/week4.csv')
df_week5 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/week5.csv')
df_week6 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/week6.csv')
df_week7 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/week7.csv')
df_week8 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/week8.csv')
df_week9 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/week9.csv')
df_week10 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/week10.csv')
df_week11 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/week11.csv')
df_week12 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/week12.csv')
df_week13 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/week13.csv')
df_week14 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/week14.csv')
df_week15 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/week15.csv')
df_week16 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/week16.csv')
df_week17 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/week17.csv')
df_tracking = pd.concat([df_week1,df_week2,df_week3,df_week4,df_week5,df_week6,df_week7,df_week8,df_week9,df_week10,df_week11,df_week12,df_week13,df_week14,df_week15,df_week16,df_week17])
df_tracking.describe()