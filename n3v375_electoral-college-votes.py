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
import seaborn as sns

import matplotlib.pyplot as plt



df = pd.read_csv('/kaggle/input/us-electoral-college-votes-per-state-17882020/Electoral_College.csv')
df.head()
df.info()
df.describe()
df['Year'].nunique()
df['State'].nunique()
df.dropna(inplace=True)
def main():

    for each in df['State'].unique():

        df_state = df[df['State'] == each]

        plt.figure(figsize=(18,6))

        chart = sns.barplot(df_state['Year'], df_state['Votes'], palette='rainbow')

        chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='center')

        plt.title(f'{each} Votes')

        yield
state = main()
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)