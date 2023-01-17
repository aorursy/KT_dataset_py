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
data = pd.read_csv('/kaggle/input/top-women-chess-players/top_women_chess_players_aug_2020.csv')
data.head()
sns.pairplot(data[['Title','Standard_Rating','Rapid_rating','Blitz_rating']],height=5,hue='Title')
sns.catplot(x="Title", y="Year_of_birth", jitter=False, data=data, kind='violin')

data_2=data[(data['Title']=='GM')|(data['Title']=='IM')]

sns.catplot(x="Federation", y="Year_of_birth", data=data_2, row='Title',kind='swarm',height=4, aspect=4);