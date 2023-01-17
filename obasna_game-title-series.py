# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_users_game1_df = pd.read_csv("../input/ds2019uec-task2/train_users_game1.csv")

train_users_game2_df = pd.read_csv("../input/ds2019uec-task2/train_users_game2.csv")

test_users_game1_df = pd.read_csv("../input/ds2019uec-task2/test_users_game1.csv")
train_users_game1_df.head()
train_users_game2_df.head()
test_users_game1_df.head()
series_game = pd.Series(np.unique(train_users_game1_df['game_title'])).str[:8].value_counts()

series_game
#上位１００個のシリーズを表示

series_game = series_game[:100]
import seaborn as sns

plt.figure(figsize=(20,10))

sns.barplot(x=series_game.index,y=series_game)

plt.xticks(rotation=90)