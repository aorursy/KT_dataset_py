# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd



df = pd.read_csv('/kaggle/input/steam-video-games/steam-200k.csv',

                                  names=["id", "Game", "Transaction", "attribute", "noise"])
df.groupby(['customerId','categoryTags']).sum()
df['Game'].unique().shape

#  12393 * 5155
df_purchase =  df.loc[df['Transaction'] == 'purchase', ['id','Game','Transaction','attribute']].rename(columns={'attribute': 'purchased'})

df_play = df.loc[df['Transaction'] == 'play', ['id','Game','attribute']].rename(columns={'attribute': 'play_hours'})



print('shape of purchase dataset : ', df_purchase.shape[0])

print('shape of play dataset : ', df_play.shape[0])



game_dataset = pd.merge(df_purchase, df_play, left_on=['id','Game'], right_on=['id','Game'], how='left').fillna(0) #left join -> if purchased but not played then means 0 hrs
game_dataset.loc[game_dataset['id'] == 151603712]