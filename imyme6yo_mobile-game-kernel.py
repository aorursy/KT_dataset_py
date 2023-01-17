# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
my_df = pd.read_csv('../input/17k-apple-app-store-strategy-games/appstore_games.csv', index_col='ID')

my_df.columns
my_df.drop(['Subtitle', 'Developer', 'Description', 'URL', 'Icon URL', 'Languages', 'Primary Genre', 'Subtitle', 'Genres', 'Current Version Release Date', 'Size'], axis=1, inplace=True)

my_df.columns
!pip install missingno
import missingno as mg

mg.matrix(my_df,color=(1,0,0))

plt.show()
# fill na

my_df[['Average User Rating', 'User Rating Count']] = my_df[['Average User Rating', 'User Rating Count']].fillna(0, axis=1)

mg.matrix(my_df,color=(1,0,0))

plt.show()



# my_df[['Average User Rating', 'User Rating Count']]
my_df.info()
my_df.describe().round(3)
my_df.query("Price==0")["Free"] = True

my_df.query("Price!=0")["Free"] = False

my_df['InAppPayment'] = my_df['In-app Purchases'].apply(lambda x: True if type(x) == str else False)

pay_to_play_game = my_df.query("FreeGame==False")

free_game = my_df.query("FreeGame==True and InAppPayment==False")

free_to_play_game = my_df.query("FreeGame==True and InAppPayment==True")
pay_to_play_game
free_game
free_to_play_game
print(f"Free Game: -> {len(free_game)}")

print(f"Free-to-play Game: -> {len(free_to_play_game)}")

print(f"Pay-to-play Game: -> {len(pay_to_play_game)}")

plt.scatter(pay_to_play_game["Price"], pay_to_play_game["Average User Rating"])

plt.show()