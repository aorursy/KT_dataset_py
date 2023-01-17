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

print(pd.get_option("display.max_rows"))

pd.set_option('display.max_rows', 4000)

! pip install pixiedust

import pixiedust

df_final_leaderboard_order = pd.read_csv('/kaggle/input/ashrae-interim-scores/final_leaderboard_order.csv')

df_final_leaderboard_order.shape[0]

df_final_leaderboard_order['PublicScore_Rank'] = df_final_leaderboard_order['PublicScore'].rank().values.astype('int64')

df_final_leaderboard_order['PrivateScore_Rank'] = df_final_leaderboard_order['PrivateScore'].rank().values.astype('int64')

df_final_leaderboard_order['Shake'] = df_final_leaderboard_order['PublicScore_Rank'] - df_final_leaderboard_order['PrivateScore_Rank']
df_final_leaderboard_order
display(df_final_leaderboard_order)