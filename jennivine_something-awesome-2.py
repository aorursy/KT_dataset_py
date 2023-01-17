# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
games = pd.read_csv('../input/catanstats.csv')
games.groupby('player').agg({'me':'count'})
seaborn.boxplot(games['player'], 
                games['points'], 
                palette=seaborn.light_palette("purple")).set(xlabel='Placement Order', 
                                                             ylabel='Points', 
                                                             title='Settlers of Catan Placement Order vs Points', 
                                                             ylim=(0,14))

#NOTE: From table above, indeed this player has played more times as player 2
#      so this graph is inconclusive of anything
seaborn.regplot(games['robberCardsGain'], 
                games['points'], 
                color='purple').set(xlabel='Robber Card Gain', 
                                    ylabel='Points', 
                                    title='Robber Card Gain vs Points', 
                                    ylim=(1,13))
seaborn.regplot(games['tradeGain'], 
                games['points'], 
                color='purple').set(xlabel='Trade Gain', 
                                    ylabel='Points', 
                                    title='Trade Gain vs Points', 
                                    ylim=(1,13))
seaborn.regplot(games['production'], 
                games['points'], 
                color='purple').set(xlabel='Production', 
                                    ylabel='Points', 
                                    title='Production vs Points', 
                                    ylim=(1,13))
games_trib = games[np.abs(games['tribute'] - np.mean(games['tribute'])) <= 3*games['tribute'].std()]
seaborn.regplot(games_trib['tribute'], 
                games_trib['points'], 
                color='purple').set(xlabel='Cards lost to Tribute', 
                                    ylabel='Points', 
                                    title='Tribute Loss vs Points', 
                                    ylim=(1,13))
seaborn.regplot(games_trib['totalGain'], 
                games_trib['points'], 
                color='purple').set(xlabel='Total Gain', 
                                    ylabel='Points', 
                                    title='Total Gain vs Points', 
                                    ylim=(1,13))