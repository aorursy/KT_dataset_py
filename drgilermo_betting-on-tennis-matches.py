# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting

import matplotlib

matplotlib.style.use('fivethirtyeight')

import os

import seaborn as sns ## plotting histograms



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
tennis_df = pd.read_csv('../input/Data.csv', encoding = 'latin1') # Read Data

tennis_df.WRank = pd.to_numeric(tennis_df.WRank, errors = 'coerce') 

tennis_df.LRank = pd.to_numeric(tennis_df.LRank, errors = 'coerce')

# New Feature: Rank difference betweehn the 2 oponents

tennis_df['Diff'] =  tennis_df.LRank - tennis_df.WRank 

# New Feature: Round the rank difference to 10's and 20's

tennis_df['Round_10'] = 10*round(np.true_divide(tennis_df.Diff,10))

tennis_df['Round_20'] = 20*round(np.true_divide(tennis_df.Diff,20))

# New Feature: Total number of sets in the match

tennis_df['Total Sets'] = tennis_df.Wsets + tennis_df.Lsets



tennis_df.W3 = tennis_df.W3.fillna(0)

tennis_df.W4 = tennis_df.W4.fillna(0)

tennis_df.W5 = tennis_df.W5.fillna(0)

tennis_df.L3 = tennis_df.L3.fillna(0)

tennis_df.L4 = tennis_df.L4.fillna(0)

tennis_df.L5 = tennis_df.L5.fillna(0)



tennis_df['Sets Diff'] = tennis_df.W1+tennis_df.W2+tennis_df.W3+tennis_df.W4+tennis_df.W5 - (tennis_df.L1+tennis_df.L2+tennis_df.L3+tennis_df.L4+tennis_df.L5)

new_df = tennis_df



# 2 New Data Frames: Grand Slam data frame (GS) and non-Grand Slam data frame (non GS)

df_non_GS = new_df[~(new_df.Series == 'Grand Slam')]

df_GS = new_df[new_df.Series == 'Grand Slam']
bins = np.arange(10,200,10)

Gs_prob = []



for bi in bins:

    

    pos = bi

    neg = -pos

    

    pos_wins = len(df_GS[df_GS.Round_10 == pos])

    neg_wins = len(df_GS[df_GS.Round_10 == neg])

    Gs_prob.append(np.true_divide(pos_wins,pos_wins + neg_wins))

    



prob = []



for bi in bins:

    

    W = np.true_divide(1,np.mean(df_GS.AvgW[df_GS.Round_10 == bi]))

    L = np.true_divide(1,np.mean(df_GS.AvgL[df_GS.Round_10 == bi]))

    ratio = np.true_divide(1,L + W)

    part_ratio = (ratio - 1)/2 + 1

    prob.append(W/part_ratio)

    

plt.figure(figsize = (10,10))    

plt.hold(True)

plt.bar(bins,prob, width = 10, color = 'blue')

plt.bar(bins,Gs_prob, width = 8, color = 'red')

plt.xlabel('Rank Difference', fontsize = 15)

plt.ylabel('Winning Probability', fontsize = 15)

plt.xlim([10,200])

plt.ylim([0.5,1])

plt.title('Are Betting Markets Good Predictors in Grand Slam Tournaments?', fontsize = 15)

plt.legend(['Betting Market Odds','Reality'], loc = 2, fontsize = 15)

plt.show()
df_non_GS = df_non_GS[~np.isnan(df_non_GS.AvgW)]

money_over = 0

money_under = 0

money_track_over = []

money_track_under = []



for row in df_non_GS.iterrows():

    

    if row[1].Diff>0:

        money_over = money_over + row[1].AvgW - 1

        money_under = money_under - 1

        

    else:

        money_over = money_over - 1

        money_under = money_under + row[1].AvgW - 1

        

    money_track_over.append(money_over)

    money_track_under.append(money_under)

    

    if np.isnan(money_over):

        break

        

    if np.isnan(money_under):

        break





plt.figure()

plt.hold(True)

plt.plot(money_track_under,'b')

plt.plot(money_track_over,'r')

plt.xlabel('Non-Grand Slam games', fontsize = 15)

plt.ylabel('Money Balance [$]', fontsize = 15)

plt.title('Assuming a 1$ bet on each non Grand Slam game', fontsize = 15)

plt.legend(['Betting on the Underdog', 'Betting on the Favorite'], loc = 3, fontsize = 15)

plt.show()
df_GS = df_GS[~np.isnan(df_GS.AvgW)]

money_over = 0

money_under = 0

money_track_over = []

money_track_under = []



for row in df_GS.iterrows():

    

    if row[1].Diff>0:

        money_over = money_over + row[1].AvgW - 1

        money_under = money_under - 1

        

    else:

        money_over = money_over - 1

        money_under = money_under + row[1].AvgW - 1

        

    money_track_over.append(money_over)

    money_track_under.append(money_under)

    

    if np.isnan(money_over):

        break

        

    if np.isnan(money_under):

        break





plt.figure()

plt.hold(True)

plt.plot(money_track_under,'b')

plt.plot(money_track_over,'r')

plt.xlabel('Grand Slam games',fontsize = 15)

plt.ylabel('Money Balance [$]', fontsize = 15)

plt.title('Assuming a 1$ bet on each Grand Slam game')

plt.legend(['Betting on the Underdog', 'Betting on the Favorite'], loc = 3, fontsize = 15)

plt.show()