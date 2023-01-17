import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



path = "../input"

filename = '/ATP.csv'

df = pd.read_csv(path + filename)



df.head(5)
df = df[(df.loser_hand == 'R') | (df.loser_hand == 'L')]

df = df[(df.winner_hand == 'R') | (df.winner_hand == 'L')]



df['w_ace'] = df.w_ace.fillna(-1)

df['l_ace'] = df.l_ace.fillna(-1)



df['w_ace'] = df.w_ace.apply(lambda x: float(x))

df['l_ace'] = df.l_ace.apply(lambda x: float(x))



df['w_df'] = df.w_df.apply(lambda x: float(x))

df['l_df'] = df.l_df.apply(lambda x: float(x))



df = df[df.w_ace>-1]

df = df[df.l_ace>-1]
R = len(df[(df.winner_hand == 'R') & (df.loser_hand == 'L')])

L = len(df[(df.winner_hand == 'L') & (df.loser_hand == 'R')])

print('The right hand player has a',round(100*np.true_divide(R, R + L),3), '% Chance of winning against a left hand player')
sns.distplot(np.random.binomial(R + L, 0.5, 100000),np.arange(9300,9900,1))

plt.plot(R,0,'o', markersize = 20)

plt.title('Binomial distribution of ' + str(L+R) + ' Coin tosses')

plt.xlabel('Number of wins for the right-handed players')

plt.legend(['# of actual right-hand wins over left-handed players','Binomial Distribution'], loc = 2)

print(R)
rand = np.random.binomial(R + L, 0.5, 100000)

print('P = ',100*np.true_divide(len(rand[rand>R]),100000),'%')
plt.figure(figsize = (10,10))

plt.subplot(2,1,1)

df['left_aces'] = -1*np.ones(len(df))

df['right_aces'] = -1*np.ones(len(df))



df['left_aces'][df.loser_hand == 'L'] = df.l_ace

df['left_aces'][df.winner_hand == 'L'] = df.w_ace



df['right_aces'][df.loser_hand == 'R'] = df.l_ace

df['right_aces'][df.winner_hand == 'R'] = df.w_ace



left = df.left_aces[df.left_aces > -1]

right = df.right_aces[df.right_aces > -1]



sns.distplot(right, np.arange(0,max(right),1))

sns.distplot(left, np.arange(0,max(right),1))

plt.xlim([0,30])

plt.xlabel('Aces per game')

plt.legend(['R','L'])

plt.title('Aces')



plt.subplot(2,1,2)

df['left_df'] = -1*np.ones(len(df))

df['right_df'] = -1*np.ones(len(df))



df['left_df'][df.loser_hand == 'L'] = df.l_df

df['left_df'][df.winner_hand == 'L'] = df.w_df



df['right_df'][df.loser_hand == 'R'] = df.l_df

df['right_df'][df.winner_hand == 'R'] = df.w_df



left_df = df.left_df[df.left_df > -1]

right_df = df.right_df[df.right_df > -1]



sns.distplot(right_df, np.arange(0,max(left_df),1))

sns.distplot(left_df, np.arange(0,max(left_df),1))

plt.xlim([0,30])

plt.legend(['R','L'])

plt.xlabel('Double faults per game')

plt.title('Double Faults')



print('Number of left hand players...',len(left))

print('Number of right hand players...',len(right))
plt.figure(figsize = (10,10))

plt.subplot(2,1,1)

sns.distplot(df.w_ace[df.winner_hand == 'R'], np.arange(0,max(df.w_ace),1))

sns.distplot(df.w_ace[df.winner_hand == 'L'], np.arange(0,max(df.w_ace),1))

plt.xlim([0,30])

plt.legend(['R','L'])

plt.title('Winners')

plt.xlabel('Aces per match')



plt.subplot(2,1,2)

sns.distplot(df.w_ace[df.loser_hand == 'R'], np.arange(0,max(df.w_ace),1))

sns.distplot(df.w_ace[df.loser_hand == 'L'], np.arange(0,max(df.w_ace),1))

plt.xlim([0,30])

plt.legend(['R','L'])

plt.xlim([0,30])

plt.legend(['R','L'])

plt.title('Losers')

plt.xlabel('Aces per match')
plt.figure(figsize = (10,10))



plt.subplot(2,1,1)

sns.distplot(df.w_ace[(df.winner_hand == 'R') & (df.loser_hand == 'L')], np.arange(0,max(df.w_ace),1))

sns.distplot(df.w_ace[(df.loser_hand == 'R') & (df.winner_hand == 'L')], np.arange(0,max(df.w_ace),1))

plt.xlim([0,30])

plt.legend(['R aces (When R won)','L aces (When L won)'])

plt.title('Winners')

plt.xlabel('Aces per game')



plt.subplot(2,1,2)

sns.distplot(df.l_ace[(df.loser_hand == 'R') & (df.winner_hand == 'L')], np.arange(0,max(df.w_ace),1))

sns.distplot(df.l_ace[(df.winner_hand == 'R') & (df.loser_hand == 'L')], np.arange(0,max(df.w_ace),1))

plt.title('Losers')

plt.xlim([0,30])

plt.legend(['R aces (When L won)','L aces (When W won)'])

plt.xlabel('Aces per game')
plt.figure(figsize = (10,10))

plt.subplot(2,1,1)

sns.distplot(df.w_df[df.winner_hand == 'R'], np.arange(0,max(df.w_df),1))

sns.distplot(df.w_df[df.winner_hand == 'L'], np.arange(0,max(df.w_df),1))

plt.xlim([0,15])

plt.legend(['Double Fualt R',' Double Fault L'])

plt.title('Winners')

plt.xlabel('Double Fualts')



plt.subplot(2,1,2)

sns.distplot(df.l_df[df.loser_hand == 'R'], np.arange(0,max(df.w_df),1))   

sns.distplot(df.l_df[df.loser_hand == 'L'], np.arange(0,max(df.w_df),1))

plt.xlim([0,15])

plt.legend(['Double Fualt R',' Double Fault L'])

plt.title('Losers')

plt.xlabel('Double Fualts')
plt.figure(figsize = (9,9))



plt.subplot(2,1,1)

plt.bar(1,np.mean(df.l_df[df.loser_hand == 'L']), color = 'r')

plt.bar(2,np.mean(df.l_df[df.loser_hand == 'R']), color = 'b')

plt.bar(3,np.mean(df.w_df[df.winner_hand == 'L']), color = 'r')

plt.bar(4,np.mean(df.w_df[df.winner_hand == 'R']), color = 'b')

plt.ylabel('Double Faults per game')

plt.xticks([1,2,3,4],['Losers Left','Losers right','Winners left','Winners right'])

plt.title('Double Faults')



plt.subplot(2,1,2)

plt.bar(1,np.mean(df.l_ace[df.loser_hand == 'L']), color = 'r')

plt.bar(2,np.mean(df.l_ace[df.loser_hand == 'R']), color = 'b')



plt.bar(3,np.mean(df.w_ace[df.winner_hand == 'L']), color = 'r')

plt.bar(4,np.mean(df.w_ace[df.winner_hand == 'R']), color = 'b')

plt.ylabel('Aces per game')

plt.title('Aces')



plt.xticks([1,2,3,4],['Losers Left','Losers right','Winners left','Winners right'])
plt.figure(figsize = (9,9))



plt.subplot(2,1,1)

plt.bar(1,np.mean(df.l_df[(df.winner_hand == 'R') & (df.loser_hand == 'L')]), color = 'r')

plt.bar(2,np.mean(df.l_df[(df.winner_hand == 'L') & (df.loser_hand == 'R')]), color = 'b')

plt.bar(3,np.mean(df.w_df[(df.winner_hand == 'L') & (df.loser_hand == 'R')]), color = 'r')

plt.bar(4,np.mean(df.w_df[(df.winner_hand == 'R') & (df.loser_hand == 'L')]), color = 'b')

plt.ylabel('Double Faults per game')

plt.xticks([1,2,3,4],['Losers Left','Losers right','Winners left','Winners right'])

plt.title('Double Faults')



plt.subplot(2,1,2)

plt.bar(1,np.mean(df.l_ace[(df.winner_hand == 'R') & (df.loser_hand == 'L')]), color = 'r')

plt.bar(2,np.mean(df.l_ace[(df.winner_hand == 'L') & (df.loser_hand == 'R')]), color = 'b')

plt.bar(3,np.mean(df.w_ace[(df.winner_hand == 'L') & (df.loser_hand == 'R')]), color = 'r')

plt.bar(4,np.mean(df.w_ace[(df.winner_hand == 'R') & (df.loser_hand == 'L')]), color = 'b')

plt.ylabel('Aces per game')

plt.xticks([1,2,3,4],['Losers Left','Losers right','Winners left','Winners right'])

plt.title('Aces')
df['winner_ht'] = df.winner_ht.fillna(-1)

df['loser_ht'] = df.winner_ht.fillna(-1)



df['winner_ht']  = df.winner_ht.apply(lambda x: float(x))

df['loser_ht'] = df.loser_ht.apply(lambda x: float(x))





df = df[df['winner_ht']>-1]

df = df[df['loser_ht']>-1]



df['left_ht'] = -1*np.ones(len(df))

df['right_ht'] = -1*np.ones(len(df))



df['left_ht'][df.loser_hand == 'L'] = df.loser_ht

df['left_ht'][df.winner_hand == 'L'] = df.winner_ht



df['right_ht'][df.loser_hand == 'R'] = df.loser_ht

df['right_ht'][df.winner_hand == 'R'] = df.winner_ht



left_ht = df.left_ht[df.left_ht > -1]

right_ht = df.right_ht[df.right_ht > -1]







sns.distplot(right_ht)

sns.distplot(left_ht)

plt.xlabel('Height')

plt.legend(['R','L'])

print('Average Height - Left hand...',np.mean(left_ht))

print('Average Height - Right hand...',np.mean(right_ht))
print(np.unique(right_ht))