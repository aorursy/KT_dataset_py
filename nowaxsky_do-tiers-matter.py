import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# live = pd.read_csv('input/live.csv',index_col=0)

back = pd.read_csv('../input/most_backed.csv',index_col=0)
back.info()
# Now I want to turn str to num format, but some data in "pledge.tier" contains "...", so I drop that.



t = []

for i,row in enumerate(back['pledge.tier']):

    if '...' in row:

        t.append(i)

print(t)
back['pledge.tier'].iloc[981]
back = back.drop(back.index[t])
# Convert str to float.

def num_tier(string):    

    return len(string[1:-1].split(', '))



def tier_num(string):

    l = []

    for s in string[1:-1].split(', '):

        l.append(float(s))

    return l



back['num of tier'] = back['pledge.tier'].apply(num_tier)

back['pledge.tier_num'] = back['pledge.tier'].apply(tier_num)

back['num.backers.tier_num'] = back['num.backers.tier'].apply(tier_num)
back = back[back['currency']=='usd']

back.reset_index(drop=True,inplace=True)
back_clean = back[['title','amt.pledged','category','goal','num.backers','num of tier',

                   'pledge.tier_num', 'num.backers.tier_num']]
back_clean.head()
back_clean.info()
# I build the contribution in each tier in the every case.



tier_contrib = pd.Series()

for i in range(len(back_clean.index)):

    a = np.array(back_clean['pledge.tier_num'].iloc[i])

    b = np.array(back_clean['num.backers.tier_num'].iloc[i])

    contrib = pd.Series(list([a*b*100/np.sum(a*b)]))

    tier_contrib = tier_contrib.append(contrib)
tier_contrib.reset_index(drop=True,inplace=True)

tier_contrib.head()
# "tier_contrib%" means the contribution in each tier.



back_clean['tier_contrib%'] = tier_contrib

back_clean.head()
back_clean['num of tier'].plot.hist(bins=50)
# I build a new column to show which tier is the most contribution.

# (pledge.tier_num is in order.)



def argmax(tier):

    return tier.argmax()+1



back_clean['argmax_tier_contrib%'] = back_clean['tier_contrib%'].apply(argmax)

back_clean.head()
fig = plt.figure(figsize=(12,6))

sns.barplot(x=back_clean.groupby('argmax_tier_contrib%').count()['amt.pledged'].index,

            y=back_clean.groupby('argmax_tier_contrib%').count()['amt.pledged'])

#plt.xlim([0,10])
# max contribution by position in the whole pricing strategy



sns.distplot(back_clean['argmax_tier_contrib%']/back_clean['num of tier'],kde=False,bins=50)
# Now I use number scale to locate the position of the most contribution price. 



contrib_in_tier = pd.Series()

for i in range(len(back_clean.index)):

    diff = back_clean['pledge.tier_num'][i][-1]-back_clean['pledge.tier_num'][i][0]

    if diff != 0:

        temp = pd.Series((back_clean['pledge.tier_num'][i][back_clean['argmax_tier_contrib%'][i]-1]-back_clean['pledge.tier_num'][i][0])/diff)

    else:

        temp = pd.Series(1)

        

    contrib_in_tier = contrib_in_tier.append(temp*100)
sns.distplot(contrib_in_tier,kde=False,bins=50)