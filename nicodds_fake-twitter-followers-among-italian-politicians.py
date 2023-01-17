# Loading relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
names = ['others', 'renzi', 'salvini', 'di_maio']
labels = ['Random users', 'Renzi', 'Salvini', 'Di Maio']
# We use this locals trick to avoid unmeaningful variable names
for i,name in enumerate(names):
    filename = "../input/%s.csv" % name
    locals()[name] = pd.read_csv(filename, encoding='utf-8', keep_default_na=False)
    locals()[name]['bio_length'] = locals()[name]['description'].apply(len)
print("""
\t Number of Tweets: some stats
--------------------------------------------------------
    \t\t|Renzi\t|Salvini|Di     |Radom
    \t\t|       |       | Maio  | Sample
--------------------------------------------------------
Sample:\t\t|%i\t|%i\t|%i\t|%i
Mean:\t\t|%3.1f\t|%3.1f\t|%3.1f\t|%3.1f
Median: \t|%3.1f\t|%3.1f\t|%3.1f\t|%3.1f
Dev Std:\t|%3.1f\t|%3.1f\t|%3.1f\t|%3.1f
Max:\t\t|%i\t|%i\t|%i\t|%i
Min:\t\t|%i\t|%i\t|%i\t|%i
""" %(len(renzi), len(salvini), len(di_maio), len(others),
     renzi.tweets.mean(), salvini.tweets.mean(), di_maio.tweets.mean(), others.tweets.mean(),
     renzi.tweets.median(), salvini.tweets.median(), di_maio.tweets.median(), others.tweets.median(),
     renzi.tweets.std(), salvini.tweets.std(), di_maio.tweets.std(), others.tweets.std(),
     renzi.tweets.max(), salvini.tweets.max(), di_maio.tweets.max(), others.tweets.max(),
     renzi.tweets.min(), salvini.tweets.min(), di_maio.tweets.min(), others.tweets.min(),
     ))
print("""
\tNumber of Followers: some stats
--------------------------------------------------------
    \t\t|Renzi\t |Salvini|Di     |Random
    \t\t|        |       | Maio  | Sample   
--------------------------------------------------------
Sample: \t|%i\t |%i\t |%i\t |%i
Media:\t\t|%3.1f\t |%3.1f\t |%3.1f\t |%3.1f
Median: \t|%3.1f\t |%3.1f\t |%3.1f\t |%3.1f
Dev Std:\t|%3.1f\t |%3.1f |%3.1f |%3.1f
Max:\t\t|%i |%i|%i |%i
Min:\t\t|%i\t |%i\t |%i\t |%i
""" %(len(renzi), len(salvini), len(di_maio), len(others),
     renzi.followers.mean(), salvini.followers.mean(), di_maio.followers.mean(), others.followers.mean(),
     renzi.followers.median(), salvini.followers.median(), di_maio.followers.median(), others.followers.median(),
     renzi.followers.std(), salvini.followers.std(), di_maio.followers.std(), others.followers.std(),
     renzi.followers.max(), salvini.followers.max(), di_maio.followers.max(), others.followers.max(),
     renzi.followers.min(), salvini.followers.min(), di_maio.followers.min(), others.followers.min()
     ))
data_bio = []
for n in names:
    tmp_data = locals()[n]
    data_bio.append(tmp_data[(tmp_data.tweets <= 100) & (tmp_data.followers <= 10000) ].bio_length)
sns.set_context("paper",font_scale=2)
sns.set_style("white")
sns.despine(left=True)

fig, ax = plt.subplots(figsize=(13,7))

ax.hist(data_bio, 10, histtype='bar', align='mid', label=labels, alpha=0.4, density=True)
#ax = sns.distplot(data_bio, label=labels[i], bins=50, kde=False, norm_hist=True)

#ax.hist(data, 10, histtype='bar', align='mid', label=names, alpha=0.4)
ax.legend()
ax.set_xlabel('Number of characters in the bio')
plt.title('Bio length distribution')
plt.show()
data = []
for n in names:
    tmp_data = locals()[n]
    data.append(tmp_data[(tmp_data.tweets <= 100) & (tmp_data.followers <= 10000) ].tweets)
sns.set_context("paper",font_scale=2)
sns.set_style("white")
sns.despine(left=True)

fig, ax = plt.subplots(figsize=(13,7))

#for i in [3,2,1,0]:
#    ax = sns.(data[i], label=labels[i], bins=50, kde=False, norm_hist=True)

ax.hist(data, 10, histtype='bar', align='mid', label=labels, alpha=0.4, density=True)
ax.legend()
ax.set_xlabel('Number of tweets')
plt.title('Per user tweets distribution (truncated)')
plt.show()
twlim=100
plt.subplots(2, 2, figsize=(18,10))
plt.subplot(221)
plt.xlim(0,500)
plt.plot(di_maio[di_maio.tweets<=twlim].followers.values, di_maio[di_maio.tweets<=twlim].tweets.values, 'yo', alpha=0.1)
plt.ylabel("Tweets Count")
plt.title(labels[3])

plt.subplot(222)
plt.xlim(0,500)
plt.plot(renzi[renzi.tweets <= twlim].followers.values,renzi[renzi.tweets <= twlim].tweets.values, 'ro', alpha=0.1)
plt.title(labels[1])

plt.subplot(223)
plt.xlim(0,500)
plt.plot(salvini[salvini.tweets<=twlim].followers.values,salvini[salvini.tweets<=twlim].tweets.values,'bo', alpha=0.1)
plt.xlabel("Followers Count")
plt.ylabel("Tweets Count")
plt.title(labels[2])

plt.subplot(224)
plt.xlim(0,500)
plt.plot(others[others.tweets<=twlim].followers.values, others[others.tweets<=twlim].tweets.values, 'go', alpha=0.1)
plt.xlabel("Followers Count")
plt.title(labels[0])

plt.show()
# Computing the share on the total
renzi_share_tweets   = len(renzi[renzi.tweets <=twlim])/len(renzi)*100.0
salvini_share_tweets = len(salvini[salvini.tweets <=twlim])/len(salvini)*100.0
di_maio_share_tweets = len(di_maio[di_maio.tweets <=twlim])/len(di_maio)*100.0
others_share_tweets  = len(others[others.tweets <=twlim])/len(others)*100.0
print("""Followers with %i tweets or less:
Renzi:   %i (%.2f%% of the sample)
Salvini: %i (%.2f%% of the sample)
Di Maio: %i (%.2f%% of the sample)
Altri:   %i (%.2f%% of the sample)
""" %(twlim,
      len(renzi[renzi.tweets <= twlim]), renzi_share_tweets,
      len(salvini[salvini.tweets <=twlim]), salvini_share_tweets,
      len(di_maio[di_maio.tweets <= twlim]), di_maio_share_tweets,
      len(others[others.tweets <= twlim]), others_share_tweets))
    
plt.rc('font', size=22)
plt.subplots(figsize=(13,7))
plt.grid()
plt.bar(labels, [others_share_tweets, renzi_share_tweets, salvini_share_tweets, di_maio_share_tweets])
plt.ylabel('%')
plt.ylim(0,100)
plt.title('Share of followers with 100 or less tweets (respect sample)')
plt.show()
