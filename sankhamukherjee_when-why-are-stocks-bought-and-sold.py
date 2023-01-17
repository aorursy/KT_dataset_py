import numpy as np 

import pandas as pd 

from scipy import signal

import matplotlib.pyplot as plt



import kagglegym
# This part is going to be for explorind the dataset ...

# so we want the entire dataset ..

with pd.HDFStore("../input/train.h5", "r") as train:

    df = train.get("train")
dfId = df[['id', 'timestamp', 'y']].groupby('id').agg([

                    np.min, np.max, len, 

                lambda m: (list(m)[0] - list(m)[-1])/np.abs(np.mean(list(m))) ]  ).reset_index()

dfId.sort_values([('timestamp', 'amax')], inplace=True, ascending=False)

print(dfId.head())

print(dfId['y'].columns)
plt.plot(dfId[('timestamp', 'amin')], dfId['id'], '.', mfc='green', mec='None', label='bought')

plt.plot(dfId[('timestamp', 'amax')], dfId['id'], '.', mfc='red',   mec='None', label='sold')

plt.xlabel('timestamp')

plt.ylabel('stock number')

plt.legend()
plt.scatter( dfId[('timestamp', 'amax')], 

             dfId['id'], 

             c    = dfId[('y', '<lambda>')], 

             s    = dfId[('timestamp', 'len')]/10,

             cmap = plt.cm.BrBG, vmin=-40, vmax=40).set_alpha(0.6)

plt.colorbar()
import seaborn as sns

dfStats = df[['y', 'id']].groupby('id').agg([np.median, np.std, np.min, np.max, np.mean]).reset_index()

dfStats.sort_values( ('y', 'median'), inplace=True )

print( dfStats.head() )

print( dfStats['y'].apply(np.median) )





sns.violinplot( dfStats[('y',  'amin')]   , color='orange')

sns.violinplot( dfStats[('y',  'median')] , color='teal')

sns.violinplot( dfStats[('y',  'amax')]   , color='indianred')





plt.figure()

temp = sns.kdeplot(dfStats[('y', 'amin')]  )

temp = sns.kdeplot(dfStats[('y', 'median')])

temp = sns.kdeplot(dfStats[('y', 'mean')]  )

temp = sns.kdeplot(dfStats[('y', 'amax')]  )

plt.yscale('log')



plt.figure()

temp = sns.kdeplot(dfStats[('y', 'std')])

plt.figure()

temp = sns.kdeplot(df['y'])

plt.yscale('log')
# Finding distributions of the result. 

# This is an entire portfolio. It will 

# be good to see how each variable changes 

# independent of each other ...

# -------------------------------------------



for i, (idVal, dfG) in enumerate(df[['id', 'timestamp', 'y']].groupby('id')):

    if i> 100: break

    df1 = dfG[['timestamp', 'y']].groupby('timestamp').agg(np.mean).reset_index()

    plt.plot(df1['timestamp'], np.cumsum(df1['y']),label='%d'%idVal)
# Finding distributions of the result. 

# This is an entire portfolio. It will 

# be good to see how each variable changes 

# independent of each other ...

# -------------------------------------------



for i, (idVal, dfG) in enumerate(df[['id', 'timestamp', 'y']].groupby('id')):

    if i> 100: break

    #df1 = dfG[['timestamp', 'y']].groupby('timestamp').agg(np.mean).reset_index()

    #plt.plot(df1['timestamp'], np.cumsum(df1['y']),label='%d'%idVal)

    dfG.head()
df2 = df[['id', 'timestamp', 'y']].pivot_table(values='y', index='timestamp', columns='id').reset_index(False)

df2.head()
cols = [ c for c in df2.columns if str(c) != 'timestamp']

lags = [1]

aCorrs = []

for i, c in enumerate(cols):

    try:

        aCorrs.append((c , max([(df2[c].autocorr(lag)) for lag in lags])))

    except:

        pass

    

aCorrs = pd.DataFrame(aCorrs, columns=['id', 'maxAcorr']).sort_values('maxAcorr', ascending=False)

print(aCorrs.head())
lags = range(1, 15)

for c in list(aCorrs.id)[:10]:

    plt.figure()

    plt.plot(list(df2[c])[:-1], list(df2[c])[1:], 's')

    plt.title(str(c))
cols = [ c for c in df2.columns if str(c) != 'timestamp']

corrs = df2[cols].corr()
corrs
temp = np.where(np.triu(corrs) < -0.9)

temp = [sorted(a) for a in zip(temp[0], temp[1]) if a[0]!=a[1]]
prevId = -1

for i, (a, b) in enumerate(temp):

    

    if a != prevId:

        plt.figure()

        prevId = a

        plt.plot(np.cumsum(df2.ix[:, a]), label='id=%d'%a)

    plt.plot(np.cumsum(df2.ix[:, b]), label='id=%d'%b)

    plt.legend()

    

    if i > 5: break

    
list(df.columns)