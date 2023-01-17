import numpy as np 

import pandas as pd 

from scipy import signal

import matplotlib.pyplot as plt



import kagglegym
# This part is going to be for explorind the dataset ...

# so we want the entire dataset ..

with pd.HDFStore("../input/train.h5", "r") as train:

    df = train.get("train")
list(set([c.split('_')[0] for c in df.columns]))
# Finding distributions of the result



df1 = df[['timestamp', 'y']].groupby('timestamp').agg([np.mean, np.std, len]).reset_index()

df1.head()
n     = df1['timestamp']

yMean = np.array(df1['y']['mean'])

yStd  = np.array(df1['y']['std'])



plt.figure()

plt.plot(n, yMean, '.')

plt.xlabel('n')

plt.ylabel('$y_{mean}$[n]')



plt.figure()

plt.plot(n, yStd, '.')

plt.xlabel('n')

plt.ylabel('$y_{std}$[n]')



plt.figure()

plt.plot(np.array(df1['y']['len']), yStd, '.')

plt.xlabel('len')

plt.ylabel('$y_{std}$[n]')



plt.figure()

plt.plot(np.array(df1['y']['len']), yMean, '.')

plt.xlabel('len')

plt.ylabel('$y_{mean}$[n]')



plt.figure()

plt.plot(np.diff(yMean), '.')

plt.xlabel('n')

plt.ylabel('$y_{mean}$[n] $-$ $y_{mean}$[n-1]')



plt.figure()

plt.plot( yMean[:-1] , yMean[1:] , '.')

plt.xlabel('$y_{mean}$[n]')

plt.ylabel('$y_{mean}$[n-1]')
#Calculate running averages ...

#yMeanRA = np.cumsum( yMean )/np.linspace( 1, len(yMean), len(yMean) )

#yStdRA  = np.cumsum( yStd )/np.linspace( 1, len(yMean), len(yMean) )



yMeanCumsum = np.cumsum( yMean )

f, t, Syy   =  signal.spectrogram(yMeanCumsum)



slope1 = np.mean(yMean)

slope2 = np.mean(df['y'])

slope3 = np.mean(df['y'][df.timestamp < 906])



plt.figure()

plt.plot(n, yMeanCumsum, '.')

plt.plot(n, slope1*n, color='black', lw=2 ) 

plt.plot(n, slope2*n, color='orange', lw=2 ) 

plt.plot(n, slope2*n, color='indianred', lw=2 ) 

plt.xlabel('n')

plt.ylabel('$y_{mean}$[n] - Cumsum')



plt.figure()

plt.pcolormesh(t, f, np.log10(Syy))

plt.ylabel('Frequency')

plt.xlabel('Time')

plt.colorbar()
# Remember that the training contains timestamps upto 905

np.mean(yMean), np.mean(df['y']), np.mean(df['y'][df.timestamp < 906])
dev = yMeanCumsum - np.mean(yMean)*n



plt.figure()

plt.plot(n, dev, '.')

plt.axhline(color='black', ls='--')

plt.xlabel('n')

plt.ylabel('$y_{mean}$[n] - without the slope')



plt.figure()

plt.plot(dev[1:], dev[:-1], '.')

plt.xlabel('$y_{std}$ [n]')

plt.ylabel('$y_{std}$ [n-1]')



plt.figure()

plt.plot(np.abs(dev), yStd, '.')

plt.xlabel('deviations from the mean')

plt.ylabel('$y_{std}$')
def getScore(slope):

    rewards = []

    print(slope)

    env = kagglegym.make()

    observation = env.reset()



    while True:

        target    = observation.target

        timestamp = observation.features["timestamp"][0]

        target['y'] = slope



        observation, reward, done, info = env.step(target)

        rewards.append(reward)

        if done: break

            

    return info['public_score'], rewards
slope = 0.0002182

x1 = np.linspace(slope - 0.0001, slope+0.0001 , 4)

y1 = [ getScore(m) for m in x1 ]

y1, rewards = zip(*y1)



plt.plot(x1, y1, 's-', color='green', mfc='green', mec='black')

plt.figure()

for r in rewards:

    plt.plot(r)
list(map(np.mean, rewards))
columns = [c for c in df.columns if 'fundamental' in c]

columns1 = columns + ['timestamp']

df2 = df[columns1].groupby('timestamp').agg([np.mean]).reset_index()



i = 0; N=5

while True:

    

    if i >= len(columns): break

    for j in range(N):    

        if i >= len(columns): break

        

        if j == 0:

            plt.figure(figsize=(8,8.0/N))

        

        plt.axes([j*1.0/N,0,1.0/N,1])

        plt.plot(df2['timestamp'], df2[ columns[i] ])

        plt.xticks([]); plt.yticks([])

        i += 1 
import seaborn as sns

corrColumns = [c for c in df2.columns if 'timestamp'not in c]

fundamentals = np.array([df2[c] for c in corrColumns])

Corr = np.corrcoef( dev, fundamentals )

sns.clustermap( pd.DataFrame(np.abs(Corr), columns=['---y--']+[c[0].split('_')[1] for c in corrColumns]) )