%matplotlib notebook

import matplotlib.pyplot as plt

import matplotlib



import numpy as np

import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 100)

pd.set_option('max_colwidth', 120)



import warnings

warnings.filterwarnings("ignore", message="More than 20 figures have been opened")



import datetime,itertools
dataTemp = '../input/data-science-bowl-2019/{}.csv'

train             = pd.read_csv(dataTemp.format('train'))

train_labels      = pd.read_csv(dataTemp.format('train_labels'))

train = train[train.installation_id.isin(train_labels.installation_id.values)].copy()
train.timestamp = pd.to_datetime(train.timestamp).dt.tz_localize(None)
alldf = train.sort_values('timestamp')

alldf = alldf.iloc[::100].copy()
np.random.seed(50)

ids = np.random.choice(train.installation_id.unique(),size=50,replace=False)

train = train[train.installation_id.isin(ids)].copy()
day = 24 * 3600

for iid,df in itertools.chain(train.groupby('installation_id'), [('all',alldf)]):

    

    



        t0 = df.timestamp.min()



        timeSincet0 = (df.timestamp - datetime.datetime(t0.year,t0.month,t0.day)).dt.total_seconds().values

        assert all(np.diff(timeSincet0) >=0)



        def plotOnCircles(v,*args,**kwargs):

            r = v/timeSincet0[-1]

            plt.plot(r * np.cos(v/day*2*np.pi),r * np.sin(v/day*2*np.pi),*args,**kwargs)





        plt.figure(figsize=(10,10))



        plt.axvline(0,color='y',alpha=0.2)

        plt.axhline(0,color='y',alpha=0.2)





        plotOnCircles(np.linspace(0,timeSincet0[-1],10000),'k-',alpha=0.3)

        

        for t,c in zip(['Clip', 'Activity', 'Game', 'Assessment'],['y','c','g','r']):

            plotOnCircles(timeSincet0[(df.type==t).values],c+'o',alpha=0.3,label=t,ms=10)



        for o in range(24):

            plt.text(1.05 * np.cos(o/24.*np.pi*2),1.05 * np.sin(o/24.*np.pi*2),str(o))



        plt.xlim(-1.1,1.1)

        plt.ylim(-1.1,1.1)

        plt.gca().set_aspect('equal')

        plt.axis('off')

        plt.title(str(iid))

        plt.legend()
