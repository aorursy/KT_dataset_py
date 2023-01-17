! conda install -c conda-forge lifelines -y
import numpy as np

import pandas as pd

from lifelines import KaplanMeierFitter

import matplotlib.pyplot as plt
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

train = train.query('Sex=="Male" & Age>60')

train.SmokingStatus.value_counts(normalize=True)
display(train.FVC.hist(),

train.FVC.mean())
train['LowFVC'] = train.FVC.lt(train.FVC.mean()).astype(int)

train.LowFVC.value_counts()
CONFIDENCE = 0.90



idx1 =  (train.SmokingStatus == 'Never smoked')

idx2 = (train.SmokingStatus.isin(['Ex-smoker', 'Currently smokes']))



durations1 = train.loc[idx1, 'Weeks']

durations2 = train.loc[idx2, 'Weeks']



events1 = train.loc[idx1, 'LowFVC']

events2 = train.loc[idx2, 'LowFVC']



kmf1 = KaplanMeierFitter()

kmf1.fit(durations1, events1, alpha=(1-CONFIDENCE), label='Never Smoked')



kmf2 = KaplanMeierFitter()

kmf2.fit(durations2, events2, alpha=(1-CONFIDENCE), label='Smoked')



plt.clf()
%matplotlib inline



plt.figure(figsize=(12,8))

plt.style.use('seaborn-whitegrid')

SMALL_SIZE = 16

MEDIUM_SIZE = 18

BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)

plt.rc('axes', titlesize=MEDIUM_SIZE)

plt.rc('axes', labelsize=MEDIUM_SIZE)

plt.rc('xtick', labelsize=MEDIUM_SIZE)

plt.rc('ytick', labelsize=MEDIUM_SIZE)

plt.rc('legend', fontsize=SMALL_SIZE)

plt.rc('figure', titlesize=BIGGER_SIZE)





p1 = kmf1.plot()

p2 = kmf2.plot(ax=p1)





plt.xlim(-5, 118)

plt.title("Maintaining Lung Capacity")

plt.xlabel("Weeks since baseline CT")

plt.ylabel("Fraction of Group with above average FVC")

plt.show()
