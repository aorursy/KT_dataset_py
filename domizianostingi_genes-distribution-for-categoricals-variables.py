

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from scipy.stats import kendalltau

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px
train_features = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")

train_targets_scored = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")

train_targets_unscored = pd.read_csv("/kaggle/input/lish-moa/train_targets_nonscored.csv")

test_features = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")

sample_submission = pd.read_csv("/kaggle/input/lish-moa/sample_submission.csv")
#Check if ctl_vehicle actually have some MoA or not

x = pd.concat([train_targets_scored.sum(axis=1), train_features['cp_type']], axis= 1 )

print('total number of MoA for all element with cp_type = ctl_vehicle : ' ,x[x.cp_type == 'ctl_vehicle'].sum(axis=0)[0])
d1 = train_features[train_features.cp_dose=='D1']

d2= train_features[train_features.cp_dose=='D2']
plt.figure(dpi=150)

plt.title('RELATION AMONG GENES AND DOSE') 

sns.distplot(d1.filter(regex='^g').mean(axis=0),label='d1')

sns.distplot(d2.filter(regex='^g').mean(axis=0),label='d2')

plt.legend()

plt.show
cp24 = train_features[train_features.cp_time == 24]

cp48 = train_features[train_features.cp_time == 48]

cp72 = train_features[train_features.cp_time == 72]
plt.figure(dpi=150)

plt.title('RELATION AMONG GENES AND TIME')

sns.distplot(cp24.filter(regex='^g').mean(axis=0),label='cp24')

sns.distplot(cp48.filter(regex='^g').mean(axis=0),label='cp48')

sns.distplot(cp72.filter(regex='^g').mean(axis=0),label='cp72')

plt.legend()

plt.show()
cp_trt = train_features[train_features.cp_type =='trt_cp']

cp_vehicle = train_features[train_features.cp_type =='ctl_vehicle']
plt.figure(dpi=150)

plt.title('RELATION AMONG GENES AND TREATMENT')

sns.distplot(cp_trt.filter(regex='^g').mean(axis=0),label='trt_cp')

sns.distplot(cp_vehicle.filter(regex='^g').mean(axis=0),label='ctl_vehicle')

plt.legend()

plt.show()