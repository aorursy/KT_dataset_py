# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
!pip install lifelines
from lifelines import KaplanMeierFitter,CoxPHFitter
from lifelines.datasets import load_gbsg2 
from lifelines.statistics import logrank_test
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/survival-unemployment/survival_unemployment.csv')
print(df.isnull().any())
print(df.describe())
df.head()
plt.hist(df['spell'])
plt.title('survival time')
plt.xlabel('score(spell)')
plt.ylabel('freq')
plt.hist(df['logwage'])
plt.title('lowgage')
plt.xlabel('score(lowgage)')
plt.ylabel('freq')
plt.hist(df['age'])
plt.title('age')
plt.xlabel('score(age)')
plt.ylabel('freq')
corr = df.corr()
print(corr)
for name, group in df.groupby('ui'):
    plt.hist(group['spell'],alpha=0.7,label='UI:'+str(name))
    plt.legend()
    plt.title('survival time')
    plt.xlabel('score(spell)')
    plt.ylabel('freq')
ui_df = df.groupby('ui')
km = KaplanMeierFitter()

for name, group in ui_df:
    km.fit(group['spell'],event_observed=group['event'],label='UI:'+str(name))
    km.plot()
    
plt.title('KM-Curve')
plt.ylabel('survival rate')
plt.show()
cph = CoxPHFitter()
cph.fit(df, duration_col="spell", event_col="event")
cph.print_summary()
cph.plot_covariate_groups('ui', [0, 1])
plt.title('survival time')
plt.xlabel('score(spell)')
plt.ylabel('freq')
d1 = df[df['ui']==1]
d0 = df[df['ui']==0]
results = logrank_test(d1['spell'], d0['spell'], d1["event"], d0["event"])
results.print_summary()