import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")

tips=sns.load_dataset("tips")

tips
sns.relplot(x='total_bill',y='tip',data=tips)
sns.relplot(x="total_bill", y="tip", hue="smoker", data=tips);

sns.relplot(x="total_bill", y="tip", hue="smoker", style="smoker", data=tips);

sns.relplot(x="total_bill", y="tip", hue="size", data=tips);



sns.relplot(x="total_bill", y="tip",hue="size", size="size", data=tips);





g = sns.relplot(x="total_bill", y="tip", kind="line",hue="size", data=tips)
fmri = sns.load_dataset("fmri")

fmri.head()
g=sns.relplot(x="timepoint",y="signal",kind="line",data=fmri)
g=sns.relplot(x='timepoint',y='signal',data=fmri,kind='line',ci=None)
g=sns.relplot(x='timepoint',y='signal',data=fmri,kind='line',estimator=None,ci=None)
g=sns.relplot(x='timepoint',y='signal',data=fmri,kind='line',ci='sd')
g=sns.relplot(x='timepoint',y='signal',data=fmri,kind='line')
g=sns.relplot(x='timepoint',y='signal',data=fmri,kind='line',hue="event",ci='sd')
g=sns.relplot(x='timepoint',y='signal',data=fmri,kind='line',hue="event",style="region",ci='sd')
g=sns.relplot(x='timepoint',y='signal',data=fmri,kind='line',hue="event",style="region",ci='sd',markers=True)
df = pd.DataFrame(dict(time=pd.date_range("2017-1-1", periods=500),

                       value=np.random.randn(500).cumsum()))

g = sns.relplot(x="time", y="value", kind="line", data=df)
fmri
g=sns.relplot(x='timepoint',y='signal',

             kind='line',hue='event',

            style='event', col='subject', data=fmri)
g=sns.relplot(x='timepoint',y='signal',

             kind='line',hue='event',

            style='event', col='subject', data=fmri,

             col_wrap=5)
g=sns.relplot(x='timepoint',y='signal',

             kind='line',hue='event',

            style='event', col='subject', data=fmri,

            height=3, aspect=.75, linewidth=2.5,

             col_wrap=5)
