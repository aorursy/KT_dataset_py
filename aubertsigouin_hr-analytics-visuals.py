import numpy as np # linear algebra



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import seaborn as sns 

import matplotlib.pyplot as plt

sns.set_palette("deep", desat=.6)
df = pd.read_csv('../input/HR_comma_sep.csv')
df.head()
# DISTRIBUTION PLOT
#df['average_montly_hours'].plot(kind='density')

ax = sns.distplot(df[df['average_montly_hours'] > 0]['average_montly_hours'], hist=True,

             kde=False, rug=False, bins= 10,

             hist_kws={"histtype": "step", 'facecolor':'r', "linewidth": 4,"alpha": 1, "color": sns.desaturate("indianred", .75)})

ax = sns.distplot(df[df['average_montly_hours'] > 0]['average_montly_hours'], hist=True,

             kde=False, rug=False, bins= 10,

             hist_kws={"histtype": "bar", 'facecolor':'b', "linewidth": 2,"alpha": 0.6, "color": 'b'})

#ax = plt.plot(df['average_montly_hours'])



#df['average_montly_hours'].plot(kind='density')

sns.jointplot('average_montly_hours','satisfaction_level',data=df,kind='hex') # SCATTER AND REG 
d = {}

for x in range(500):

    d[x] = np.array(df.iloc[x].iloc[0:4])

small = pd.DataFrame(list(d.values()))

small['Y/N'] = list(np.random.randint(0,2,len(small)))

small

sns.pairplot(small,hue='Y/N',palette='coolwarm', size=2,aspect=1)
fig = plt.figure(figsize=(10,12))

fig.add_subplot(211)

df['average_montly_hours'].plot(kind='hist')

fig.add_subplot(212)

df['average_montly_hours'].plot(kind='kde',color='red')
sns.rugplot(small[3])
# REGRESSION
sns.lmplot('average_montly_hours','number_project',data=df, hue='salary',palette='GnBu_d')

sns.lmplot('number_project','satisfaction_level', hue='left',data=df, palette='GnBu_d')

# col for separate grid

sns.lmplot('time_spend_company','satisfaction_level',data=df, row='left',

           col='salary', palette='GnBu_d', aspect=1, size=3)

# CATEGORICAL
sns.barplot(x= 'salary',y='time_spend_company', data=df,estimator=np.std)
sns.countplot(x='salary', hue='left',data=df)
sns.boxplot(x= 'salary',y='satisfaction_level', hue = 'left', data=df, palette='rainbow')

plt.legend(loc='center left',bbox_to_anchor=(1.0,0.5))
sns.violinplot(x= 'salary',y='satisfaction_level', hue = 'left', data=df, palette='Set1',split=True)
sns.stripplot(x='left',y='satisfaction_level', hue='salary', data=df, jitter=True, split=True)

#sns.swarmplot(x='left',y='satisfaction_level', hue='salary', data=df, split=True)
g = sns.factorplot(x="time_spend_company", y="salary",  hue="left",

                   data=df,

                 orient="h", size=5, aspect=1, palette="Set3",

                   kind="violin", cut=0, bw=.2)
#MATRIX PLOT
sns.heatmap(df.corr(), annot=True, cmap='coolwarm',linecolor='black', linewidths = 2)
fp = df.pivot_table(index='salary', columns = 'number_project', values = 'time_spend_company')
sns.heatmap(fp, annot=True, cmap='coolwarm', linecolor='black',linewidths=1)
sns.clustermap(fp, cmap='coolwarm', standard_scale = 1)
fig = plt.figure(figsize=(12,3))

fig.add_subplot(221)

sns.set_context('paper',font_scale=1.5)

sns.set_style('whitegrid') #ticks , 

sns.countplot(x='number_project', data=df)

sns.despine(left=True,bottom=True) # border ot not

fig.add_subplot(221)

sns.set_style('ticks') #ticks 

sns.lmplot(x='average_montly_hours', y ='satisfaction_level', data=df[:1000])
from plotly import __version__

print(__version__)