import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', size=20)
matplotlib.rc('axes', titlesize=20)
matplotlib.rc('axes', labelsize=20)
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('legend', fontsize=20)
matplotlib.rc('figure', titlesize=20)
import seaborn as sns

sns.set_style("darkgrid")
%matplotlib inline
data = pd.read_csv("../input/daily-inmates-in-custody.csv")
data.sample(5)
data.info()
plt.figure(figsize=(20,7))
h = plt.hist(pd.to_numeric(data.AGE).dropna(), facecolor='g', alpha=0.75, bins=100)
plt.title("Distribution of Ages")
plt.xlabel("Age of Inmates")
plt.ylabel("Count")
def my_autopct(pct):
    return ('%.2f' % pct) if pct > 3 else ''
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
f, ax = plt.subplots(1,2, figsize=(15,7))
#sns.countplot(x='RACE', hue='GENDER', data=data, ax=ax[1][1], palette="Set2")
pie = ax[0].pie(list(data['GENDER'].value_counts()), 
                   labels=list(data.GENDER.unique())[1:],
                  autopct='%1.1f%%', shadow=True, startangle=90)
pie = ax[1].pie(list(data['RACE'].value_counts()), 
                   labels=list(data.RACE.unique())[1:],
                  autopct=my_autopct, shadow=True, startangle=90, colors=colors)
ax[0].set_title("GENDER DISTRIBUTION AMONG INMATES")
ax[1].set_title("RACE")
#ax[1][1].set_title("RACE - GENDER DISTRIBUTION")
plt.figure(figsize=(20,7))
sns.countplot(x='RACE', hue='GENDER', data=data, palette="Set2",
             order = data['RACE'].value_counts().index)
plt.ylabel("Number of Inmates")
plt.figure(figsize=(10,7))
sns.countplot(x='GENDER', hue='RACE', data=data, palette="Set2",
             order = data['GENDER'].value_counts().index)
plt.ylabel("Number of Inmates")
f, ax = plt.subplots(2,1, figsize=(7,15))
#sns.countplot(x='RACE', hue='GENDER', data=data, ax=ax[1][1], palette="Set2")
pie = ax[0].pie(list(data['BRADH'].value_counts()), 
                   labels=list(data.BRADH.unique()),
                  autopct='%1.1f%%', shadow=True, startangle=90)
sns.countplot(x='BRADH', hue='RACE', data=data, palette="Set2",
             order = data['BRADH'].value_counts().index, ax=ax[1])
ax[0].set_title("Distribution on the basis of GENDER of inmates under Mental Observation")
ax[1].set_xlabel("Inmates under Mental Observation? Y-Yes, N-No")
#ax[1].set_title("RACE")
plt.figure(figsize=(7,7))
explode = (0,0,0.1)
f, ax = plt.subplots(1,2, figsize=(15,7))
#sns.countplot(x='RACE', hue='GENDER', data=data, ax=ax[1][1], palette="Set2")
pie = ax[0].pie(list(data['CUSTODY_LEVEL'].value_counts()), 
                   labels=list(data.CUSTODY_LEVEL.unique())[1:],
                  autopct='%1.1f%%', shadow=True, startangle=90, explode=explode)
pie = ax[1].pie(list(data.SRG_FLG.value_counts()), 
                   labels=list(data.SRG_FLG.unique()),
                  autopct=my_autopct, shadow=True, startangle=90, colors=colors)
ax[0].set_title("% of detainees with MIN/MAX/MID level of Detention")
ax[1].set_title("Member of the Gang?")
plt.figure(figsize=(10,7))
sns.countplot(x='SRG_FLG', hue='RACE', data=data, palette="Set2",
             order = data['SRG_FLG'].value_counts().index)
plt.legend(loc="upper right")
plt.title("Affiliation of Gang by Race")
plt.xlabel("Gang Affiliations? Y-Yes, N-No")
