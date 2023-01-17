import os

import pandas as pd
tipdata = pd.read_csv('../input/tipping/tips.csv')
tipdata.head()
import seaborn as sns

sns.countplot(x="sex", data= tipdata )
import seaborn as sns

sns.countplot(x="smoker", data = tipdata)
sns.distplot(tipdata.tip)
import seaborn as sns

sns.countplot(x="time", data = tipdata)
import matplotlib.pyplot as plt



fig, ax = plt.subplots(figsize=(10,6))        

sns.scatterplot(x="tip", y="total_bill", hue="sex", data=tipdata, ax=ax)
sns.kdeplot(tipdata.tip, tipdata.total_bill)
sns.distplot(tipdata.total_bill)
fig, ax = plt.subplots(figsize=(7,4))  

sns.violinplot(x='tip', y='sex', data=tipdata)

sns.catplot(x="day", y="total_bill", kind="box", data=tipdata);
sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tipdata);