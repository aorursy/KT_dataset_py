import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
tips = pd.read_csv('../input/tips.csv')
tips.head()
sns.barplot(x = 'sex', y ='total_bill', data = tips)
sns.distplot(tips['total_bill'])
sns.distplot(tips['total_bill'],kde = False)
sns.jointplot(x = 'total_bill', y = 'tip', data = tips)
sns.jointplot(x = 'total_bill', y = 'tip', data = tips ,kind = 'hex')
sns.jointplot(x = 'total_bill', y = 'tip', data = tips ,kind = 'reg')
sns.jointplot(x = 'total_bill', y = 'tip', data = tips ,kind = 'kde')
sns.pairplot(tips)
sns.pairplot(tips ,hue ='sex', markers=["o", "s"])
sns.rugplot(tips['total_bill'])
sns.kdeplot(tips['total_bill'], shade=True)
sns.boxplot(x = 'day', y= 'total_bill', data = tips)
sns.boxplot(x = 'day', y= 'total_bill', data = tips, hue = 'sex')
sns.violinplot(x = 'day', y= 'total_bill', data = tips)
sns.violinplot(x = 'day', y= 'total_bill', data = tips, hue = 'sex', split  = True)
sns.stripplot(x = 'day', y = 'total_bill', data = tips)
sns.stripplot(x = 'day', y = 'total_bill', data = tips, jitter= True,hue = 'sex', dodge = True)
sns.swarmplot(x = 'day', y = 'total_bill', data = tips)
sns.factorplot(x = 'day', y = 'total_bill', kind = 'box', data = tips)
sns.heatmap(tips.corr())
g = sns.PairGrid(tips)
g.map_diag(sns.distplot)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
