import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
#!pip uninstall matplotlib
sns.set(style = 'darkgrid')
tips = sns.load_dataset('tips')

tips.head()
sns.relplot(x= "total_bill", y = "tip", data=tips, color = "r", marker = '+', hue="sex")

plt.legend();
tips['smoker'].value_counts()
#!pip install seaborn

df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")

df.isnull().sum()
sns.relplot(x = 'total_bill', y = 'tip', data = tips,hue="time");

tips.time[:5]
sns.relplot(x = 'total_bill', y = 'tip', hue = 'size', data = tips);
sns.relplot(x = 'total_bill', y = 'tip', hue = 'size', data = tips);
sns.relplot(x = 'total_bill', y = 'tip', data = tips,hue = 'smoker', style = 'time', size = 'size');
sns.relplot(x = 'total_bill', y = 'tip', data = tips, size = 'size');
from numpy.random import randn
df = pd.DataFrame(dict(time = np.arange(500), value = randn(500).cumsum()))
df.head()
plt.figure(figsize=(15,160))

fig = sns.relplot(x = 'time', y = 'value', kind = 'line', data = df, sort = True)

sns.despine()
df = pd.DataFrame(randn(500, 2).cumsum(axis = 0), columns = ['time', 'value'])
df.head()
sns.relplot(x = 'time', y = 'value', kind = 'line', data = df, sort = True)
fmri = sns.load_dataset('fmri')

fmri.head()
sns.relplot(x = 'timepoint', y = 'signal', kind = 'line', data = fmri)
sns.relplot(x = 'timepoint', y = 'signal', kind = 'line', data = fmri, ci = 'sd')
sns.relplot(x = 'timepoint', y = 'signal', estimator = None, kind = 'line', data = fmri)
sns.relplot(x = 'timepoint', y = 'signal', hue = 'event', kind = 'line', data = fmri)
fmri.head()
sns.relplot(x = 'timepoint', y = 'signal', hue = 'region', style = 'event', kind = 'line', data = fmri)
sns.relplot(x = 'timepoint', y = 'signal', hue = 'region', style = 'event', kind = 'line', data = fmri, markers = True, dashes = False)
sns.relplot(x = 'timepoint', y = 'signal', hue = 'event', style = 'event', kind = 'line', data = fmri)
df = pd.DataFrame(dict(time = pd.date_range('2019-06-02', periods = 500), value = randn(500).cumsum()))
df.head()
g = sns.relplot(x = 'time', y = 'value', kind = 'line', data = df)

g.fig.autofmt_xdate()
tips.head()
sns.relplot(x = 'total_bill', y = 'tip', hue = 'smoker', col = 'time', data = tips)
sns.relplot(x = 'total_bill', y = 'tip', hue = 'smoker', col = 'size', data = tips)
sns.relplot(x = 'total_bill', y = 'tip', hue = 'smoker', col = 'size', data = tips, col_wrap=3, height=3)
sns.scatterplot(x = 'total_bill', y = 'tip', data = tips)
fmri.head()
sns.lineplot(x = 'timepoint', y  = 'signal', style = 'event', hue = 'region', data = fmri, markers = True, ci = 68, err_style='bars')
sns.scatterplot(x = 'total_bill', y = 'tip', data = tips, hue = 'smoker', size = 'size', style = 'time')
iris = sns.load_dataset('iris')
iris.head()
sns.scatterplot(x = 'sepal_length', y = 'petal_length', data = iris)
sns.scatterplot(x = iris['sepal_length'], y = iris['petal_length'])
tips.head()
titanic = sns.load_dataset('titanic')
titanic.head()
#catplot()
sns.catplot(x = 'day', y = 'total_bill', data = tips)
sns.catplot(y = 'day', x = 'total_bill', data = tips)
sns.catplot(x = 'day', y = 'total_bill', data = tips, jitter = False)
sns.catplot(x = 'day', y = 'tip', data = tips, kind = 'swarm', hue = 'size')
sns.catplot(x = 'smoker', y = 'tip', data = tips, order= ['No', 'Yes'])
tips.head()
sns.catplot(x = 'day', y = 'tip', kind = 'box', data = tips, hue = 'sex')
sns.catplot(x = 'day', y = 'total_bill', kind = 'box', data = tips, hue = 'sex', dodge = False)
diamonds = sns.load_dataset('diamonds')

diamonds.head()
sns.catplot(x = 'color', y = 'price', kind = 'boxen', data = diamonds.sort_values('color'))
sns.catplot(x = 'color', y = 'price', kind = 'boxen', data = diamonds.sort_values('color'))
sns.catplot(x = 'day', y = 'total_bill', kind = 'boxen', data = tips, dodge = False)
sns.catplot(x = 'total_bill', y = 'day', hue = 'sex', kind = 'violin', data = tips, split = True,)
g = sns.catplot(x = 'day', y = 'total_bill', kind = 'violin', inner = None, data = tips)

sns.swarmplot(x = 'day', y = 'total_bill', color = 'k', size = 3, data = tips, ax = g.ax)
titanic.head()
sns.catplot(x = 'deck', kind = 'count', palette = 'ch:0.95', data = titanic, hue = 'class')
x = randn(100)
sns.distplot(x, kde = True, hist = True, rug= False, bins= 30)
tips.head()
x = tips['total_bill']

y = tips['tip']
sns.jointplot(x = x, y=y)
sns.set()

sns.jointplot(x = x, y=y, kind = 'hex')
sns.jointplot(x = x, y = y, kind = 'kde')
f, ax = plt.subplots(figsize = (6,6))

cmap = sns.cubehelix_palette(as_cmap = True, dark = 0, light = 1, reverse= True)

sns.kdeplot(x, y, cmap = cmap, n_levels=60, shade=True)
g = sns.jointplot(x, y, kind = 'kde', color = 'r')

g.plot_joint(plt.scatter, c = 'w', s = 30, linewidth = 1, marker = '+')

g.ax_joint.collections[0].set_alpha(0)
sns.pairplot(iris)
g = sns.PairGrid(iris)

g.map_diag(sns.kdeplot)

g.map_offdiag(sns.kdeplot, n_levels = 10)
tips.head()
sns.lmplot(x = 'total_bill', y= 'tip', data = tips)
data = sns.load_dataset('anscombe')

data.head()
data['dataset'].value_counts()
fig, ax = plt.subplots(figsize = (8,4))

sns.regplot(x = 'total_bill', y = 'tip', data = tips, ax = ax);
sns.lmplot(x = 'total_bill', y= 'tip', data = tips)
sns.lmplot(x = 'total_bill', y = 'tip', data = tips, col = 'day', col_wrap=2, height = 4)
def sinplot():

    x = np.linspace(0, 14, 100)

    for i in range(1, 5):

        plt.plot(x, np.sin(x+i*0.5)*(7-i))
sinplot()
sns.set_style('ticks', {'axes.grid': True, 'xtick.direction': 'in'})

sinplot()

sns.despine(left = True, bottom= False)
sns.axes_style()
sns.set_style('darkgrid')
sns.set_context('talk', font_scale=1.5)

sinplot()
current_palettes = sns.color_palette()

sns.palplot(current_palettes)
sns.palplot(sns.color_palette('hls', 8))
!pip install --upgrade plotly