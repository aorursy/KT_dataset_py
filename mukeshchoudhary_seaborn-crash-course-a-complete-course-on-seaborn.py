# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib.pyplot as plt
tip= sns.load_dataset('tips')

tip.head()
sns.relplot(x= 'total_bill', y= 'tip', data= tip, color= 'b')
sns.relplot(x= 'total_bill', y= 'tip', data= tip, hue= 'smoker')
sns.relplot(x= 'total_bill', y= 'tip', data= tip, hue= 'smoker', style= 'smoker', palette='viridis')
sns.relplot(x= 'total_bill', y= 'tip', data= tip, hue= 'smoker', style= 'time' )
sns.relplot(x= 'total_bill', y= 'tip', hue= 'size', data= tip)
sns.relplot(x= 'total_bill', y= 'tip', size= 'size', data= tip)
sns.relplot(x= 'total_bill', y= 'tip', size= 'size', sizes= (15,200), data= tip)
df= pd.DataFrame(dict(time= np.arange(500), value= np.random.randn(500).cumsum()))

df.head()
sns.lineplot( x= 'time', y= 'value', data= df)
sns.relplot(x= 'time', y= 'value', kind= 'line', data= df)
df1= pd.DataFrame(np.random.randn(500, 2).cumsum(axis= 0), columns= ['x', 'y'])

sns.relplot(x= 'x', y= 'y', data= df1, sort= False, kind= 'line')
sns.lineplot(x= 'x', y= 'y', data= df1, sort= False)
fmri= sns.load_dataset('fmri')

fmri.head()
sns.relplot(x= 'timepoint', y= 'signal', kind= 'line', data= fmri, color= 'b')
sns.relplot(x= 'timepoint', y= 'signal', kind= 'line', data= fmri, ci= None, color= 'b')
sns.relplot(x= 'timepoint', y= 'signal', kind= 'line', data= fmri, ci='sd', color= 'b')
sns.relplot(x= 'timepoint', y= 'signal', data= fmri, kind= 'line', estimator= None)
sns.relplot(x= 'timepoint', y= 'signal', hue= 'event', data= fmri, kind= 'line')
sns.relplot(x= 'timepoint', y= 'signal', hue= 'event', data= fmri, kind= 'line', estimator= None)
sns.relplot(x= 'timepoint', y= 'signal', hue= 'region', style= 'event', data= fmri, kind= 'line')
sns.relplot(x= 'timepoint', y= 'signal', hue= 'region', style= 'event', data= fmri, kind= 'line', markers= True, dashes= False)
sns.relplot(x= 'timepoint', y= 'signal', hue= 'event', style= 'event', data= fmri, kind= 'line', markers= True, dashes= False)
dots=sns.load_dataset('dots')

dots.head()
sns.relplot(x= 'time', y= 'firing_rate', data= dots, kind= 'line', hue= 'coherence', style= 'choice')
palette= sns.cubehelix_palette(light= 0.6, n_colors= 6)

sns.relplot(x= 'time', y= 'firing_rate', data= dots, kind= 'line', hue= 'coherence', style= 'choice', palette= palette)
df2= pd.DataFrame(dict(time= pd.date_range('2019-1-1', periods= 500), value= np.random.randn(500).cumsum()))

g= sns.relplot(x= 'time', y= 'value', kind= 'line', data= df2)

g.fig.autofmt_xdate()
tips= sns.load_dataset('tips')

tips.tail()
sns.relplot(x= 'total_bill', y= 'tip', data= tips, hue= 'smoker', col= 'time', palette= 'viridis')
sns.relplot(x= 'total_bill', y= 'tip', data= tips, hue= 'smoker', row= 'sex', col= 'time', palette= 'viridis')
sns.relplot(x= 'total_bill', y= 'tip', data= tips, hue= 'smoker',row= 'size', col= 'time', palette= 'viridis')
fmri.tail()
sns.set(style= 'darkgrid')
sns.relplot(x= 'timepoint', y= 'signal', hue= 'subject', col= 'region', row= 'event', palette= 'viridis', height= 5,

           kind= 'line', estimator= None, data= fmri)
sns.relplot(x= 'timepoint', y= 'signal', hue= 'event', style= 'event', col= 'subject',

           col_wrap= 5, palette= 'viridis', height= 3, aspect= .75, linewidth= 2.5,

           kind= 'line', data= fmri.query(" region== 'frontal' "))
sns.relplot(x= 'timepoint', y= 'signal', hue= 'event', style= 'event', col= 'subject',

           col_wrap= 5, palette= 'viridis', height= 3, aspect= .75, linewidth= 2.5,

           kind= 'line', data= fmri.query(" region== 'parietal' "))
fmri.head()
sns.lineplot(x= 'timepoint', y= 'signal',hue= 'region', style= 'event', data= fmri, ci= 68, markers= True, err_style= 'bars')
sns.lineplot(x= 'timepoint', y= 'signal', hue= 'event', units= 'subject', lw= 1, estimator= None, data= fmri.query("region=='frontal'"))
dots.head()
sns.lineplot(x= 'time', y= 'firing_rate', data= dots, hue= 'coherence', style= 'choice')
sns.scatterplot(x= 'total_bill', y= 'tip', hue= 'smoker', size= 'size', data= tips)
sns.scatterplot(x= 'total_bill', y= 'tip', hue= 'smoker', size= 'size', data= tips, style= 'time')
iris= sns.load_dataset('iris')

iris.head()
sns.scatterplot(x= 'sepal_length', y= 'petal_length', data= iris)
sns.set( style= 'ticks', color_codes= True)
tips.head()
sns.catplot(x= 'day', y= 'total_bill', data= tips, jitter= True)
sns.catplot(x= 'day', y= 'total_bill', data= tips, jitter= False)
sns.catplot(x= 'day', y= 'total_bill', data= tips, kind= 'swarm')
sns.catplot(x= 'day', y= 'total_bill', kind= 'swarm', hue= 'sex', data= tips)
sns.catplot(x="day", y="total_bill", hue="smoker",

             aspect=.6,

            kind="swarm", data=tips);
sns.catplot(x="day", y="total_bill", hue="smoker",

            col="time", aspect=.6,

            kind="swarm", data=tips);
sns.catplot(x= 'size', y= 'total_bill', kind= 'swarm', data= tips.query("size != 3"))
sns.catplot( x= 'smoker', y= 'tip', data= tips)
sns.catplot( x= 'smoker', y= 'tip', order= ['No', 'Yes'] , data= tips)
sns.catplot(x="total_bill", y="day", hue="time", kind="swarm", data=tips)
sns.catplot(x= 'day', y= 'total_bill', kind= 'box', data= tips)
sns.catplot(x= 'day', y= 'total_bill', kind= 'box', data= tips, hue= 'smoker')
sns.catplot(x= 'day', y= 'total_bill', kind= 'box', data= tips, hue= 'smoker', dodge= False)
titanic= sns.load_dataset('titanic')
titanic.head() # we can also use facet here
g = sns.catplot(x="fare", y="survived", row="class",

                kind="box", orient="h", height=1.5, aspect=4,

                data=titanic.query("fare > 0"))

g.set(xscale="log");
sns.catplot(x= 'day', y= 'total_bill', kind= 'boxen', data= tips)

sns.catplot(x= 'day', y= 'total_bill', kind= 'boxen', data= tips, hue= 'smoker')
sns.catplot(x= 'day', y= 'total_bill', kind= 'boxen', data= tips, hue= 'smoker', dodge= False)
diamonds= sns.load_dataset('diamonds')

diamonds.head()
sns.catplot(x= 'color', y= 'price', kind= 'boxen', data= diamonds.sort_values("color"))
sns.catplot(x="total_bill", y="day", hue="sex",

            kind="violin", data=tips)
sns.catplot(x="total_bill", y="day", hue="sex",

            kind="violin", data=tips, split= True)
sns.catplot(x="total_bill", y="day", hue="sex",

            kind="violin", data=tips, split= True, inner= "stick")
sns.catplot(x="total_bill", y="day", hue="sex",

            kind="violin", data=tips, split= True, inner= "stick", palette= 'pastel')

g= sns.catplot( x= 'day', y= 'total_bill', kind= 'violin', inner= None, data= tips)

sns.swarmplot(x= 'day', y= 'total_bill', color= 'k', data= tips, ax= g.ax)
g= sns.catplot( x= 'day', y= 'total_bill', kind= 'violin', inner= None, data= tips)

sns.swarmplot(x= 'day', y= 'total_bill', size=3, color= 'k', data= tips, ax= g.ax)
titanic.tail()
sns.barplot(x= 'sex', y= 'survived', data= titanic)
sns.barplot(x= 'sex', y= 'survived',hue= 'class', data= titanic)
sns.catplot(x= 'deck', kind= 'count', data= titanic)
sns.catplot(x= 'deck', kind= 'count', data= titanic, palette= "ch:.25")
sns.catplot(x= 'deck', kind= 'count', data= titanic, palette= "pastel", hue= 'class')
sns.catplot(x= 'deck', kind= 'count', data= titanic, palette= "pastel", hue= 'class', edgecolor= "0.6")
f, ax = plt.subplots(figsize=(7, 3))

sns.countplot(x= 'deck', data= titanic, palette= "pastel", hue= 'class', edgecolor= "0.6")
titanic.head()
sns.catplot(x= 'sex', y= 'survived', kind= 'point', data= titanic, hue= 'class')
sns.catplot(x= 'class', y= 'survived', hue= 'sex',

           palette= {'male':'g', 'female':'m'},

           markers= ['^','o'], linestyles=['-','--'],

           kind= 'point', data= titanic)
sns.catplot(data= iris, orient= 'h', kind= 'box')
sns.violinplot(x=iris.species, y=iris.sepal_length);
sns.set()

x = np.random.normal(size=100)

sns.distplot(x);
sns.distplot(x, bins= 20);
x= np.random.randn(100)

sns.distplot(x, kde= True)
sns.distplot(x, bins= 30)
sns.distplot(x, kde= True, rug= True)
sns.distplot(x, kde= False, rug= False)
sns.distplot(x, kde= True, hist= False)
sns.kdeplot(x)
sns.kdeplot(x, shade= True)
sns.kdeplot(x, shade= True, bw= 1)
sns.kdeplot(x, shade= True, bw= 0.2)
sns.kdeplot(x, shade= True, bw= 1, cut=0)
sns.kdeplot(x, shade= True, bw= 1, cut= 5)
mean, cov = [0, 2], [(1, .5), (.5, 1)]

x, y = np.random.multivariate_normal(mean, cov, size=50).T

ax = sns.kdeplot(x, color= 'r', shade= True)
ax = sns.kdeplot(x, y)
ax = sns.kdeplot(x, y, color= 'r', shade= True)
ax = sns.kdeplot(x, y, n_levels=30, cmap="Purples_d")
ax = sns.kdeplot(x, vertical=True)
ax = sns.kdeplot(x, cut=0)
ax = sns.kdeplot(x, y, cbar=True)
tips.head()
x= tips['total_bill']

y= tips['tip']
sns.jointplot(x= x, y= y)
sns.jointplot(x= x, y= y, kind= 'hex')
sns.jointplot(x= x, y= y, kind= 'kde')
f, ax= plt.subplots(figsize= (6,6))

cmap= sns.cubehelix_palette(light=1, dark= 0, reverse= True, as_cmap= True)

sns.kdeplot(x, y, cmap= cmap, n_levels= 60, shade= True)
f, ax= plt.subplots(figsize= (6,6))

cmap= sns.cubehelix_palette(start= 0, rot= 0.4, gamma= 1.0, hue= 0.8,

                   light=0, dark= 1, reverse= False, as_cmap= True)

sns.kdeplot(x, y, cmap= cmap, n_levels= 60, shade= True)
f, ax= plt.subplots(figsize= (6,6))

cmap= sns.cubehelix_palette(start= 0, rot= 0.4, gamma= 1.0, hue= 0.8,

                   light=1, dark= 0, reverse= False, as_cmap= True)

sns.kdeplot(x, y, cmap= cmap, n_levels= 60, shade= True)
g= sns.jointplot(x= x, y= y, kind= 'kde', color='m')

g.plot_joint(plt.scatter, c= 'w', s= 30, linewidth= 1, marker= '+')

#g.ax_joint.collections[0].set_alpha(0)
g= sns.jointplot(x= x, y= y, kind= 'kde', color='m')

g.plot_joint(plt.scatter, c= 'w', s= 30, linewidth= 1, marker= '+')

g.ax_joint.collections[0].set_alpha(0)
iris.head()
sns.pairplot(iris)
g= sns.PairGrid(iris)

g.map_diag(sns.kdeplot)

g.map_offdiag(sns.kdeplot, n_levels= 10)
setosa = iris.loc[iris.species == "setosa"]

virginica = iris.loc[iris.species == "virginica"]

ax = sns.kdeplot(setosa.sepal_width, setosa.sepal_length,

                 cmap="Reds", shade=True, shade_lowest=False)

ax = sns.kdeplot(virginica.sepal_width, virginica.sepal_length,

                 cmap="Blues", shade=True, shade_lowest=False)
from scipy import stats

x = np.random.normal(0, 1, size=30)

bandwidth = 1.06 * x.std() * x.size ** (-1 / 5.)

support = np.linspace(-4, 4, 200)



kernels = []

for x_i in x:



    kernel = stats.norm(x_i, bandwidth).pdf(support)

    kernels.append(kernel)

    plt.plot(support, kernel, color="r")



sns.rugplot(x, color=".2", linewidth=3);
sns.kdeplot(x)

sns.kdeplot(x, bw=.2, label="bw: 0.2")

sns.kdeplot(x, bw=2, label="bw: 2")

plt.legend();
sns.set(color_codes=True)
tips.head()
sns.regplot(x="total_bill", y="tip", data=tips);
sns.lmplot(x="total_bill", y="tip", data=tips);
sns.lmplot(x="size", y="tip", data=tips);
sns.lmplot(x="size", y="tip", data=tips, x_jitter=.05);
sns.lmplot(x="size", y="tip", data=tips, x_estimator=np.mean);
anscombe = sns.load_dataset("anscombe")
sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'I'"),

           ci=None, scatter_kws={"s": 80});
sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'II'"),

           ci=None, scatter_kws={"s": 80});
sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'II'"),

           order=2, ci=None, scatter_kws={"s": 80});
sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'III'"),

           ci=None, scatter_kws={"s": 80});
sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'III'"),

           robust=True, ci=None, scatter_kws={"s": 80});
tips["big_tip"] = (tips.tip / tips.total_bill) > .15

sns.lmplot(x="total_bill", y="big_tip", data=tips,

           y_jitter=.03);

sns.lmplot(x="total_bill", y="big_tip", data=tips,

           logistic=True, y_jitter=.03);
sns.lmplot(x="total_bill", y="tip", data=tips,

           lowess=True);
sns.residplot(x="x", y="y", data=anscombe.query("dataset == 'I'"),

              scatter_kws={"s": 80});
sns.residplot(x="x", y="y", data=anscombe.query("dataset == 'II'"),

              scatter_kws={"s": 80});
sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips);

sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips,

           markers=["o", "x"], palette="Set1");
sns.lmplot(x="total_bill", y="tip", hue="smoker", col="time", data=tips);
f, ax = plt.subplots(figsize=(5, 6))

sns.regplot(x="total_bill", y="tip", data=tips, ax=ax);
sns.jointplot(x="total_bill", y="tip", data=tips, kind="reg");
sns.pairplot(tips, x_vars=["total_bill", "size"], y_vars=["tip"],

             height=5, aspect=.8, kind="reg")
sns.pairplot(tips, x_vars=["total_bill", "size"], y_vars=["tip"],

             hue="smoker", height=5, aspect=.8, kind="reg");

def sinplot(flip=1):

    x = np.linspace(0, 14, 100)

    for i in range(1, 7):

        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)
sinplot()
sns.set_style('dark')

sinplot()
sns.set_style("darkgrid")

sinplot()
sns.set_style('white')

sinplot()
sns.set_style("whitegrid")

sinplot()
sns.set_style("ticks")

sinplot()
sinplot()

sns.despine()
f, ax = plt.subplots()

sns.violinplot(data=iris)

sns.despine(offset=10, trim=True)
sns.set_style("whitegrid")

sns.boxplot(data=iris, palette="deep")

sns.despine(left=True)
f = plt.figure(figsize=(6, 6))

gs = f.add_gridspec(2, 2)



with sns.axes_style("darkgrid"):

    ax = f.add_subplot(gs[0, 0])

    sinplot()



with sns.axes_style("white"):

    ax = f.add_subplot(gs[0, 1])

    sinplot()



with sns.axes_style("ticks"):

    ax = f.add_subplot(gs[1, 0])

    sinplot()



with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[1, 1])

    sinplot()



f.tight_layout()
sns.set_context("paper")

sinplot()
sns.set_context("talk")

sinplot()
sns.set_context("poster")

sinplot()
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

sinplot()