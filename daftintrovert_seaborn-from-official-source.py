import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")





tips = sns.load_dataset("tips")

sns.relplot(x="total_bill", y="tip", data=tips);
sns.relplot(x="total_bill", y="tip", hue="smoker", data=tips);
sns.relplot(x="total_bill", y="tip", hue="smoker", style="smoker",

            data=tips);
sns.relplot(x="total_bill", y="tip", hue="size", data=tips);
sns.relplot(x="total_bill", y="tip", hue="size", palette="ch:r=-.5,l=.75", data=tips);
sns.relplot(x="total_bill", y="tip", size="size", sizes=(15, 200), data=tips);
df = pd.DataFrame(dict(time=np.arange(500),

                       value=np.random.randn(500).cumsum()))

g = sns.relplot(x="time", y="value", kind="line",markers = True,data=df)

g.fig.autofmt_xdate()
dots = sns.load_dataset("dots").query("align == 'dots'")
dots.head()
dots.info()
dots.describe()
sns.relplot(x="time", y="firing_rate",

            hue="coherence", style="choice",

            kind="line", data=dots);
fmri = sns.load_dataset("fmri")
fmri.head()
fmri.info()
fmri.describe()
sns.pairplot(fmri)
fmri.corr()
sns.heatmap(fmri.corr())
sns.relplot(x="timepoint", y="signal", hue="subject",

            col="region", row="event", height=3,

            kind="line", estimator=None, data=fmri);
titanic = sns.load_dataset("titanic")

sns.catplot(x="sex", y="survived", hue="class", kind="bar", data=titanic);
sns.catplot(x="deck", kind="count", palette="ch:.25", data=titanic);
sns.catplot(y="deck", hue="class", kind="count",

            palette="pastel", edgecolor=".6",

            data=titanic);
sns.catplot(x="sex", y="survived", hue="class", kind="point", data=titanic);
sns.catplot(x="class", y="survived", hue="sex",

            palette={"male": "g", "female": "m"},

            markers=["^", "o"], linestyles=["-", "--"],

            kind="point", data=titanic);
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats
sns.set(color_codes=True)

x = np.random.normal(size=100)

sns.distplot(x);
sns.distplot(x, kde=False, rug=True);
sns.distplot(x, bins=20, kde=False, rug=True);
sns.distplot(x, hist=False, rug=True);
x = np.random.normal(0, 1, size=30)

bandwidth = 1.06 * x.std() * x.size ** (-1 / 5.)

support = np.linspace(-4, 4, 200)



kernels = []

for x_i in x:



    kernel = stats.norm(x_i, bandwidth).pdf(support)

    kernels.append(kernel)

    plt.plot(support, kernel, color="r")



sns.rugplot(x, color=".2", linewidth=3);
sns.kdeplot(x, shade=True);
sns.kdeplot(x)

sns.kdeplot(x, bw=.2, label="bw: 0.2")

sns.kdeplot(x, bw=2, label="bw: 2")

plt.legend();
sns.kdeplot(x, shade=True, cut=0)

sns.rugplot(x);
x = np.random.gamma(6, size=200)

sns.distplot(x, kde=False, fit=stats.gamma);
mean, cov = [0, 1], [(1, .5), (.5, 1)]

data = np.random.multivariate_normal(mean, cov, 200)

df = pd.DataFrame(data, columns=["x", "y"])

df.head()
sns.jointplot(x="x", y="y", data=df);
x, y = np.random.multivariate_normal(mean, cov, 1000).T

with sns.axes_style("white"):

    sns.jointplot(x=x, y=y, kind="hex", color="k");
sns.jointplot(x="x", y="y", data=df, kind="kde");
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(df.x, df.y, ax=ax)

sns.rugplot(df.x, color="g", ax=ax)

sns.rugplot(df.y, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(df.x, df.y, cmap=cmap, n_levels=60, shade=True);
g = sns.jointplot(x="x", y="y", data=df, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$X$", "$Y$");
iris = sns.load_dataset("iris")

sns.pairplot(iris);
g = sns.PairGrid(iris)

g.map_diag(sns.kdeplot)

g.map_offdiag(sns.kdeplot, n_levels=6);