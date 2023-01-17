import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")

%matplotlib inline



import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 100
tips = pd.read_csv('../input/seaborn-tips-dataset/tips.csv')

sns.relplot(x="total_bill", y="tip",color = 'b', data=tips);
sns.relplot(x="total_bill", y="tip", hue="smoker",palette = 'viridis', data=tips);
sns.relplot(x="total_bill", y="tip", hue="smoker", style="smoker",

            data=tips,palette = 'viridis');
sns.relplot(x="total_bill", y="tip", hue="smoker", style="time", data=tips,palette = 'viridis');
sns.relplot(x="total_bill", y="tip", hue="size", data=tips);
sns.relplot(x="total_bill", y="tip", size="size", data=tips);
sns.relplot(x="total_bill", y="tip", size="size", sizes=(15, 200), data=tips);
df = pd.DataFrame(dict(time=np.arange(500),

                       value=np.random.randn(500).cumsum()))



g = sns.relplot(x="time", y="value", kind="line", data=df)



g.fig.autofmt_xdate()
df = pd.DataFrame(np.random.randn(500, 2).cumsum(axis=0), columns=["x", "y"])

sns.relplot(x="x", y="y", sort=False, kind="line", data=df);
fmri = pd.read_csv('../input/seaborn-visual-data/fmri.csv')

fmri.head()
sns.relplot(x="timepoint", y="signal", kind="line", data=fmri,color = 'blue');
sns.relplot(x="timepoint", y="signal", ci=None, kind="line",color='blue', data=fmri);
sns.relplot(x="timepoint", y="signal", kind="line", ci="sd", data=fmri);
sns.relplot(x="timepoint", y="signal", estimator=None, kind="line", data=fmri);
sns.relplot(x="timepoint", y="signal", hue="event", kind="line", data=fmri);
sns.relplot(x="timepoint", y="signal", hue="region", style="event",

            kind="line", data=fmri);
sns.relplot(x="timepoint", y="signal", hue="region", style="event",

            dashes=False, markers=True, kind="line", data=fmri);
sns.relplot(x="timepoint", y="signal", hue="event", style="event",

            kind="line", data=fmri);
dots = pd.read_csv('../input/dots-seaborn/dots.csv').query("align == 'dots'")

sns.relplot(x="time", y="firing_rate",

            hue="coherence", style="choice",

            kind="line", data=dots);
palette = sns.cubehelix_palette(light=0.6, n_colors=6)

sns.relplot(x="time", y="firing_rate",

            hue="coherence", style="choice",

            palette=palette,

            kind="line", data=dots);
df = pd.DataFrame(dict(time=pd.date_range("2017-1-1", periods=500),

                       value=np.random.randn(500).cumsum()))

g = sns.relplot(x="time", y="value", kind="line", data=df)

g.fig.autofmt_xdate()
sns.relplot(x="total_bill", y="tip", hue="smoker",

            col="time", data=tips,palette='viridis');
sns.relplot(x="timepoint", y="signal", hue="subject",

            col="region", row="event",palette = 'viridis', height=3,

            kind="line", estimator=None, data=fmri);
sns.relplot(x="timepoint", y="signal", hue="event", style="event",

            col="subject", col_wrap=5,palette = 'viridis',

            height=3, aspect=.75, linewidth=2.5,

            kind="line", data=fmri.query("region == 'frontal'"));
sns.set(style="ticks", color_codes=True)
sns.catplot(x="day", y="total_bill", data=tips, jitter = True);
sns.catplot(x="day", y="total_bill", data=tips, jitter = False);
sns.catplot(x="day", y="total_bill", kind="swarm", data=tips);
sns.catplot(x="day", y="total_bill", hue="sex", kind="swarm", data=tips);
sns.catplot(x="size", y="total_bill", kind="swarm",

            data=tips.query("size != 3"));
sns.catplot(x="smoker", y="tip", order=["No", "Yes"], data=tips);
sns.catplot(x="total_bill", y="day", hue="time", kind="swarm", data=tips);
sns.catplot(x="day", y="total_bill", kind="box", data=tips);
sns.catplot(x="day", y="total_bill", hue="smoker", kind="box", data=tips);
tips["weekend"] = tips["day"].isin(["Sat", "Sun"])

sns.catplot(x="day", y="total_bill", hue="weekend",

            kind="box", dodge=False, data=tips);
diamonds = pd.read_csv('../input/diamonds-seaborn/diamonds.csv')

sns.catplot(x="color", y="price", kind="boxen",

            data=diamonds.sort_values("color"));
sns.catplot(x="total_bill", y="day", hue="sex",

            kind="violin", data=tips);
sns.catplot(x="day", y="total_bill", hue="sex",

            kind="violin", split=True, data=tips);
sns.catplot(x="day", y="total_bill", hue="sex",

            kind="violin", inner="stick", split=True,

            palette="pastel", data=tips);
g = sns.catplot(x="day", y="total_bill", kind="violin", inner=None, data=tips)

sns.swarmplot(x="day", y="total_bill", color="k", size=3, data=tips, ax=g.ax);
titanic = pd.read_csv('../input/titanic-seaborn/titanic.csv')

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
iris = pd.read_csv('../input/iris-seaborn/iris.csv')

sns.catplot(data=iris, orient="h", kind="box");
sns.violinplot(x=iris.species, y=iris.sepal_length);
f, ax = plt.subplots(figsize=(7, 3))

sns.countplot(y="deck", data=titanic, color="c");
sns.catplot(x="day", y="total_bill", hue="smoker",

            col="time", aspect=.6,

            kind="swarm", data=tips);
g = sns.catplot(x="fare", y="survived", row="class",

                kind="box", orient="h", height=1.5, aspect=4,

                data=titanic.query("fare > 0"))

g.set(xscale="log");
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
from scipy.integrate import trapz

density = np.sum(kernels, axis=0)

density /= trapz(density, support)

plt.plot(support, density);
sns.kdeplot(x, shade=True);
sns.kdeplot(x)

sns.kdeplot(x, bw=.2, label="bw: 0.2")

sns.kdeplot(x, bw=2, label="bw: 2")

plt.legend();
sns.kdeplot(x, shade=True, cut=0)

sns.rugplot(x);
mean, cov = [0, 1], [(1, .5), (.5, 1)]

data = np.random.multivariate_normal(mean, cov, 200)

df = pd.DataFrame(data, columns=["x", "y"])
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
sns.pairplot(iris);
sns.pairplot(iris, hue="species");
g = sns.PairGrid(iris)

g.map_diag(sns.kdeplot)

g.map_offdiag(sns.kdeplot, n_levels=6);
sns.regplot(x="total_bill", y="tip", data=tips);
sns.lmplot(x="size", y="tip", data=tips);
sns.lmplot(x="size", y="tip", data=tips, x_jitter=.05);
sns.lmplot(x="size", y="tip", data=tips, x_estimator=np.mean);
anscombe = pd.read_csv('../input/anscombe-seaborn/anscombe.csv')

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
sns.lmplot(x="total_bill", y="tip", hue="smoker",

           col="time", row="sex", data=tips);
f, ax = plt.subplots(figsize=(5, 6))

sns.regplot(x="total_bill", y="tip", data=tips, ax=ax);
sns.lmplot(x="total_bill", y="tip", col="day", data=tips,

           col_wrap=2, height=3);
sns.lmplot(x="total_bill", y="tip", col="day", data=tips,

           aspect=.5);
sns.set(style="ticks")
g = sns.FacetGrid(tips, col="time")
g = sns.FacetGrid(tips, col="time")

g.map(plt.hist, "tip");
g = sns.FacetGrid(tips, col="sex", hue="smoker")

g.map(plt.scatter, "total_bill", "tip", alpha=.7)

g.add_legend();
g = sns.FacetGrid(tips, row="smoker", col="time", margin_titles=True)

g.map(sns.regplot, "size", "total_bill", color=".3", fit_reg=False, x_jitter=.1);
g = sns.FacetGrid(tips, col="day", height=4, aspect=.5)

g.map(sns.barplot, "sex", "total_bill");
ordered_days = tips.day.value_counts().index



g = sns.FacetGrid(tips, row="day", row_order=ordered_days,

                  height=1.7, aspect=4,)



g.map(sns.distplot, "total_bill", hist=False, rug=True);
pal = dict(Lunch="seagreen", Dinner="gray")

g = sns.FacetGrid(tips, hue="time", palette=pal, height=5)

g.map(plt.scatter, "total_bill", "tip", s=50, alpha=.7, linewidth=.5, edgecolor="white")

g.add_legend();
g = sns.FacetGrid(tips, hue="sex", palette="Set1", height=5, hue_kws={"marker": ["^", "v"]})

g.map(plt.scatter, "total_bill", "tip", s=100, linewidth=.5, edgecolor="white")

g.add_legend();
attend = pd.read_csv('../input/attention-seaborn/attention.csv').query("subject <= 12")

g = sns.FacetGrid(attend, col="subject", col_wrap=4, height=2, ylim=(0, 10))

g.map(sns.pointplot, "solutions", "score", order=[1, 2, 3], color=".3", ci=None);
with sns.axes_style("white"):

    g = sns.FacetGrid(tips, row="sex", col="smoker", margin_titles=True, height=2.5)

g.map(plt.scatter, "total_bill", "tip", color="#334488", edgecolor="white", lw=.5);

g.set_axis_labels("Total bill (US Dollars)", "Tip");

g.set(xticks=[10, 30, 50], yticks=[2, 6, 10]);

g.fig.subplots_adjust(wspace=.02, hspace=.02);
g = sns.FacetGrid(tips, col="smoker", margin_titles=True, height=4)

g.map(plt.scatter, "total_bill", "tip", color="#338844", edgecolor="white", s=50, lw=1)

for ax in g.axes.flat:

    ax.plot((0, 50), (0, .2 * 50), c=".2", ls="--")

g.set(xlim=(0, 60), ylim=(0, 14));
from scipy import stats

def quantile_plot(x, **kwargs):

    qntls, xr = stats.probplot(x, fit=False)

    plt.scatter(xr, qntls, **kwargs)



g = sns.FacetGrid(tips, col="sex", height=4)

g.map(quantile_plot, "total_bill");
def qqplot(x, y, **kwargs):

    _, xr = stats.probplot(x, fit=False)

    _, yr = stats.probplot(y, fit=False)

    plt.scatter(xr, yr, **kwargs)



g = sns.FacetGrid(tips, col="smoker", height=4)

g.map(qqplot, "total_bill", "tip");
g = sns.FacetGrid(tips, hue="time", col="sex", height=4)

g.map(qqplot, "total_bill", "tip")

g.add_legend();
g = sns.FacetGrid(tips, hue="time", col="sex", height=4,

                  hue_kws={"marker": ["s", "D"]})

g.map(qqplot, "total_bill", "tip", s=40, edgecolor="w")

g.add_legend();
def hexbin(x, y, color, **kwargs):

    cmap = sns.light_palette(color, as_cmap=True)

    plt.hexbin(x, y, gridsize=15, cmap=cmap, **kwargs)



with sns.axes_style("dark"):

    g = sns.FacetGrid(tips, hue="time", col="time", height=4)

g.map(hexbin, "total_bill", "tip", extent=[0, 50, 0, 10]);
g = sns.PairGrid(iris, hue="species")

g.map_diag(plt.hist)

g.map_offdiag(plt.scatter)

g.add_legend();
g = sns.PairGrid(iris, vars=["sepal_length", "sepal_width"], hue="species")

g.map(plt.scatter);
g = sns.PairGrid(iris)

g.map_upper(plt.scatter)

g.map_lower(sns.kdeplot)

g.map_diag(sns.kdeplot, lw=3, legend=False);
g = sns.pairplot(iris, hue="species", palette="Set2", diag_kind="kde", height=2.5)