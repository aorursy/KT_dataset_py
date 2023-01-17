import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)
tips = sns.load_dataset("tips")
sns.catplot(x="day", y="total_bill", data=tips)
sns.catplot(x="day", y="total_bill", jitter=False, data=tips);
sns.catplot(x="day", y="total_bill", kind="swarm", data=tips);
sns.catplot(x="day", y="total_bill", hue="sex", kind="swarm", data=tips);
sns.catplot(x="total_bill", y="day", hue="time", kind="swarm", data=tips);
sns.catplot(x="day", y="total_bill", kind="box", data=tips);
sns.catplot(x="day", y="total_bill", hue="smoker", kind="box", data=tips);
tips["weekend"] = tips["day"].isin(["Sat", "Sun"])
sns.catplot(x="day", y="total_bill", hue="weekend",
            kind="box", dodge=False, data=tips);

sns.catplot(x="total_bill", y="day", hue="sex",kind="violin", data=tips);
sns.catplot(x="day", y="total_bill", hue="sex",kind="violin", split=True, data=tips);
sns.catplot(x="day", y="total_bill", hue="sex",
            kind="violin", inner="stick", split=True,
            palette="pastel", data=tips);
titanic = sns.load_dataset("titanic")
sns.catplot(x="sex", y="survived", hue="class", kind="bar", data=titanic);
sns.catplot(y="deck", hue="class", kind="count",
            palette="pastel", edgecolor=".6",
            data=titanic);
sns.catplot(x="sex", y="survived", hue="class", kind="point", data=titanic);
sns.catplot(x="class", y="survived", hue="sex",
            palette={"male": "g", "female": "m"},
            markers=["^", "o"], linestyles=["-", "--"],
            kind="point", data=titanic);
import numpy as np  # to generate random data
import pandas as pd
x = np.random.normal(size=100)
sns.distplot(x);
sns.distplot(x, kde=False, rug=True);
# When drawing histograms, the main choice you have is the number of bins to use and where to place them.
sns.distplot(x, bins=20, kde=False, rug=True);
sns.distplot(x, hist=False, rug=True);
sns.kdeplot(x, shade=True);
mean, cov = [0, 1], [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x", "y"])
sns.jointplot(x="x", y="y", data=df);
x, y = np.random.multivariate_normal(mean, cov, 1000).T
with sns.axes_style("white"):
    sns.jointplot(x=x, y=y, kind="hex", color="k");
# Kernel density estimation  / kind= "kde"
sns.jointplot(x="x", y="y", data=df, kind="kde");
f, ax = plt.subplots(figsize=(6, 6))
sns.kdeplot(df.x, df.y, ax=ax)
sns.rugplot(df.x, color="g", ax=ax)
sns.rugplot(df.y, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))
cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
sns.kdeplot(df.x, df.y, cmap=cmap, n_levels=60, shade=True);
iris = sns.load_dataset("iris")
sns.pairplot(iris);
sns.pairplot(iris, hue="species");
g = sns.PairGrid(iris)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels=6);
sns.regplot(x="total_bill", y="tip", data=tips);
sns.lmplot(x="total_bill", y="tip", data=tips);
sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips);
sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips,
           markers=["o", "x"], palette="Set1");
sns.lmplot(x="total_bill", y="tip", hue="smoker", col="time", data=tips);
sns.lmplot(x="total_bill", y="tip", hue="smoker",
           col="time", row="sex", data=tips);
sns.jointplot(x="total_bill", y="tip", data=tips, kind="reg");
sns.pairplot(tips, x_vars=["total_bill", "size"], y_vars=["tip"],
             height=5, aspect=.8, kind="reg");
# Like lmplot(), but unlike jointplot(), conditioning on an additional categorical variable is built into pairplot() using the hue parameter:

sns.pairplot(tips, x_vars=["total_bill", "size"], y_vars=["tip"],
             hue="smoker", height=5, aspect=.8, kind="reg");
