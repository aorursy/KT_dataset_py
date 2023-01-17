import seaborn as sns

sns.set()

tips = sns.load_dataset("tips")

sns.relplot(x="total_bill", y="tip", col="time",

            hue="smoker", style="smoker", size="size",

            data=tips);
dots = sns.load_dataset("dots")

sns.relplot(x="time", y="firing_rate", col="align",

            hue="choice", size="coherence", style="choice",

            facet_kws=dict(sharex=False),

            kind="line", legend="full", data=dots);
fmri = sns.load_dataset("fmri")

sns.relplot(x="timepoint", y="signal", col="region",

            hue="event", style="event",

            kind="line", data=fmri);
sns.lmplot(x="total_bill", y="tip", col="time", hue="smoker",

           data=tips);
sns.catplot(x="day", y="total_bill", hue="smoker",

            kind="swarm", data=tips);
sns.catplot(x="day", y="total_bill", hue="smoker",

            kind="violin", split=True, data=tips);
sns.catplot(x="day", y="total_bill", hue="smoker",

            kind="bar", data=tips);
import matplotlib.pyplot as plt

f, axes = plt.subplots(1, 2, sharey=True, figsize=(6, 4))

sns.boxplot(x="day", y="tip", data=tips, ax=axes[0])

sns.scatterplot(x="total_bill", y="tip", hue="day", data=tips, ax=axes[1]);
sns.relplot(x="time", y="firing_rate", col="align",

            hue="choice", size="coherence", style="choice",

            height=4.5, aspect=2 / 3,

            facet_kws=dict(sharex=False),

            kind="line", legend="full", data=dots);
iris = sns.load_dataset("iris")

sns.jointplot(x="sepal_length", y="petal_length", data=iris);
sns.pairplot(data=iris, hue="species");
sns.set(style="ticks", palette="muted")

sns.relplot(x="total_bill", y="tip", col="time",

            hue="smoker", style="smoker", size="size",

            data=tips);
sns.relplot(x="total_bill", y="tip", col="time",

            hue="size", style="smoker", size="size",

            palette="YlGnBu", markers=["D", "o"], sizes=(10, 125),

            edgecolor=".2", linewidth=.5, alpha=.75,

            data=tips);
g = sns.catplot(x="total_bill", y="day", hue="time",

                height=3.5, aspect=1.5,

                kind="box", legend=False, data=tips);

g.add_legend(title="Meal")

g.set_axis_labels("Total bill ($)", "")

g.set(xlim=(0, 60), yticklabels=["Thursday", "Friday", "Saturday", "Sunday"])

g.despine(trim=True)

g.fig.set_size_inches(6.5, 3.5)

g.ax.set_xticks([5, 15, 25, 35, 45, 55], minor=True);

plt.setp(g.ax.get_yticklabels(), rotation=30);