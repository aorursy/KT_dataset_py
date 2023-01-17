import warnings

warnings.filterwarnings('ignore')



import numpy             as np

import pandas            as pd

import seaborn           as sns

import matplotlib.pyplot as plt



%matplotlib inline



plt.rcParams['figure.dpi'] = 100
# Path of the file to read

candy_filepath = '../input/data-for-datavis/candy.csv'



# Fill in the line below to read the file into a variable candy_data

candy_data = pd.read_csv(candy_filepath, index_col='id')



# Scatter plot showing the relationship between 'sugarpercent' and 'winpercent'

sns.scatterplot(x=candy_data['sugarpercent'], y=candy_data['winpercent'])
# Scatter plot showing the relationship between 'pricepercent', 'winpercent', and 'chocolate'

sns.scatterplot(x=candy_data['pricepercent'], y=candy_data['winpercent'], hue=candy_data['chocolate'])
# Color-coded scatter plot with regression lines

sns.lmplot(x='pricepercent', y='winpercent', hue='chocolate', data=candy_data)
# Scatter plot showing the relationship between 'chocolate' and 'winpercent'

sns.swarmplot(x=candy_data['chocolate'], y=candy_data['winpercent'])
tips = pd.read_csv('../input/seaborn-tips-dataset/tips.csv')



# Scatter plot with two variables (two-demensional)

sns.relplot(x='total_bill', y='tip', color='b', data=tips)
# Scatter plot with three variables (third dimension is a color of points)

sns.relplot(x='total_bill', y='tip', hue='smoker', palette='viridis', data=tips)
# Scatter plot with different marker styles

sns.relplot(x='total_bill', y='tip', hue='smoker', style='smoker',

            data=tips, palette = 'viridis')
# Scatter plot with four variables

sns.relplot(x='total_bill', y='tip', hue='smoker', style='time', data=tips, palette='viridis')
# Scatter plot with numeric hue semantic

sns.relplot(x='total_bill', y='tip', hue='size', data=tips)
# Scatter plot with size semantic as third variable 

sns.relplot(x='total_bill', y='tip', size='size', data=tips)
# Scatter plt with customized markers size

sns.relplot(x='total_bill', y='tip', size='size', sizes=(15, 200), data=tips)
# Path of the file to read

museum_filepath = '../input/data-for-datavis/museum_visitors.csv'



# Fill in the line below to read the file into a variable museum_data

museum_data = pd.read_csv(museum_filepath, index_col='Date', parse_dates=True)



# Line chart showing the number of visitors to each museum over time

plt.figure(figsize=(10, 5))

sns.lineplot(data=museum_data)
# Line plot showing the number of visitors to Avila Adobe over time

plt.figure(figsize=(10, 5))

sns.lineplot(data=museum_data['Avila Adobe'])
df = pd.DataFrame(dict(time=np.arange(500), value=np.random.randn(500).cumsum()))
# Line plot using sns.relplot() with kind='line'

g = sns.relplot(x='time', y='value', kind='line', data=df)

g.fig.autofmt_xdate()
# Line plot without sorting x values: sort=False

df = pd.DataFrame(np.random.randn(500, 2).cumsum(axis=0), columns=['x', 'y'])

sns.relplot(x='x', y='y', sort=False, kind='line', data=df);
fmri = pd.read_csv('../input/seaborn-fmri-dataset/fmri.csv')



# Line plot with default aggregation the multiple measurements at each x value 

# plotting the mean and the 95% confidence interval around the mean



sns.relplot(x='timepoint', y='signal', kind='line', data=fmri, color='blue')
# Line plot without visualization of Confidence Interval: ci=None

sns.relplot(x='timepoint', y='signal', ci=None, kind='line', color='blue', data=fmri)
# Lineplot with standard deviation instead of confidence interval

sns.relplot(x='timepoint', y='signal', kind='line', ci="sd", data=fmri)
# Line plot without aggregation: estimator=None

sns.relplot(x='timepoint', y='signal', estimator=None, kind='line', data=fmri)
# Line plot with aggregation for three variables (hue semantic)

sns.relplot(x='timepoint', y='signal', hue='event', kind='line', data=fmri)
# Line plot with aggregation wuth four variables (x, y, hue, style) without markers

sns.relplot(x="timepoint", y="signal", hue="region", style="event",

            kind="line", data=fmri)
# line plot with aggregation wuth four variables (x, y, hue, style) with markers



sns.relplot(x="timepoint", y="signal", hue="region", style="event",

            dashes=False, markers=True, kind="line", data=fmri);
# line plot with both hue and style used for one variable



sns.relplot(x="timepoint", y="signal", hue="event", style="event",

            kind="line", data=fmri);
# line plot with numeric hue variable



dots = pd.read_csv('../input/seaborn-dots-dataset/dots.csv').query("align == 'dots'")

sns.relplot(x="time", y="firing_rate",

            hue="coherence", style="choice",

            kind="line", data=dots);
# line plot with customized specific color values for each line



palette = sns.cubehelix_palette(light=0.6, n_colors=6)

sns.relplot(x="time", y="firing_rate",

            hue="coherence", style="choice",

            palette=palette,

            kind="line", data=dots);
# line plot with Data values



df = pd.DataFrame(dict(time=pd.date_range("2017-1-1", periods=500),

                       value=np.random.randn(500).cumsum()))

g = sns.relplot(x="time", y="value", kind="line", data=df)

g.fig.autofmt_xdate()
# line plots on one picture with subset of data devided by columns and rows



sns.relplot(x="timepoint", y="signal", hue="subject",

            col="region", row="event",palette = 'viridis', height=3,

            kind="line", estimator=None, data=fmri);
# line plots faceted on the columns and 'wraped' into rows



sns.relplot(x="timepoint", y="signal", hue="event", style="event",

            col="subject", col_wrap=5,palette = 'viridis',

            height=3, aspect=.75, linewidth=2.5,

            kind="line", data=fmri.query("region == 'frontal'"));
sns.set(style="ticks", color_codes=True)
# catplot with default parameters



sns.catplot(x="day", y="total_bill", data=tips, jitter = True)
# catplot with jitter=False



sns.catplot(x="day", y="total_bill", data=tips, jitter = False);
# catplot without points overlapping: kind='swarm'



sns.catplot(x="day", y="total_bill", kind="swarm", data=tips)
# catplot with third variable - hue semantic



sns.catplot(x="day", y="total_bill", hue="sex", kind="swarm", data=tips);
sns.catplot(x="size", y="total_bill", kind="swarm",

            data=tips.query("size != 3"));
# catplot with customized values order



sns.catplot(x="smoker", y="tip", order=["No", "Yes"], data=tips);
# catplot with horizontal axis of categorical data



sns.catplot(x="total_bill", y="day", hue="time", kind="swarm", data=tips);
# boxplot



sns.catplot(x="day", y="total_bill", kind="box", data=tips);
# boxplot with third variable by hue semantic



sns.catplot(x="day", y="total_bill", hue="smoker", kind="box", data=tips);
# boxplot without 'dodging': dodge=False



tips["weekend"] = tips["day"].isin(["Sat", "Sun"])

sns.catplot(x="day", y="total_bill", hue="weekend",

            kind="box", dodge=False, data=tips);
# boxplot with 'boxen' style



diamonds = pd.read_csv('../input/diamonds/diamonds.csv')

sns.catplot(x="color", y="price", kind="boxen",

            data=diamonds.sort_values("color"));
sns.catplot(x="total_bill", y="day", hue="sex",

            kind="violin", data=tips);
# violinplot with 'splited' violins



sns.catplot(x="day", y="total_bill", hue="sex",

            kind="violin", split=True, data=tips);
sns.catplot(x="day", y="total_bill", hue="sex",

            kind="violin", inner="stick", split=True,

            palette="pastel", data=tips);
# violinplot combined with swarmplot



g = sns.catplot(x="day", y="total_bill", kind="violin", inner=None, data=tips)

sns.swarmplot(x="day", y="total_bill", color="k", size=3, data=tips, ax=g.ax);
# Path of the file to read

ign_filepath = '../input/data-for-datavis/ign_scores.csv'



# Fill in the line below to read the file into a variable ign_data

ign_data = pd.read_csv(ign_filepath, index_col="Platform")



# Bar chart showing average score for racing games by platform

plt.figure(figsize=(15, 4)) # Your code here

sns.barplot(x=ign_data.index, y=ign_data['Racing'])
titanic = pd.read_csv('../input/python-seaborn-datas/titanic.csv')

sns.catplot(x="Sex", y="Survived", hue="Pclass", kind="bar", data=titanic);
sns.catplot(x="Survived", kind="count", palette="ch:.25", data=titanic);
sns.catplot(y="Survived", hue="Pclass", kind="count",

            palette="pastel", edgecolor=".6",

            data=titanic);
plt.figure(figsize=[10,5])

sns.countplot(x = 'chocolate', hue = 'hard', data = candy_data)

plt.xticks(rotation = 20);
# Heatmap showing average game score by platform and genre

plt.figure(figsize=(14, 7))

sns.heatmap(data=ign_data, annot=True)
sns.catplot(x="Sex", y="Survived", hue="Pclass", kind="point", data=titanic);
sns.catplot(x="Pclass", y="Survived", hue="Sex",

            palette={"male": "g", "female": "m"},

            markers=["^", "o"], linestyles=["-", "--"],

            kind="point", data=titanic);
iris = pd.read_csv('../input/seaborn-iris-dataset/iris.csv')

sns.catplot(data=iris, orient="h", kind="box");
sns.violinplot(x=iris.species, y=iris.sepal_length);
g = sns.catplot(x="Fare", y="Survived", row="Pclass",

                kind="box", orient="h", height=1.5, aspect=4,

                data=titanic.query("Fare > 0"))

g.set(xscale="log");
from scipy import stats

sns.set(color_codes=True)
# Paths of the files to read

cancer_b_filepath = '../input/data-for-datavis/cancer_b.csv'

cancer_m_filepath = '../input/data-for-datavis/cancer_m.csv'



# Fill in the line below to read the (benign) file into a variable cancer_b_data

cancer_b_data = pd.read_csv(cancer_b_filepath, index_col='Id')



# Fill in the line below to read the (malignant) file into a variable cancer_m_data

cancer_m_data = pd.read_csv(cancer_m_filepath, index_col='Id')
# Histograms for benign and maligant tumors

sns.distplot(a=cancer_m_data['Area (mean)'], kde=False)

sns.distplot(a=cancer_b_data['Area (mean)'], kde=False)
# KDE plots for benign and malignant tumors

sns.kdeplot(data=cancer_b_data['Radius (worst)'], shade=True)

sns.kdeplot(data=cancer_m_data['Radius (worst)'], shade=True)
x = np.random.normal(size=100)

sns.distplot(x);
# displot as Histogram



sns.distplot(x, kde=False, rug=True);
sns.distplot(x, bins=20, kde=False, rug=True);
sns.distplot(x, hist=False, rug=True);
sns.kdeplot(x, shade=True);
# The bandwidth (bw) parameter of the KDE controls how tightly the estimation is fit to the data,

# much like the bin size in a histogram. 

# The default behavior tries to guess a good value using a common reference rule, 

# but it may be helpful to try larger or smaller values



sns.kdeplot(x)

sns.kdeplot(x, bw=.2, label="bw: 0.2")

sns.kdeplot(x, bw=2, label="bw: 2")

plt.legend();
sns.kdeplot(x, shade=True, cut=0)

sns.rugplot(x);
mean, cov = [0, 1], [(1, .5), (.5, 1)]

data = np.random.multivariate_normal(mean, cov, 200)

df = pd.DataFrame(data, columns=["x", "y"])
# jointplot for sctterplots - visualizing a bivariate distribution



sns.jointplot(x="x", y="y", data=df);
# jointplot 'hexbin' plot - a bivariate analogue of a histogram - 



x, y = np.random.multivariate_normal(mean, cov, 1000).T

with sns.axes_style("white"):

    sns.jointplot(x=x, y=y, kind="hex", color="k");
# jointplot as KDE



sns.jointplot(x="x", y="y", data=df, kind="kde");
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(df.x, df.y, ax=ax)

sns.rugplot(df.x, color="g", ax=ax)

sns.rugplot(df.y, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(df.x, df.y, cmap=cmap, n_levels=60, shade=True);
# customizing JointGrid object



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
# lmplot with binary y variable



tips["big_tip"] = (tips.tip / tips.total_bill) > .15

sns.lmplot(x="total_bill", y="big_tip", data=tips,

           y_jitter=.03);
# lmplot with Loistic regression (in case of binary y data)



sns.lmplot(x="total_bill", y="big_tip", data=tips,

           logistic=True, y_jitter=.03);
sns.lmplot(x="total_bill", y="tip", data=tips,

           lowess=True);
sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips);
sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips,

           markers=["o", "x"], palette="Set1");
sns.lmplot(x="total_bill", y="tip", hue="smoker",

           col="time", row="sex", data=tips);
g = sns.FacetGrid(tips, col="sex", hue="smoker")

g.map(plt.scatter, "total_bill", "tip", alpha=.7)

g.add_legend();
ordered_days = tips.day.value_counts().index



g = sns.FacetGrid(tips, row="day", row_order=ordered_days,

                  height=1.7, aspect=4,)



g.map(sns.distplot, "total_bill", hist=False, rug=True);
# Change the style of the figure to the "dark" theme

sns.set_style('dark')

plt.figure(figsize=(10, 4))

sns.lineplot(data=museum_data['Avila Adobe'])