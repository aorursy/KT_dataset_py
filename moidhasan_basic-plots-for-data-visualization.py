

import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()


# Read dataset
# Next, we'll load the Iris flower dataset, which is in the "../input/" directory
import pandas as pd
iris = pd.read_csv("../input/iris-dataset/Iris.csv", index_col=0)
# the iris dataset is now a Pandas DataFrame
#df = pd.read_csv('C:\\Users\\mhass14\\Downloads\\Iris.csv')
iris.head()
pokemon = pd.read_csv("../input/pokemon-dataset/Pokemon.csv",encoding = 'latin-1',index_col=0)
#pokemon.head()
tips = pd.read_csv("../input/tips-dataset/tips.csv")
tips.head()
# Recommended way for Scatterplot
sns.lmplot(x='PetalWidthCm', y='SepalWidthCm', data=iris)
sns.jointplot(x="PetalWidthCm", y="SepalWidthCm", data=iris)
# One piece of information missing in the plots above is what species each plant is
# We'll use seaborn's FacetGrid to color the scatterplot by species
sns.FacetGrid(iris, hue="Species", height=5) \
   .map(plt.scatter, "PetalWidthCm", "SepalWidthCm") \
   .add_legend()
#SeaBorn
#Here is some of the functionality that seaborn offers:

#We can examine relationships between multiple variables.
#We can show observations or aggregate statistics for categorical variables.
#We have Options for visualizing univariate or bivariate distributions and for comparing them between subsets of data.
#Automatic estimation and plotting of linear regression models for different kinds dependent variables

#Seaborn aims to make visualization a central part of exploring and understanding data.
#Its dataset-oriented plotting functions operate on dataframes and arrays.

#Uni-Variate: Bar(Nominal categorical data), Line(Ordinal categorical), Area Chart and Histogram(Interval Data)
#Bi-Variate: Scatter and Hex Plot, Stacked Line Bivariate Line Chart
#tips = pd.read_csv("/axp/buanalytics/csriskadv/dev/mhass14/visualization/tips.csv")
#tips.head(20)
tips.shape
tips[tips['time']=='Lunch']['day'].value_counts().plot.bar()
(tips[tips['time']=='Dinner']['day'].value_counts()/len(tips)).plot.bar()
#This tells us that people go out for dinner on mostly saturday and sunday.

tips['day'].value_counts().plot.bar()
(tips[tips['time']=='Lunch']['day'].value_counts()/len(tips)).plot.bar()
tips[tips['time']=='Lunch']['day'].value_counts().plot.bar()
tips[tips['smoker']=='Yes']['day'].value_counts().plot.bar()
tips[tips['smoker']=='No']['day'].value_counts().plot.bar()
tips[tips['time']=='Lunch']['size'].value_counts().sort_index().plot.bar()
tips[tips['time']=='Dinner']['size'].value_counts().sort_index().plot.bar()
tips[tips['time']=='Lunch']['day'].value_counts()
len(tips)
tips['day'].value_counts()
tips['day'].value_counts().plot.line()
tips['total_bill'].plot.hist()
#histograms have one major shortcoming (the reason for our 200$ caveat earlier).
#Because they break space up into even intervals, they don't deal very well with skewed data
tips.plot.hexbin(x='total_bill',y='tip',gridsize=20)
tips.plot.scatter(x='total_bill',y='tip')
tips['size'].plot.bar(stacked=True)
sns.countplot(tips['day'])
sns.kdeplot(tips.query('tip < 1').total_bill)
sns.kdeplot(tips.total_bill)
sns.boxplot(
    x='day',
    y='total_bill',
    data=tips)
sns.violinplot(
    x='day',
    y='total_bill',data=tips)
#We draw a faceted scatter plot with multiple semantic variables.

##This particular plot shows the relationship between five variables in the tips dataset.
##Three are numeric, and two are categorical.
##Two numeric variables (total_bill and tip) determined the position of each point on the axes,
#and the third (size) determined the size of each point.
##One categorical variable(col<->time) split the dataset onto two different axes (facets),
#and the other(hue<->smoker) determined the color(hue) and shape(style) of each point.

sns.relplot(x="total_bill", y="tip", col="time",
            hue="smoker", style="smoker", size="size",
            data=tips)
sns.FacetGrid(tips, hue="smoker", height=5) \
   .map(plt.scatter, "total_bill", "tip") \
   .add_legend()
sns.jointplot(x="tip", y="total_bill", data=tips)
# Recommended way for Scatterplot
sns.lmplot(x='total_bill', y='tip', data=tips)
#We draw a faceted scatter plot with multiple semantic variables.

##This particular plot shows the relationship between five variables in the tips dataset.
##Three are numeric, and two are categorical.
##Two numeric variables (total_bill and tip) determined the position of each point on the axes,
#and the third (size) determined the size of each point.
##One categorical variable(col<->time) split the dataset onto two different axes (facets),
#and the other(hue<->smoker) determined the color(hue) and shape(style) of each point.

sns.relplot(x="total_bill", y="tip", col="time",
            hue="smoker", style="smoker", size="size",
            data=tips)
sns.relplot(x="total_bill", y="tip", col="time",
            hue="smoker", size="size", style="smoker",
            facet_kws=dict(sharex=False),
            kind="line",legend="full", data=tips,height=10)
#Often we are interested in the average value of one variable as a function of other variables.
#Many seaborn functions can automatically perform the statistical estimation that is neccesary to answer these questions:

sns.relplot(x="total_bill", y="tip", col="time",
            hue="smoker", style="smoker",
            kind="line", data=tips)
#Statistical estimation in seaborn goes beyond descriptive statisitics. For example,
#it is also possible to enhance a scatterplot to include a linear regression model (and its uncertainty) using lmplot() 
sns.lmplot(x="total_bill", y="tip", col="time", hue="smoker",
           data=tips)
##Standard scatter and line plots visualize relationships between numerical variables,
#but many data analyses involve categorical variables.
#There are several specialized plot types in seaborn that are optimized for visualizing this kind of data.
#They can be accessed through catplot().
#Similar to relplot(), the idea of catplot() is that it exposes a common dataset-oriented API that generalizes over different representations of the relationship between one numeric variable and one (or more) categorical variables.
sns.catplot(x="day", y="total_bill", hue="smoker",
            kind="box", data=tips,col='time',height=5)
#Alternately, you could use kernel density estimation to represent the underlying distribution that the points are sampled
#from:
sns.catplot(x="day", y="total_bill", hue="smoker",
            kind="bar", data=tips,col='time')
t_s = tips[(tips['day']=='Thur') & (tips['smoker']=='Yes')]
t_ns = tips[(tips['day']=='Thur') & (tips['smoker']=='No')]
t_ns['tip'].describe()
t_s['tip'].describe()
#We could show the only mean value and its confidence interval 
sns.catplot(x="day", y="tip", hue="smoker",
            kind="box", data=tips)
tips.shape
tips.head()
import matplotlib.pyplot as plt
f, axes = plt.subplots(1, 2, sharey=True, figsize=(6, 4))
sns.boxplot(x="day", y="tip", data=tips, ax=axes[1])
sns.scatterplot(x="total_bill", y="tip", hue="day", data=tips, ax=axes[0])
axes[0]
iris.head()
sns.jointplot(x="SepalLengthCm", y="PetalLengthCm", data=iris)
#pairplot(), takes a broader view, showing all pairwise relationships and the marginal distributions,
#optionally conditioned on a categorical variable :
#By default, this function will create a grid of Axes such that each variable in data will by shared in the y-axis across a single row and in the x-axis across a single column.
#The diagonal Axes are treated differently,
#drawing a plot to show the univariate distribution of the data for the variable in that column.
sns.pairplot(data=iris, hue="Species", kind ='scatter',height=2)
g = sns.catplot(x="total_bill", y="day", hue="time",
                height=3.5, aspect=1.5,
                kind="box", legend=False, data=tips);
g.add_legend(title="Meal")
g.set_axis_labels("Total bill ($)", "")
g.set(xlim=(0, 60), yticklabels=["Thursday", "Friday", "Saturday", "Sunday"])
g.despine(trim=True)
g.fig.set_size_inches(6.5, 3.5)
g.ax.set_xticks([5, 15, 25, 35, 45, 55], minor=True);
plt.setp(g.ax.get_yticklabels(), rotation=30)
sns.distplot(tips['total_bill'], hist=True, rug=False, kde=False)
sns.distplot(tips['tip'], hist=True, rug=True, kde=True)
sns.lineplot(x="total_bill", y="tip",
             hue="smoker", style="smoker",
             data=tips,size='size')
##### 


sns.lmplot(x='Attack', y='Defense', data=pokemon)
#Seaborn does not have dedicated scatterplot, this functions is plotting and fitting a regression line.
sns.lmplot(x='total_bill', y='tip', data=tips,
           fit_reg=False, # No regression line
           hue='smoker')   # Color by evolution stage

# Tweak using Matplotlib
plt.ylim(0, None)
plt.xlim(0, None)
sns.boxplot(data=tips)
tips.describe()
# Set theme
sns.set_style('whitegrid')
 
# Violin plot
#They show the distribution (through the thickness of the violin) instead of only the summary statistics.
sns.violinplot(x='day', y='total_bill', data=tips)
plt.figure(figsize=(10,6))

sns.violinplot(x='day', y='total_bill', data=tips, inner=None)
sns.swarmplot(x='day', y='total_bill', data=tips,color='k', # Make points black
              alpha=0.7) # and slightly transparent)
cor = tips.corr()
# Heatmap
sns.heatmap(cor)
c = tips.corr
cor
# Density Plot
sns.kdeplot(tips.total_bill, tips.tip)
