import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%config InlineBackend.figure_format = 'svg'

%matplotlib inline
dice = np.random.randint(1,7,200) + np.random.randint(1,7,200)  # 2d6 Dices roll 200 times

dice
sns.distplot(dice)
sns.distplot(dice, kde = False, bins = 8, vertical = True, color = 'red')
tips = sns.load_dataset("tips") # Load Tips dataset

tips.head()
sns.jointplot(x = tips.total_bill, y = tips.tip, kind='scatter') # kind - kind of plot
sns.jointplot(tips.total_bill, tips.tip, color = '#a631e7', kind = 'reg', space = 0) # regression plot
iris = sns.load_dataset("iris") # Load iris dataset

iris.head()
sns.pairplot(iris, hue = 'species')
# try with tips data

palete = ['#E9473F', '#3FE1E9']

sns.pairplot(tips, hue = 'sex', markers = 'X',corner = True, palette = sns.color_palette(palete))
g = sns.FacetGrid(tips, col="time", row="smoker") # Only this line show us an empty graph

g = g.map(plt.hist, "total_bill")
g = sns.FacetGrid(tips, col  = 'time', hue = 'day')

g = (g.map(plt.scatter, "total_bill", "tip", edgecolor="w").add_legend())
sns.barplot(x = 'sex', y = 'total_bill', data = tips)
sns.barplot(x = tips.time, y = tips.tip, hue = tips.sex)
sns.barplot('size', y = 'total_bill', data = tips, palette = 'Blues_d')
palete = ['#001eff', '#f000ff']

plt.style.use("dark_background") # set black background

sns.countplot(x = 'sex', data = tips, facecolor = (0,0,0,0),

                                        linewidth=5,

                                         edgecolor=sns.color_palette(palete))
plt.style.available # All styles
plt.style.use("Solarize_Light2")

sns.boxplot(x = tips.total_bill)
plt.style.use('grayscale')

sns.boxplot(x="day", y="total_bill", data=tips)
plt.style.use("bmh")

sns.boxenplot(x="day", y="total_bill", hue="time", data=tips, linewidth=2.5)
sns.reset_orig() # reset style to original

sns.violinplot(x = 'total_bill', y = 'day', hue = 'sex', data = tips, palete = 'rainbow')
sns.violinplot(x="day", y="total_bill", hue="sex",

                    data=tips, palette="Set1", split=True,

                    scale="count", inner="quartile")
sns.swarmplot(x="time", y="tip", data=tips,

              order=["Dinner", "Lunch"],

              color = 'green').set_title('Christmas trees') # Use to set tittle
ax = sns.violinplot(x='day', y = 'total_bill', data = tips, inner = None, color = '#a27250')

ax = sns.swarmplot(x='day', y = 'total_bill',

                   data = tips, 

                   color = 'black',

                   marker="h").set_title('Cacao beans') # I know that my imagination is crazy
palette = ('#ea7643', '#a91b1b')

sns.pointplot(x="day", y="tip", hue="sex", data=tips,

              capsize=.2, markers=["D", "x"], palette = sns.color_palette(palette),

             linestyles=["-", "--"])