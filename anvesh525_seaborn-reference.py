# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
titanic = pd.read_csv("../input/titanic/train.csv")

iris = pd.read_csv("../input/iriscsv/Iris.csv")



uniform_data = np.random.rand(10,12)

data = pd.DataFrame({'x': np.arange(1,101), 'y': np.random.normal(0,4,100)})

print(titanic.columns, iris.columns)

f, ax = plt.subplots(figsize=(5,6))  # Create a figure and a one subplot

sns.set()                            # Set the matplotlib paramters

sns.set_style("whitegrid")           # Set the matplotlib paramters

sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8} )           # Set the matplotlib paramters

sns.axes_style("whitegrid")           # Set the matplotlib paramters      # Retuns a dict of params or use with with to temporarily set the style
g = sns.FacetGrid(titanic, col="Survived", row="Sex")

g = g.map(plt.hist, "Age")
sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=titanic)
sns.lmplot(x="SepalWidthCm", y="SepalLengthCm",  hue="Species", data=iris)
h = sns.PairGrid(iris)

h = h.map(plt.scatter)
sns.pairplot(iris)
i = sns.JointGrid(x="x", y="y", data=data)

i = i.plot(sns.regplot, sns.distplot)
sns.jointplot("SepalLengthCm", "SepalWidthCm", data= iris, kind='kde')
sns.stripplot(x="Species", y="PetalLengthCm", data=iris)

sns.swarmplot(x="Species", y="PetalLengthCm", data=iris)
sns.barplot(x="Sex", y="Survived", hue="Pclass", data=titanic)
sns.countplot(x="Parch", data=titanic, palette="Greens_d")
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=titanic, pallette={"male":"g", "female": "m"}, markers=["^","o"], linestyles=["-", "--"])
sns.boxplot(x="Survived", y="Age", hue="Sex", data=titanic)
sns.boxplot(data=iris, orient="h")
sns.violinplot(x="Age", y="Sex", hue="Survived", data=titanic)
ax = sns.regplot(x="SepalWidthCm", y="SepalLengthCm", data=iris)
plot = sns.distplot(data.y, kde=False, color="b") 
sns.heatmap(uniform_data, vmin=0, vmax=1)
g.despine(left=True)     # remove left spines 



g.set_ylabels("Survived")  # Set the labels of the y-axis 



g.set_xticklabels(rotation=45)  #  Set the tick labels for x



g.set_axis_labels("Survived", "Sex")  # Set the axis labels



h.set(xlim=(0,5), ylim=(0,5), xticks=[0,2.5,5], yticks=[0,2.5,5])    # Set the limit and ticks of the x-and y-axis
plt.title("A Title") # Add plot title

plt.ylabel("Survived") # Adjust the label of the y-axis

plt.xlabel("Sex") # Adjust the label of the x-axis

plt.ylim(0,100) # Adjust the limits of the y-axis

plt.xlim(0,10) # Adjust the limits of the x-axis

plt.setp(ax,yticks=[0,5]) # Adjust a plot property

plt.tight_layout() #Adjust subplot params
plt.show() # Show the plot

plt.savefig("foo.png") # Save the plot as a figure

plt.savefig("foo.png", transparent=True)  # Save transparent figure 



plt.cla() # Clear an axis

plt.clf() # Clear an entire 

plt.close() # Close a window

data = pd.DataFrame({'x': np.arange(0,500000,5000), 'y': [140, 242, 371, 523, 742, 1140, 1285, 1679, 2289, 2964, 3414, 3953, 4144, 5824, 7144, 7710, 8585, 10007, 10750, 11421]})
#data.plot()

f, ax = plt.subplots(figsize=(40,20))

ax = sns.lineplot(x="x", y="y", data=data)
datay= [54.6875,125,210.9375,300.7813,402.3438,515.625,656.25,839.8438,988.2813,1148.438,1285.156,1488.281,1570.313,1710.938,1898.438,2097.656,2257.813,2316.406,2597.656,2300.781,3421.875,3414.063,3585.938,4023.438,4171.875,4242.188,4210.938,4507.813,4675.781,4597.656,4371.094,4238.281,4644.531,4546.875,4769.531,4835.938,5136.719,5351.563,5347.656,5804.688,6835.938,8996.094,9277.344,8902.344,7128.906,7285.156,7113.281,6765.625,7566.406,7183.594,7394.531,7519.531,8609.375,9070.313,8332.031,8273.438,9050.781,9234.375,9343.75,9632.813,13542.97,10453.13,10550.78,12550.78,14855.47,11250,11414.06,11207.03,11312.5,14847.66,12878.91,16421.88,13203.13,13472.66,14886.72,15632.81,15968.75,16152.34,15894.53,13937.5,14519.53,13519.53,13492.19,14066.41,14058.59,14214.84,20460.94,17546.88,15726.56,16410.16,20875,19667.97,18324.22,18320.31,18140.63,18335.94,18968.75,23558.59,18125,18949.22]



data_r = [23.4375,31.25,46.875,105.4688,93.75,136.7188,167.9688,222.6563,250,285.1563,308.5938,332.0313,347.6563,378.9063,496.0938,433.5938,453.125,519.5313,507.8125,519.5313,589.8438,578.125,609.375,667.9688,710.9375,714.8438,894.5313,800.7813,789.0625,808.5938,812.5,988.2813,992.1875,914.0625,1027.344,1023.438,1187.5,1144.531,1082.031,1156.25,1230.469,1312.5,1226.563,1968.75,1355.469,1718.75,1402.344,1734.375,1421.875,2023.438,1628.906,1921.875,1894.531,2003.906,2691.406,2109.375,2464.844,2218.75,3328.125,2128.906,2859.375,2656.25,3828.125,2656.25,2449.219,2894.531,2925.781,3726.563,3085.938,4718.75,3082.031,3421.875,5386.719,4144.531,4113.281,4117.188,4437.5,4429.688,5019.531,4890.625]



data_w = [93.75,195.3125,187.5,375,367.1875,617.1875,761.7188,953.125,1167.969,1394.531,1582.031,1855.469,2007.813,2242.188,2390.625,2753.906,2761.719,3230.469,3414.063,3546.875,4468.75,4414.063,4554.688,4847.656,5453.125,5472.656,6750,6949.219,6531.25,6628.906,6890.625,7343.75,7589.844,8195.313,8351.563,8664.063,9562.5,10105.47,9671.875,10503.91,11472.66,11250,11511.72,14945.31,13582.03,13175.78,14035.16,14718.75,14941.41,18039.06,15781.25,16335.94,16867.19,17253.91,18980.47,18019.53,18578.13,18914.06,21933.59,21640.63,20003.91,23667.97,23414.06,23671.88,21328.13,22250,24476.56,23113.28,24238.28,24847.66,25507.81,24617.19,25949.22,29210.94,27691.41,26312.5,28105.47,30117.19,27695.31,29566.41]

#'applicationValuesCount': np.arange(0,400000,5000), 

data = pd.DataFrame({'timeForAdding_ms': data_w, 'timeForReading_ms': data_r })

f, ax = plt.subplots(figsize=(40,20))

ax = sns.lineplot(data=data)