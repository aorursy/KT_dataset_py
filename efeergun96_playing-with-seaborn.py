# this notebook is created to use as playground during my presentation about seaborn. 



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
# we will use a dataset from seaborn (https://github.com/mwaskom/seaborn-data)

my_data = sns.load_dataset("tips")

print(type(my_data))
my_data.head()
my_data.describe(include=["category","integer","float"])
sns.lmplot(x="total_bill", y="tip", data=my_data);  
# Bonus

#sns.set_style("white")  # darker with style

sns.lmplot(x="total_bill", y="tip", hue="sex", data=my_data);     
#plt.figure(figsize=(10,10))   # plt can be used anytime (on most of the sns plots)

sns.heatmap(my_data.corr(),annot=True, cmap="Wistia_r");  # ruin the cmap
another_data = sns.load_dataset("mpg") # only for the following reason.

sns.clustermap(another_data.corr(), cmap="YlGnBu_r",center=0, figsize=(12, 10));
sns.pairplot(data=my_data);
# Bonus

graph = sns.pairplot(data=my_data, hue="smoker", palette="husl")

# if someone is curious, y stands for y axis. [between 0.0 and 1.0]

graph.fig.suptitle("Your plot title",y=1);  # maybe let's try with x axis too?
plt.figure(figsize=(20,10))

sns.violinplot(data=my_data, x="day", y="tip");

plt.show()
# Bonus

plt.figure(figsize=(20,10))

sns.violinplot(data=my_data, x="day", y="tip", hue="sex", palette="nipy_spectral_r",inner='points'); # split=True 

#plt.grid()

plt.show()
# Even More Bonus with catplot

plt.figure

graph = sns.catplot(data=my_data, x="day", y="tip", hue="sex", palette="afmhot_r", kind='violin',height=8 , aspect=2) # kind = ["box","violin","swarm","point","boxen"]

plt.grid()
dist1 = np.random.normal(1, 0.9, 1000)

dist2 = np.random.normal(2, 1.1, 1000)



sns.distplot(dist1)

sns.distplot(dist2);
plt.figure(figsize=(10,10))

sns.kdeplot(my_data.total_bill,my_data.tip);

#plt.grid()
# Bonus

plt.figure(figsize=(10,10))

sns.kdeplot(my_data.total_bill,my_data.tip, shade=True, n_levels=24, cmap="Purples_r");    # change n_levels to 4

#plt.grid()
# Even more Bonus

sns.jointplot(my_data.total_bill, my_data.tip, kind='kde', height=10); #kind :  “scatter” , “reg” , “resid” , “kde” , “hex”

#plt.grid()
sns.set_style("darkgrid")  

# easy multi dimensional scatter plots

sns.relplot(x="total_bill", y="tip", hue="day", size="sex", style="smoker", sizes=(100, 400), alpha=.6, palette="gnuplot", height=10, data=my_data);  # 5 dimensions