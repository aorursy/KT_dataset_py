import numpy as np # Linear Algebra
import pandas as pd # Data Processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Basic Visualization
import seaborn as sns # Visualition
import re # Regular Expression

import os
print(os.listdir("../input"))
#Reading Comma Separated Values (.csv) file.
data = pd.read_csv("../input/2015.csv")

#Taking basic informations.
data.info()
data.describe()
#Taking first 5 samples from data.
data.head()
#Backing up the columns name to compare it with the modified columns name.
pre_names = [each for each in data.columns]

#Searching for gaps, other invalid characters and clearing them with Regular Expression library.
data.columns = [ re.sub("[ \(\)]", "", each).lower() for each in data.columns]

#Now let's look comparison with before and after.

lis = list(data.columns)
print("Before Names of Columns".center(94,"-"),end="\n\n")
for each in pre_names:
    print(each.ljust(29), end=" | ")
print("\n")
print("After Names of Columns".center(94,"-"),end="\n\n")
for each in lis:
    print(each.ljust(29), end=" | ")
#Line Plot - 1
data.healthlifeexpectancy.plot(kind = "Line", label = "healthlifeexpectancy", color = "r",
                              linewidth = 1, linestyle = "--", grid = True, alpha = 0.7,
                              figsize = (20,10))
data.economygdppercapita.plot(label = "economygdppercapita", color = "b",
                             linewidth = 1, linestyle = "-.", alpha = 0.7)
plt.legend(loc = "upper right", fontsize = "large")
plt.title("This is Line Plot - Relationsip Between Healt Life Expectancy and Economy GDP per Capita")
plt.xlabel("Happiness Rank", fontsize = "large")
plt.ylabel("Health Life Expectancy and Economy GDP per Capita", fontsize = "large")
plt.show()
#Line Plot - 2
data.generosity.plot(kind = "Line", label = "generosity", color = "g",
                    linewidth = 2, linestyle = "-.", grid = True,
                    figsize = (20,10))
plt.legend(loc = "upper right")
plt.title("This is a Line Plot - Generosity")
plt.xlabel("Happiness Rank")
plt.ylabel("Generosity")
plt.show()
#Scatter Plot - 1
data.plot(kind = "scatter", x = "happinessscore", y = "freedom", color = "g",
          alpha = 0.5, grid = True,s=80,
          figsize =(20,10))
plt.show()
#Scatter Plot - 2
ax = data.plot(kind = "scatter", x = "freedom", y = "generosity", color = "red",
          alpha = 0.5, grid = True,s=50,
          figsize =(20,10))
data.plot(kind = "scatter", x = "freedom", y = "healthlifeexpectancy", color = "blue",
          alpha = 0.5, grid = True,s=50,
          figsize =(20,10), ax=ax)
plt.show()
#Scatter Plot - 3
data.plot(kind = "scatter", x = "freedom", y = "healthlifeexpectancy", color = "blue",
          alpha = 0.5, grid = True,s=data['freedom']*650,
          figsize =(20,10))
plt.show()
#Box Plot -1

data.generosity.plot(kind = "box", grid = True, figsize = (10,10))
plt.title("This is a Box Plot")
plt.show()
#Area Plot - 1

data.happinessscore.plot(kind = "area", label = "happinessscore", color = "b",
                 linewidth = 1, linestyle = "--", grid = True,
                 alpha = 0.5, stacked=False, figsize = (20,10))
plt.show()
# Bar Plot

a = list(data['region'].unique())
b = []

for each in range(len(a)):
    x = data["region"] == a[each]
    k = list(data[x].happinessscore.describe())[1]
    b.append(k)
    if len(a[each])> 20:
        t = []
        for i in a[each].split():
            t.append(i[:3])
        a[each] = ".".join(t)

plt.figure(figsize=(20,10))
plt.bar(a,b)
plt.xlabel("Regions", fontsize = "large")
plt.ylabel("Average Happiness Score According to Regions", fontsize = "large")
plt.show()
#Data Correlation
data.corr()
#Heatmap Plot

f,ax = plt.subplots(figsize = (15,15))
sns.heatmap(data.corr(), annot = True, linewidth = .5, fmt = ".2f", ax=ax)
plt.show()
#Filtering with np.logical_and

x = data["happinessscore"]>7
y = data["happinessscore"]<7.2

data[np.logical_and(x, y)]
#This is just show a sample for using while loop.
i = 0
while data["happinessscore"][i]>6:
    i +=1

print("The happiness score value of {} countries is higher than 6.".format(i))