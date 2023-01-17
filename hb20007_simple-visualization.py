import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # 2D plotting library
import seaborn as sns # Data visualization library based on matplotlib

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv");
train.info(verbose=True, null_counts=True)
# verbose=False would not display data for each column
survived_sex = train[train["Survived"] == 1]["Sex"].value_counts()
print(survived_sex)
dead_sex = train[train["Survived"] == 0]["Sex"].value_counts()
print(dead_sex)
df = pd.DataFrame([survived_sex, dead_sex])
print(df)
df.index = ["Survived", "Dead"]
print(df)
myfont = {"family": "serif",
        "color":  "darkred",
        "weight": "normal",
        "size": 16,
        } # For plot titles

myfont2 = {"size": 16} # For matplotlib axis labels. The default font is too small.
plt.style.use("seaborn") # This line is optional. It sets matplotlib's style (colors) to a style called "seaborn", so it will match the chart we will do later with Seaborn

df.plot(kind="bar", stacked=True, figsize=(15, 8))
plt.title("Survival vs sex", fontdict=myfont) # fontdict is an optional parameter.
plt.ylabel("No. of ppl", fontdict=myfont2)
plt.show()
g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train,
                   size=6, aspect=2, kind="bar", palette="muted") # "aspect" is optional. Image size can be defined using "size" only.
# There are six variations of the default theme, called color palette: deep, muted, pastel, bright, dark, and colorblind.
g.fig.suptitle("Survival rate vs passenger class, by sex", fontdict=myfont)