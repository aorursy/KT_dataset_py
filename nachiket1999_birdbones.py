import numpy as np 

import seaborn as sns

import pandas as pd 







from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

from sklearn import datasets 



import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use("ggplot")


bird = pd.read_csv(

    "../input/birds-bones-and-living-habits/bird.csv", 

    dtype={"id": "str"}

).dropna(axis=0, how="any")



bird.shape




bird.describe()



size_of_each_group = bird.groupby("type").size().sort_values(ascending=False)



ax = size_of_each_group.plot(

    kind="bar", 

    color="#32CD32",

    figsize=((6, 4)),

    rot=0

)



ax.set_title("No. Of Specimens By Ecological Group", fontsize=10)

ax.set_xlabel("Ecologoical Groups")

ax.set_ylabel("No. Of Specimens")

for x, y in zip(np.arange(0, len(size_of_each_group)), size_of_each_group):

    ax.annotate("{:d}".format(y), xy=(x-(0.14 if len(str(y)) == 3 else 0.1), y-6), fontsize=10, color="#DC143C")
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler(with_mean=False)



bird_raw = bird.copy()



feature_columns = ['huml', 'humw', 'ulnal', 'ulnaw', 'feml', 'femw', 'tibl', 'tibw', 'tarl', 'tarw'] 



corr = bird_raw[feature_columns].corr()



co, ax = plt.subplots(figsize=(5, 5))



sns.heatmap(

    corr, 

    cmap=sns.light_palette("#00304e", as_cmap=True), 

    square=True, 

    cbar=False, 

    ax=ax, 

    annot=True, 

    annot_kws={"fontsize": 8}

)



co = ax.set_title("Correlation Matrix For Bone Sizes", fontsize=10)



#A correlation matrix is a table showing correlation coefficients between variables.

#Each cell in the table shows the correlation between two variables. 

#In this case, the variables involved are the bone lengths and widths(or diameters).

#Te corresponding value for the variables explains the level of correlation.

# i.e. 0 - lowest & 1 - highest level of correlation.
sns.catplot(x="humw", y="huml", data= bird);

plt.xlabel('Humerus Width')

plt.ylabel('Humerus Length')
sns.catplot(x="ulnaw", y="ulnal", data= bird);

plt.xlabel('Ulna Width')

plt.ylabel('Ulna Length')
sns.catplot(x="femw", y="feml", data= bird);

plt.xlabel('Femur Width')

plt.ylabel('Femur Length')
sns.catplot(x="tibw", y="tibl", data= bird);

plt.xlabel('Tibia Width')

plt.ylabel('Tibia Length')
sns.catplot(x="tarw", y="tarl", data= bird);

plt.xlabel('Tarsal Width')

plt.ylabel('Tarsal Length')