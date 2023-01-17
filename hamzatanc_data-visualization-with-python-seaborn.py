import pandas as pd # It is used to perform data manipulation and analysis. 

import numpy as np # It is used to make scientific calculations.

import matplotlib.pyplot as plt

import seaborn as sns

from numpy import median

import missingno as msno # Used to visualize missing values

mtcars_data = pd.read_csv("../input/mtcars-data/mtcars.csv")
mtcars_data.head()
mtcars_data = mtcars_data.rename(columns = {"Unnamed: 0" : "car_model"})

mtcars_data.head()
sns.lineplot(data = mtcars_data[["mpg","cyl","qsec"]]);
sns.lineplot(data = mtcars_data[["mpg","cyl","qsec"]],markers = True);
plt.title(" - mpg - cyl - qsec in mtcars - ")

sns.lineplot(data = mtcars_data[["mpg","cyl","qsec"]], markers = True);

plt.title(" - mpg - cyl - qsec in mtcars - ")

sns.lineplot(data = mtcars_data[["mpg","cyl","qsec"]], markers = True, lw = 5);
plt.title(" - mpg - cyl - qsec in mtcars - ")

sns.lineplot(data = mtcars_data["mpg"])

sns.lineplot(data = mtcars_data["cyl"])

sns.lineplot(data = mtcars_data["qsec"]);
plt.title(" - mpg - cyl - qsec in mtcars - ")

sns.lineplot(data = mtcars_data["mpg"], label = "Miles/(US) gallon")

sns.lineplot(data = mtcars_data["cyl"], label = "Number of cylinders")

sns.lineplot(data = mtcars_data["qsec"], label = "1/4 mile time");
plt.title(" - mpg - cyl - qsec in mtcars - ")

sns.lineplot(data = mtcars_data["mpg"], label = "Miles/(US) gallon", lw = 1)

sns.lineplot(data = mtcars_data["cyl"], label = "Number of cylinders", lw = 3)

sns.lineplot(data = mtcars_data["qsec"], label = "1/4 mile time", lw = 5);
plt.title("- Relationship between qsec & cyl -")

sns.lineplot(x = "cyl", y = "qsec", data = mtcars_data);
plt.title("- Relationship between qsec & cyl -")

sns.lineplot(x = "cyl", y = "qsec", hue = "am", data = mtcars_data);
plt.title("am & qsec with bar plot")

sns.barplot(x = "am", y = "qsec", data = mtcars_data);
sns.barplot(x = "am", y = "qsec", data = mtcars_data, estimator = median);
sns.barplot(x = "am", y = "qsec",hue = "cyl", data = mtcars_data);
sns.catplot(x="gear", y="qsec",

            hue="cyl", col="am",

            data=mtcars_data, kind="bar",

            height=5, aspect=.9);
sns.catplot(x="am", y="qsec",

            hue="cyl", col="gear",

            data=mtcars_data, kind="bar",

            height=5, aspect=.9);
bar_data = mtcars_data.groupby(["am","cyl"]).qsec.agg([len, min, max])

bar_data
sns.barplot(x = bar_data.index, y = bar_data["len"]);
sns.barplot(x = bar_data.index, y = bar_data["len"], palette = "YlGn");
gold = pd.read_csv("../input/price-gold/gold.csv")

gold.head()
gold_heatmap = gold.pivot("Month", "Year", "Gold_Price")

gold_heatmap
sns.heatmap(data = gold_heatmap);
heatmap_data = gold_heatmap.iloc[:,-6:-1]

heatmap_data
sns.heatmap(data = heatmap_data, annot = True);
sns.heatmap(data = heatmap_data, annot = True, fmt = "");
sns.heatmap(data = gold_heatmap, linewidths = 0.9);
sns.heatmap(data = gold_heatmap, cmap="YlGnBu");
mtcars_data = pd.read_csv("../input/mtcars-data/mtcars.csv")
mtcars_data.head()
mtcars_data.dtypes
plt.title("- hp & qsec Scatter Plots -")

sns.scatterplot(x = mtcars_data["hp"], y = mtcars_data["qsec"]);
plt.title("- hp & qsec Scatter Plots by 'am' -")

sns.scatterplot(x = "hp", y = "qsec", hue = "am", data = mtcars_data);
plt.title("- hp & qsec regplot -")

sns.regplot(x = "hp", y = "qsec", data = mtcars_data);
sns.lmplot(x = "hp", y = "qsec", data = mtcars_data);
sns.lmplot(x = "hp", y = "qsec",  hue = "am", data = mtcars_data);
mtcars_data_scatter = mtcars_data.loc[:,["hp","qsec","mpg"]]

#mtcars_data_scatter = mtcars_data.iloc[:,1:-1]

mtcars_data_scatter.head()
sns.pairplot(mtcars_data_scatter);
mtcars_data_scatter = mtcars_data.loc[:,["hp","qsec","mpg","am"]]

mtcars_data_scatter.head()
sns.pairplot(mtcars_data_scatter, hue = "am");
sns.distplot(a = mtcars_data["qsec"], kde = False);
sns.distplot(a = mtcars_data["qsec"], kde = True);
sns.distplot(a = mtcars_data["qsec"], hist = False);
plt.title("- qsec & mpg density -")

sns.kdeplot(data = mtcars_data["qsec"], label = "qsec", shade = True)

sns.kdeplot(data = mtcars_data["mpg"], label = "mpg", shade = True);
sns.boxplot(x = mtcars_data["qsec"]);
sns.boxplot(x = "cyl", y = "qsec", data = mtcars_data);
sns.boxplot(x = "cyl", y = "qsec", hue = "am", data = mtcars_data);
NaN_Covid_Data = pd.DataFrame(

    data = {"Date" : ["13_07_2020","14_07_2020","15_07_2020","16_07_2020","17_07_2020","18_07_2020","19_07_2020"],

            "Number_of_Test" : [46492, np.nan, 42320, np.nan, np.nan, 40943, 41310],

            "Number_of_Case" : [1008, 992, np.nan, 933, 926, 918, np.nan],

            "Number_of_Dead" : [np.nan, np.nan, 17, 21, 18, 17, 16]})

NaN_Covid_Data
msno.matrix(NaN_Covid_Data);
msno.bar(NaN_Covid_Data);