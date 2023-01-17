# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
diamonds = pd.read_csv("/kaggle/input/diamonds/diamonds.csv", index_col=0)
diamonds["volume"] = diamonds["x"] * diamonds["y"] * diamonds["z"]

diamonds = diamonds.drop(["x", "y", "z"], axis= 1)

diamonds = diamonds.drop(diamonds.index[diamonds["volume"]== 0], axis= 0)
diamonds.head()
plt.figure(figsize=(10, 9))

df_corr = diamonds.corr()

sns.heatmap(df_corr, cmap= sns.diverging_palette(250, 15, s=75, l=40,n=9, center="dark"), annot=True)

plt.title("Correlation of White and Red Wine")

plt.show()
plt.figure(figsize=(15, 7))

sns.distplot(diamonds["price"], color="#5E3434")

plt.xlabel("Price")

plt.title("Distribution of Price in Diamond dataset")

print("Highest Price in Diamond dataset: ", diamonds["price"].max())

plt.show()
plt.figure(figsize=(15, 7))

sns.distplot(diamonds["carat"], color="#5E3434")

plt.xlabel("Carat")

plt.title("Distribution of Carat in Diamond dataset")

print("Highest Carat in Diamond dataset: ", diamonds["carat"].max())

plt.show()
# From Worst to Best (Fair= Worst) ,(Ideal= Best)

plt.figure(figsize= (15, 7))

cut = sns.countplot(x= "cut", data= diamonds, palette= sns.color_palette("cubehelix", 5),

              order=["Fair", "Good", "Very Good", "Premium", "Ideal"])

plt.xlabel("Cut")

plt.ylabel("Number of Observations")

for p in cut.patches:

    height = p.get_height().round(2)

    text = str(height)

    cut.text(p.get_x()+p.get_width()/2,height + 200,text, ha="center")

plt.show()
plt.figure(figsize= (15, 7))

sns.boxplot(x= "cut", y= "price", data= diamonds, palette= sns.color_palette("cubehelix", 5),

            order=["Fair", "Good", "Very Good", "Premium", "Ideal"])

plt.xlabel("Cut")

plt.ylabel("Price")

plt.show()
# From Worst to Best (I1= Worst) (IF= Best)

plt.figure(figsize= (15, 7))

clarity = sns.countplot(x= "clarity", data= diamonds, order=["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], palette= "Set2")

plt.xlabel("Clarity")

plt.ylabel("Number of Observations")

for p in clarity.patches:

    height = p.get_height().round(2)

    text = str(height)

    clarity.text(p.get_x()+p.get_width()/2,height + 200,text, ha="center")



plt.show()
plt.figure(figsize= (15, 7))

sns.violinplot(x= "clarity", y= "price", data= diamonds, inner= None,

            order=["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], palette= "Set2")

plt.xlabel("Clarity")

plt.ylabel("Price")

plt.show()
color_dict = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71", "#5E3434"]
# From Worst to Best (J= Worst) (D= Best)

plt.figure(figsize= (15, 7))

color = sns.countplot(x= "color", data= diamonds, palette= color_dict, order=["J", "I", "H", "G", "F", "E", "D"])

plt.xlabel("Color")

plt.ylabel("Number of Observations")

for p in color.patches:

    height = p.get_height().round(2)

    text = str(height)

    color.text(p.get_x()+p.get_width()/2,height + 200,text, ha="center")

plt.show()
plt.figure(figsize= (15, 7))

sns.boxplot(x= "color", y= "price", data= diamonds, palette= color_dict, order=["J", "I", "H", "G", "F", "E", "D"])

plt.xlabel("Color")

plt.ylabel("Price")

plt.show()
plt.figure(figsize= (9, 9))

sns.scatterplot(x= "depth", y= "table", data= diamonds, y_jitter=True, x_jitter= True, alpha=.5, hue= "cut")

plt.xlabel("Depth")

plt.ylabel("Table")

plt.legend(ncol= 3)

plt.xlim(50, 75)

plt.ylim(48, 72)

print("Correlation of Depth and Table of Diamonds: ", round(diamonds["table"].corr(diamonds["depth"]),2))

plt.show()
fig = px.scatter_3d(diamonds, x='carat', y='volume', z='price', color= "cut")

fig.show()