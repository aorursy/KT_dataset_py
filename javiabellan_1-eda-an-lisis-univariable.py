import numpy   as np 

import pandas  as pd 

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline
!ls ../input/murcia-beer-challenge
df = pd.read_csv("../input/murcia-beer-challenge/beer_train.csv")

df.head()
# Variables numéricas

def plot_num(variable, title="", min=False, max=False, zeros=True, size=(16,4), opacity=.1):

    if not zeros:

        variable=variable[variable!=0]

        title += " (no zeros)"

    if min:

        variable = variable[variable >= min]

        title += " (min: "+str(min)+")"

    if max:

        variable = variable[variable <= max]

        title += " (max: "+str(max)+")"



    fig, ax = plt.subplots(figsize=size)

    ax.set_title(title, fontsize=20)

    ax2 = ax.twinx()

    sns.violinplot(variable, cut=0, palette="Set3", inner="box", ax=ax)

    sns.scatterplot(variable, y=variable.index, color="grey", linewidth=0, s=20, alpha=opacity, ax=ax2).invert_yaxis()

    

# Variables ordinales

def plot_ord(variable, title="", size=(16,4), zeros=True):

    if not zeros:

        variable=variable[variable!=0]

        title += " (no zeros)"

    plt.figure(figsize=size)

    sns.countplot(variable, color='royalblue').set_title(title, fontsize=20);

    

# Variables categoricas

def plot_cat(variable, title="", size=(16,4)):

    plt.figure(figsize=size)

    sns.countplot(y=variable, order=variable.value_counts().index).set_title(title, fontsize=20);
plot_num(df["Size(L)"], "Size (L): Amount brewed for recipe listed")
plot_num(df["Size(L)"], "Size (L): Amount brewed for recipe listed", max=100)
plot_num(df["OG"], "OG: Specific gravity of wort before fermentation")
plot_num(df["OG"], "OG", min=1.5)
plot_num(df["OG"], "OG", max=1.5, opacity=0.1)
plot_num(df["FG"], "FG: Specific gravity of wort after fermentation")
plot_num(df["FG"], "FG", min=0.95, max=1.05)
plot_num(df["FG"], "FG", min=1.05)
plot_num(df["ABV"], "ABV: Alcohol By Volume")
plot_num(df["ABV"], "ABV: Alcohol", max=15)
plot_num(df["IBU"], "IBU: International Bittering Units")
plot_num(df["IBU"], "IBU: International Bittering Units", max=200, opacity=.1)
plot_num(df["Color"], "Color: Standard Reference Method. Light to dark. Ex. 40 = black")
plot_num(df["Color"], "Color", max=50)
plot_num(df["BoilSize"], "BoilSize: Fluid at beginning of boil", opacity=.5)
plot_num(df["BoilSize"], "BoilSize", max=100)
plot_num(df["BoilTime"], "BoilTime: Time wort is boiled", opacity=.5)
plot_num(df["BoilGravity"], "BoilGravity: Specific gravity of wort before the boil")
plot_num(df["BoilGravity"], "BoilGravity", min=1, max=1.5)
plot_num(df["BoilGravity"], "BoilGravity", min=1.5, opacity=.5)
plot_num(df["Efficiency"], "Efficiency: Beer mash extraction efficiency")
plot_num(df["MashThickness"], "MashThickness: Amount of water per pound of grain")
plot_num(df["MashThickness"], "MashThickness", max=10)
plot_num(df["PrimaryTemp"], "PrimaryTemp: Temperature at the fermenting stage")
plot_ord(df["PitchRate"], "PitchRate: Yeast added to the fermentor per gravity unit - M cells/ml/deg P")
plot_cat(df["SugarScale"], "SugarScale: Scale to determine the concentration of dissolved solids in wort")
plot_cat(df["BrewMethod"], "Brew Method: Various techniques for brewing")
plot_cat(df["Style"], "Style: Tipo de cerveza (VARIABLE A PREDECIR)")
from sklearn.linear_model import Lasso

from sklearn.linear_model import LassoCV

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, LabelEncoder