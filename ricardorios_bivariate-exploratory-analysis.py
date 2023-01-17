# Loading libraries 

import numpy as np

import pandas as pd

import seaborn as sns 

import matplotlib.pyplot as plt 

import warnings

warnings.filterwarnings('ignore')



sns.set(style="ticks", color_codes=True)
df_train = pd.read_csv("../input/train.csv")
df_train.pop("id");

df_train["target"] = df_train["target"].apply(int)

df_train["target"] = df_train["target"].apply(str)
g = sns.FacetGrid(df_train, col="target")

g.map(sns.violinplot, "0")
g = sns.FacetGrid(df_train, col="target")

g.map(sns.violinplot, "1")
g = sns.FacetGrid(df_train, col="target")

g.map(sns.violinplot, "2")
g = sns.FacetGrid(df_train, col="target")

g.map(sns.violinplot, "3")
g = sns.FacetGrid(df_train, col="target")

g.map(sns.violinplot, "4")
g = sns.FacetGrid(df_train, col="target")

g.map(sns.violinplot, "7")
g = sns.FacetGrid(df_train, col="target")

g.map(sns.violinplot, "220")
def plot_violin(l):

    for x in l:

        g = sns.FacetGrid(df_train, col="target")

        g.map(sns.violinplot, x)

    
plot_violin(["33", "65", "91"])
def plot_pair_plot(l):

    df = df_train.loc[:, l]

    g = sns.pairplot(df, hue="target")
lista_variables = ['target', '0', '1', '2', '3']

plot_pair_plot(lista_variables)
lista_variables = ['target', '33', '65', '91']

plot_pair_plot(lista_variables)