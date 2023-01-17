import os

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import numpy as np
filepath = "/kaggle/input/world-happiness/"

dfs = [pd.read_csv(filepath + "2015.csv"), pd.read_csv(filepath + "2016.csv"), pd.read_csv(filepath + "2017.csv")] 
def display_corr(df):

    # Supprime les colonnes qui ne sont pas pertinentes pour la matrice de corrélation

    data = df

    for columnName in ['Happiness Rank', 'Standard Error', 'Lower Confidence Interval', 'Upper Confidence Interval', \

                       'Happiness.Rank', 'Whisker.high', 'Whisker.low']:

        if columnName in df.columns:

            data = data.drop(columnName, axis=1)



    sns.set(style="white")



    # Calcule la matrice de corrélation

    corr = data.corr()



    # Génére un masque pour ne pas afficher les valeurs de la partie supérieure de la matrice car ce sont des doublons

    mask = np.zeros_like(corr, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True



    # Redimensionne la fenêtre du graphe 

    plt.subplots(figsize=(8, 8))



    cmap = sns.color_palette("coolwarm", 14)



    # Affiche la heatmap

    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, square=True, linewidths=0.5, annot=True, cbar_kws={"shrink": .5})
display_corr(dfs[0])
display_corr(dfs[1])
display_corr(dfs[2])
def show_happiness_per_region(df):

    sns.set(style="whitegrid")



    # Calcul les moyennes des scores de bonheur par région et les tri par ordre décroissant 

    data = df.groupby("Region")

    data = data.mean()

    data = data["Happiness Score"]

    data = data.sort_values(ascending=False)

    data = data.to_frame()



    # Redimensionne le graphe

    f, ax = plt.subplots(figsize=(7, 7))

    

    # Affiche la légende des abscisses avec une inclinaison de 45 degrés

    plt.xticks(rotation=45, rotation_mode="anchor", ha="right")



    graph = sns.barplot(x=data.index, y="Happiness Score", data=data)

    # Affiche les valeurs au dessus des colonnes (légende)

    for rect in ax.patches:

        ax.text(rect.get_x() + 0.05, rect.get_height() + 0.1, round(rect.get_height(), 2))
show_happiness_per_region(dfs[0])
show_happiness_per_region(dfs[1])
datas = []

for df in dfs:

    data = df.rename(columns={'Economy..GDP.per.Capita.': 'Economy', 'Economy (GDP per Capita)': 'Economy', \

                             'Health (Life Expectancy)': 'Health', 'Health..Life.Expectancy.':'Health', \

                             'Trust (Government Corruption)': 'Trust', 'Trust..Government.Corruption.': 'Trust'})

    datas.append(data)
def show_top_ten(dfs, columnName):

    df_years = []

    for df in dfs:

        # Garde uniquement les 5 premiers pays du classement

        data = df.head(5)

        # Les tris par rapport à la colonne passée en paramètre

        data = data.sort_values([columnName], ascending=False)

        # Recupère le nom du pays et la colonne qui nous intéresse

        data = data[["Country", columnName]]

        # Insère les données dans un dictionnaire

        data = dict(zip(data["Country"], data[columnName]))

        df_years.append(data)

        

    # Converti les trois dictionnaires (1 par année) en DataFrame

    dfbar = pd.DataFrame.from_dict({"2015": df_years[0], "2016": df_years[1], "2017": df_years[2]}, orient='index')

    # Ajoute une colonne année

    dfbar['Year'] = dfbar.index

    

    dfbar = pd.melt(dfbar, id_vars="Year", var_name="Country", value_name=columnName)

    

    # Affiche le graphe

    sns.catplot(x="Year", y=columnName, hue="Country", data=dfbar, kind='bar')
show_top_ten(datas, "Family")
show_top_ten(datas, "Freedom")
show_top_ten(datas, "Generosity")
show_top_ten(datas, "Economy")
show_top_ten(datas, "Health")
show_top_ten(datas, "Trust")