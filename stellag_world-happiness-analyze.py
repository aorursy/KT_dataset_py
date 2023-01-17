import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



sns.set()



dfs = []



# Récupèration des bases de données sous forme de dataframe.

dfs.append(pd.read_csv('/kaggle/input/world-happiness/2015.csv'))

dfs.append(pd.read_csv('/kaggle/input/world-happiness/2016.csv'))

dfs.append(pd.read_csv('/kaggle/input/world-happiness/2017.csv'))

           

# Renommage des colonnes

dfs[2].rename(columns={'Economy..GDP.per.Capita.': 'Economy', 'Health..Life.Expectancy.': 'Health','Trust..Government.Corruption.': 'Trust'}, inplace=True)

for df in dfs[0:2]:

    df.rename(columns={'Economy (GDP per Capita)': 'Economy', 'Health (Life Expectancy)': 'Health','Trust (Government Corruption)': 'Trust'}, inplace=True)

    
dfMax = []



for df in dfs:

    # Pour chaque pays, on récupère la valeur la plus élevée ainsi que la catégorie correspondante.

    df['Max value'] = df[["Family", "Generosity", "Economy", "Health", "Freedom", "Trust"]].max(axis = 1)

    df['Max category'] = df[["Family", "Generosity", "Economy", "Health", "Freedom", "Trust"]].idxmax(axis = 1)

    # Ensuite, on compte le nombre d'apparitions de chaque catégorie.

    dfMax.append(df.groupby("Max category").size().reset_index(name="Max number"))



# Mise sous forme de dictionnaire des colonnes "Catégorie", "Nombre d'apparitions" pour chaque année.

df2015 = dict(zip(dfMax[0]['Max category'],dfMax[0]['Max number']))

df2016 = dict(zip(dfMax[1]['Max category'],dfMax[1]['Max number']))

df2017 = dict(zip(dfMax[2]['Max category'],dfMax[2]['Max number']))



# Création d'une nouvelle dataframe afin d'afficher toutes les années sur un même graphe.

dfbar = pd.DataFrame.from_dict({"2015": df2015, "2016": df2016, "2017": df2017}, orient='index')

dfbar['Years'] = dfbar.index

# Réarengement avec création de nouvelles colonnes.

dfbar = pd.melt(dfbar, id_vars="Years", var_name="Categories", value_name="Count")



# Affichage du graphe

graph = sns.catplot(x="Years", y="Count", hue="Categories", data=dfbar, kind="bar")

graph.fig.set_size_inches(15,8)

dfMin = []

for df in dfs:

    # Pour chaque pays, on récupère la valeur la moins élevée ainsi que la catégorie correspondante.

    df['Min value'] = df[["Family", "Generosity", "Economy", "Health", "Freedom", "Trust"]].min(axis = 1)

    df['Min category'] = df[["Family", "Generosity", "Economy", "Health", "Freedom", "Trust"]].idxmin(axis = 1)

    # Ensuite, on compte le nombre d'apparitions de chaque catégorie.

    dfMin.append(df.groupby("Min category").size().reset_index(name="Max number"))



# Mise sous forme de dictionnaire des colonnes "Catégorie", "Nombre d'apparitions" pour chaque année.

df2015 = dict(zip(dfMin[0]['Min category'],dfMin[0]['Max number']))

df2016 = dict(zip(dfMin[1]['Min category'],dfMin[1]['Max number']))

df2017 = dict(zip(dfMin[2]['Min category'],dfMin[2]['Max number']))



# Création d'une nouvelle dataframe afin d'afficher toutes les années sur un même graphe.

dfbar = pd.DataFrame.from_dict({"2015": df2015, "2016": df2016, "2017": df2017}, orient='index')

dfbar['Years'] = dfbar.index

# Réarengement avec création de nouvelles colonnes.

dfbar = pd.melt(dfbar, id_vars="Years", var_name="Categories", value_name="Count")



# Affichage du graphe

graph = sns.catplot(x="Years", y="Count", hue="Categories", data=dfbar, kind="bar")

graph.fig.set_size_inches(16,8)
dfMaxRegion = []

dfMaxValue = []



for df in dfs[0:2]:

    # Pour chaque région, on compte le nombre d'apparitions de chaque catégorie étant la plus représentée.

    dfMaxRegion.append(df.groupby(["Region", "Max category"]).size().reset_index(name="count"))

    # On fait la moyenne des valeurs pour chaque colonne et récupère seulement la colonne des valeurs maximales.

    data = df.groupby(["Region", "Max category"]).mean()

    data = data["Max value"]

    dfMaxValue.append(data.to_frame())



# Affichage du graphe en barre pour l'année 2015.

graph = sns.catplot(x="Region", y="count", hue="Max category", data=dfMaxRegion[0], kind="bar")

plt.title("2015")

plt.xticks(rotation=45, rotation_mode="anchor", ha="right")

graph.fig.set_size_inches(20,8)

plt.show()



# Affichage du nuage de points pour l'année 2015.

scatter = sns.scatterplot(x=dfMaxValue[0].index.get_level_values(0), y=dfMaxValue[0].index.get_level_values(1), hue="Max value", size="Max value", sizes=(30, 250), data=dfMaxValue[0])

scatter.legend(loc="center left", bbox_to_anchor=(1.1, 0.5), ncol=1)

plt.xticks(rotation=45, rotation_mode="anchor", ha="right")

graph.fig.set_size_inches(20,8)



# Affichage du graphe en barre pour l'année 2016.

graph = sns.catplot(x="Region", y="count", hue="Max category", data=dfMaxRegion[1], kind="bar")

plt.title("2016")

plt.xticks(rotation=45, rotation_mode="anchor", ha="right")

graph.fig.set_size_inches(20,8)

plt.show()



# Affichage du nuage de points pour l'année 2016.

scatter = sns.scatterplot(x=dfMaxValue[1].index.get_level_values(0), y=dfMaxValue[1].index.get_level_values(1), hue="Max value", size="Max value", sizes=(30, 250), data=dfMaxValue[1])

scatter.legend(loc="center left", bbox_to_anchor=(1.1, 0.5), ncol=1)

plt.xticks(rotation=45, rotation_mode="anchor", ha="right")

graph.fig.set_size_inches(20,8)
dfMinRegion = []

dfMinValue = []



for df in dfs[0:2]:

    # Pour chaque région, on compte le nombre d'apparitions de chaque catégorie étant la moins représentée.

    dfMinRegion.append(df.groupby(["Region", "Min category"]).size().reset_index(name="count"))

    # On fait la moyenne des valeurs pour chaque colonne et récupère seulement la colonne des valeurs minimales.

    data = df.groupby(["Region", "Min category"]).mean()

    data = data["Min value"]

    dfMinValue.append(data.to_frame())



# Affichage du graphe en barre pour l'année 2015.

graph = sns.catplot(x="Region", y="count", hue="Min category", data=dfMinRegion[0], kind="bar")

plt.title("2015")

plt.xticks(rotation=45, rotation_mode="anchor", ha="right")

graph.fig.set_size_inches(20,8)

plt.show()

# Affichage du nuage de points pour l'année 2015.

scatter = sns.scatterplot(x=dfMinValue[0].index.get_level_values(0), y=dfMinValue[0].index.get_level_values(1), hue="Min value", size="Min value", sizes=(30, 250), data=dfMinValue[0])

scatter.legend(loc="center left", bbox_to_anchor=(1.1, 0.5), ncol=1)

plt.xticks(rotation=45, rotation_mode="anchor", ha="right")

graph.fig.set_size_inches(35,8)



# Affichage du graphe en barre pour l'année 2016.

graph = sns.catplot(x="Region", y="count", hue="Min category", data=dfMinRegion[1], kind="bar")

plt.title("2016")

plt.xticks(rotation=45, rotation_mode="anchor", ha="right")

graph.fig.set_size_inches(20,8)

plt.show()

# Affichage du nuage de points pour l'année 2016.

scatter = sns.scatterplot(x=dfMinValue[1].index.get_level_values(0), y=dfMinValue[1].index.get_level_values(1), hue="Min value", size="Min value", sizes=(30, 250), data=dfMinValue[1])

scatter.legend(loc="center left", bbox_to_anchor=(1.1, 0.5), ncol=1)

plt.xticks(rotation=45, rotation_mode="anchor", ha="right")

graph.fig.set_size_inches(20,8)