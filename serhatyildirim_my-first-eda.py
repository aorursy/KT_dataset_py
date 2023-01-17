%matplotlib inline

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="ticks", palette="pastel")

sns.set_style("ticks", {"xtick.major.size":12, "ytick.major.size":12})
df = pd.read_csv("../input/ks-projects-201801.csv")



# Chargement du ficher 
df.head()
df.tail() # Donné à la fin du tableau
df.shape # (Nombre de lignes, nombre de colonnes)
df.ndim # Tableau 2d
df.size # Taille du tableau 
df.info()

# Listage des type de chaque colonne 
df.columns.values # List des noms de column
df.isnull().any() # Detection s'il y a des valeurs nuls
df.isna().sum() #Somme des valeurs nuls
df = df.dropna() # Suppression des nom nul
df.isna().sum() 
df.duplicated().sum() # Detection des valeurs dupliqué
df.nunique() #nombre d'element unique par catégorie
df[['category','main_category', 'currency', 'state', 'country']].describe()
df['main_category'].value_counts().plot.bar()

plt.title("Catégory de projet présente dans le Dataset")

#ax = ax.set(xlabel='x', ylabel='sin(x)')

plt.rcParams.update({'font.size': 10})
df['currency'].value_counts().plot.bar()

plt.title("Devise les plus courrantes")

plt.rcParams.update({'font.size': 10})
df['state'].value_counts().plot.pie(figsize=(10, 10),  autopct='%.2f')

plt.title("État des projets")
df['country'].value_counts().plot.bar()

plt.title("Pays d'origin des projets")
df.boxplot(['goal', 'pledged', 'backers', 'usd pledged','usd_pledged_real', 'usd_goal_real'],figsize=(10,10))

plt.semilogy()
sns.boxplot(x="country", y='backers', data=df)

plt.title("Projet qui on rassemblé le plus de donateur par pays")

#df['backers'].astype(int)

#df.plot.scatter(x='goal', y='usd_goal_real')
#sns.boxplot(x='main_category', y='state',  data=df)

outliers = df[df['backers'] > 100000]

outliers.head()

#sns.boxplot(x="country", y='backers', data=outliers)
df[['backers']].describe() #Analyse statisque sur le nombre de donneur 
df['backers'].median() # Calcule de la médiane des donateurs
df_countries_goal_real = df[df['usd_goal_real' ]< 100000]

df_countries_goal_real_mean = (df_countries_goal_real.groupby(['country'])[['country', 'usd_goal_real']].mean())



df_countries_pledged_real = df[df['usd_pledged_real' ]< 100000]

df_countries_goal_pledged_mean = (df_countries_goal_real.groupby(['country'])[['country', 'usd_pledged_real']]).mean()

#.groupby(['country']).mean()

#final_means = pd.merge(df_countries_goal_real_mean, df_countries_goal_pledged_mean)
df3 = pd.merge(df_countries_goal_real_mean, df_countries_goal_pledged_mean, left_index=True, right_index=True )

df3.sort_values('usd_goal_real', ascending=False).plot.bar(figsize=(10,10))

plt.title("Moyenne des Objectif (en USD) à atteindre par pays par rapport à la moyenne récolté ")
sns.violinplot(data=df3, inner="points")

plt.title("Répartission des moyennes (objectif et réalité) sur l'interval 0 à 25000")
df.groupby(['country'])[['backers']].count().sort_values('backers', ascending=False).plot.bar()

#df.plot.hexbin(x='country', y='backers',gridsize=25)

plt.title("Les projets qui ont contabilité le plus de donnant")
#main_cat = df[['main_category']]

#main_cat

#df4 = pd.merge(df_countries_goal_real_mean, df_countries_goal_pledged_mean, main_cat, left_index=True, right_index=True )

#df4



#df[['main_category'] & df'usd_pledged_real', 'usd_goal_real']].groupby(['main_category']).mean().plot.bar()

#df_categories = pd.merge(df_countries_goal_real_mean, df_countries_goal_pledged_mean, left_index=True, right_index=True )

#plt.title("Moyenne des Objectif (en USD) à atteindre par pays par rapport à la moyenne récolté ")
#df4.sort_values('usd_goal_real', ascending=False).plot.bar(figsize=(10,10))
f, ax = plt.subplots(figsize=(9, 6))

sns.heatmap(df.corr(), fmt="F",annot=True, vmin=0, vmax=1,ax=ax)

plt.title("Correlation entre les différentes colonnnes du DataSet")
df.plot.scatter(x='backers', y='usd_pledged_real');
#Repartission des paiement

df['currency'].value_counts().plot.pie(figsize=(10, 10),  autopct='%.2f')

plt.title("Part des devise dans les différents projets sur Kickstarter")