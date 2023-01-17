#Kernel preperations

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier as knn

from sklearn.model_selection import train_test_split as tts

from sklearn.neighbors import NeighborhoodComponentsAnalysis as nca

from sklearn.pipeline import Pipeline as pipe



#Data import

df_base = pd.read_csv('../input/oakland-street-trees/oakland-street-trees.csv')

print(df_base.info())
#Basic summary statistics

print(df_base.describe())
#Start of data cleaning - returning the number of NaN values in the set

print(len(df_base) - df_base['WELLWIDTH'].count())
#Describing the missing data

print(df_base[df_base['WELLWIDTH'].isna()].describe())
#Dropping the observations since they do not contain useful information

df_base = df_base.dropna()



#Confirming that 5 observations were dropped (should expect 38,608 obs)

print(len(df_base))
#Bar chart for top 20 species of trees

#Subsampling data into a pandas series of top planted tree species

df_topSpecies = df_base['SPECIES'].groupby(df_base['SPECIES']).count().sort_values(ascending=False).head(20)

fig = plt.figure(dpi=90)

df_topSpecies.plot.bar()

plt.style.use('seaborn-pastel')

plt.title('Most planted trees, by species')

plt.xlabel('Species')

plt.ylabel('Number of trees')

plt.show()



print("These top 20 species make up for", df_topSpecies.sum(), "of", len(df_base), "trees planted (or", df_topSpecies.sum()/len(df_base),"% of trees).")
#Boxplots showing wellwidth for the top 5 tree species (saving space)

#subample

df_tsg = df_base[(df_base.SPECIES == 'Liquidambar styraciflua') | (df_base.SPECIES == 'Platanus acerifolia') |(df_base.SPECIES == 'Pyrus calleryana cvs') | (df_base.SPECIES == 'Prunus cerasifera/blireiana') | (df_base.SPECIES == 'Lagerstroemia indica')]



#Checking that this new data set matches the desired output

print(len(df_tsg))

print(sum(df_topSpecies.head(5)))
fig = plt.subplots(1,2)



plt.subplot(121)

ww_bp = sns.boxplot(y='WELLWIDTH',x='SPECIES', data=df_tsg, width = 0.5, palette='colorblind')

plt.xticks(rotation=90)

plt.ylabel('Well Width (ft.)')



plt.subplot(122)

wl_bp = sns.boxplot(y='WELLLENGTH',x='SPECIES', data=df_tsg, width = 0.5, palette='colorblind')

plt.xticks(rotation=90)

plt.ylabel('Well Length (ft.)')

wl_bp.yaxis.tick_right()

wl_bp.yaxis.set_label_position("right")



plt.suptitle('Dispersion of well lengths and widths, top 5 species')

plt.show(fig)
#Scatter plot for well length and width

scat = sns.pairplot(data=df_tsg, x_vars='WELLLENGTH', y_vars='WELLWIDTH',kind='scatter',hue='SPECIES', height=4, aspect=2, palette='dark')

plt.xlabel('Well length (ft.)')

plt.ylabel('Well width (ft.)')

plt.title('Length vs. width of wells, top 5 species')
#Creating the subsample of the original dataframe conditional on being a top 20 species

species_top = df_topSpecies.index

df_knn = df_base[df_base['SPECIES'].isin(species_top)]

print('Number of species in this dataset:', len(df_knn['SPECIES'].unique()))

print('Number of observations in this dataset:', len(df_knn))

print('Porportion to the original dataset:', len(df_knn)/len(df_base))
#Using sklearn to encode our categorical data

from sklearn import preprocessing as pp

le = pp.LabelEncoder()

target = le.fit_transform(df_knn['SPECIES']) #Values of 0 through 19

#target = df_knn['target']
from sklearn.neighbors import KNeighborsClassifier as knn

from sklearn.model_selection import train_test_split as tts

from sklearn.neighbors import NeighborhoodComponentsAnalysis as nca

from sklearn.pipeline import Pipeline as pipe



#Splitting the data into training and testing (70:30)

X_train, X_test, y_train, y_test = tts(df_knn, target, test_size= .30, random_state = 42)
nca_dat = nca(random_state = 42)

knn_model = knn(n_neighbors = 3)

nca_pipe = pipe([('nca', nca_dat),('knn',knn_model)])

nca_pipe.fit(X_test[['WELLWIDTH','WELLLENGTH']],y_test)
score = nca_pipe.score(X_test[['WELLWIDTH','WELLLENGTH']],y_test)

print('The NCA pipeline calssification has an accuracy score of', round(score,4))