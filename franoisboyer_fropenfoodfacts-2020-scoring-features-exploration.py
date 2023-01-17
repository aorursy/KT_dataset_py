%matplotlib inline



import os

import zipfile

import urllib



import matplotlib.pyplot as plt



import numpy as np



import pandas as pd

from pandas.plotting import scatter_matrix



import qgrid



FOOD_PATH = os.path.join("/kaggle/input/fropenfoodfacts-2020-feature-engineered-data/")

FOOD_TRANSFORMED_PATH_FILE = os.path.join(FOOD_PATH, "fr.openfoodfacts.org.products_transformed.csv")



import seaborn as sns

sns.set()



plt.rcParams["figure.figsize"] = [16,9] # Taille par défaut des figures de matplotlib



pd.set_option("display.max_columns", 1000)

pd.set_option("display.max_rows",1000)



import pandas as pd



def load_food_data(csv_path=FOOD_TRANSFORMED_PATH_FILE):

    return pd.read_csv(csv_path, sep=',', header=0, encoding='utf-8', low_memory=False)



food = load_food_data()
food.head()
food.info()
scoring_features = ['nutrition_scoring', 'bio_scoring', 'no_ingredients_scoring',

       'additives_nocive_scoring', 'energy_100g_scoring', 'salt_100g_scoring',

       'sugars_100g_scoring', 'saturated-fat_100g_scoring',

       'fiber_100g_scoring', 'proteins_100g_scoring', 'nova_scoring']



quantity_features = ['energy_100g', 'sugars_100g', 'salt_100g', 'saturated-fat_100g', 'fiber_100g', 'proteins_100g' ]
food.info()
food.describe()
food[scoring_features].hist(bins=5, figsize=(20,15))
plt.style.use('default')

#On peut utiliser le code ci-dessous pour changer le style et la palette de couleurs :

#print(plt.style.available)

#plt.style.use('seaborn-ticks')

#current_palette = sns.color_palette('hls', 4)



for scoring_feature in ['additives_nocive_scoring', 'sugars_100g_scoring', 'saturated-fat_100g_scoring', 'bio_scoring', 'no_ingredients_scoring']:

        ax = food.groupby(scoring_feature)[[scoring_feature]].count().plot.pie(subplots=True, title=scoring_feature)

        plt.gca().axes.get_yaxis().set_visible(False)

        # On peut utiliser le code ci-dessous pour spécifier les couleurs une par une :

        #food.groupby(scoring_feature)[[scoring_feature]].count().plot.pie(subplots=True, colors=['#2c73d2', '#0081cf', '#0089ba', '#008e9b', '#008f7a'])

    

    
def log_convert(df, features_list_toconvert):

    features_list_converted = []

    for feature_name in features_list_toconvert:

        df[feature_name + '_log'] = np.log10(df[df[feature_name] > 0][feature_name])

        features_list_converted.append(feature_name + '_log')

        

    return(features_list_converted)
plt.rcParams["figure.figsize"] = [16,9]

food[quantity_features].hist(bins=50)

plt.suptitle("Analyse univariée des quantités\nEchelle abscisses : proportion en grammes / 100g \nEchelle ordonnées : nombre d'aliments")
features_list_log = log_convert(food, quantity_features)
features_list_log
food[features_list_log].hist(bins=50)

plt.suptitle("Analyse univariée logarithmique des quantités\nEchelle des abscisses logarithmique : -3=0.001, ..., 1 = 10, 2=100, 3=1000, ... en g / 100g\nEchelle des ordonnées : nombre d'aliments")
food.describe()
plt.figure(figsize=(16, 10))



plt.axvline(np.log10(2345), 0, 1, color='red', label='à gauche de la barre rouge : scoring énergie > 1\nà droite de la barre rouge   : scoring énergie = 1')

# la valeur 2345 correspond au scoring énergie = 1 (voir notebook de cleaning)

plt.legend()

plt.title("Distribution des proportions d'énergie")

sns.distplot(food[food['energy_100g_log'].notnull()]['energy_100g_log'], kde=True, label='Densité de probabilité', axlabel='energie pour 100g (échelle logarithmique: 1 = 10, 2=100, 3=1000, ...)')

plt.legend()
plt.figure(figsize=(16, 10))

plt.axvline(np.log10(1.575), 0, 1, color='red', label='à gauche de la barre rouge : scoring sel > 1\nà droite de la barre rouge   : scoring sel = 1')

# la valeur 1.575 correspond au scoring sel = 1 (voir notebook de cleaning)

plt.legend()

plt.title('Distribution des proportions de sel')



sns.distplot(food[food['salt_100g_log'].notnull()]['salt_100g_log'], kde=True, label='Densité de probabilité', axlabel='sel pour 100g (échelle logarithmique: -3=0.001, ..., 1 = 10, 2=100, 3=1000, ...)')

plt.legend()
plt.figure(figsize=(16, 10))

plt.axvline(np.log10(31), 0, 1, color='red', label='à gauche de la barre rouge : scoring sucre > 1\nà droite de la barre rouge   : scoring sucre = 1')

# la valeur 31 correspond au scoring sucre = 1 (voir notebook de cleaning)

plt.legend()

plt.title('Distribution des proportions de sucre')

sns.distplot(food[food['sugars_100g_log'].notnull()]['sugars_100g_log'], kde=True, label='Densité de probabilité', axlabel='sucre pour 100g (échelle logarithmique: -3=0.001, ..., 1 = 10, 2=100, 3=1000, ...)')

plt.legend()
corr_matrix = food.corr()
corr_matrix[quantity_features].loc[quantity_features]
plt.title('Corrélation entre les proportions')

sns.heatmap(corr_matrix[quantity_features].loc[quantity_features], 

        xticklabels=corr_matrix[quantity_features].loc[quantity_features].columns,

        yticklabels=corr_matrix[quantity_features].loc[quantity_features].columns, cmap='coolwarm' ,center=0.20)
corr_matrix[scoring_features].loc[scoring_features]
plt.title('Corrélation entre les scorings de qualité nutritionnelle')

sns.heatmap(corr_matrix[scoring_features].loc[scoring_features], 

        xticklabels=corr_matrix[scoring_features].loc[scoring_features].columns,

        yticklabels=corr_matrix[scoring_features].loc[scoring_features].columns, cmap='coolwarm', center=0.20)
X = "bio_scoring"

Y = "nutrition_scoring"

data = food



cont = data[[X,Y]].pivot_table(index=X,columns=Y,aggfunc=len,margins=True,margins_name="Total")

cont
tx = cont.loc[:,["Total"]]

ty = cont.loc[["Total"],:]

n = len(data)

indep = tx.dot(ty) / n



c = cont.fillna(0) # On remplace les valeurs nulles par 0

measure = (c-indep)**2/indep

xi_n = measure.sum().sum()

table = measure/xi_n



sns.heatmap(table.iloc[:-1,:-1],annot=c.iloc[:-1,:-1], linewidths=.5)

plt.show()

# Voir https://en.wikipedia.org/wiki/Correlation_and_dependence
scatter_matrix(food[features_list_log], figsize=(16,16))

plt.suptitle('Diagramme de dispersion des quantités')
from sklearn import decomposition

from sklearn import preprocessing



# Import `PCA` from `sklearn.decomposition`

from sklearn.decomposition import PCA



# Build the model

pca = PCA(n_components=2)



# import de l'échantillon

data = food



# selection des colonnes à prendre en compte dans l'ACP

data_pca = food[['nutrition_scoring', 'no_ingredients_scoring',

       'additives_nocive_scoring', 'energy_100g_scoring', 'salt_100g_scoring',

       'sugars_100g_scoring', 'saturated-fat_100g_scoring',

       'fiber_100g_scoring', 'proteins_100g_scoring', 'bio_scoring']]



data_pca = data_pca.dropna()



X = data_pca.values

features = data_pca.columns



# Centrage et Réduction

std_scale = preprocessing.StandardScaler().fit(X)

X_scaled = std_scale.transform(X)



# Reduce the data, output is ndarray

reduced_data = pca.fit_transform(X_scaled)



# Inspect shape of the `reduced_data`

print(reduced_data.shape)



# print out the reduced data

print(reduced_data)
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection

import numpy as np

import pandas as pd

from scipy.cluster.hierarchy import dendrogram



def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):

    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes

        if d2 < n_comp:



            # initialisation de la figure

            fig, ax = plt.subplots(figsize=(16,16))



            # détermination des limites du graphique

            if lims is not None :

                xmin, xmax, ymin, ymax = lims

            elif pcs.shape[1] < 30 :

                xmin, xmax, ymin, ymax = -1, 1, -1, 1

            else :

                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])



            # affichage des flèches

            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité

            if pcs.shape[1] < 30 :

                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),

                   pcs[d1,:], pcs[d2,:], 

                   angles='xy', scale_units='xy', scale=1, color="grey")

                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)

            else:

                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]

                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))

            

            # affichage des noms des variables  

            if labels is not None:  

                for i,(x, y) in enumerate(pcs[[d1,d2]].T):

                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :

                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)

            

            # affichage du cercle

            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')

            plt.gca().add_artist(circle)



            # définition des limites du graphique

            plt.xlim(xmin, xmax)

            plt.ylim(ymin, ymax)



        

            # affichage des lignes horizontales et verticales

            plt.plot([-1, 1], [0, 0], color='grey', ls='--')

            plt.plot([0, 0], [-1, 1], color='grey', ls='--')



            # nom des axes, avec le pourcentage d'inertie expliqué

            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))

            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))



            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))

            plt.show(block=False)

        

def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None, illustrative_legend=None):

    for d1,d2 in axis_ranks:

        if d2 < n_comp:

 

            # initialisation de la figure       

            fig = plt.figure(figsize=(10,6))

        

            # affichage des points

            if illustrative_var is None:

                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)

            else:

                illustrative_var = np.array(illustrative_var)

                for value in np.unique(illustrative_var):

                    selected = np.where(illustrative_var == value)

                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)

                plt.legend()



            # affichage des labels des points

            if labels is not None:

                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):

                    plt.text(x, y, labels[i],

                              fontsize='14', ha='center',va='center') 

                

            # détermination des limites du graphique

            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1

            plt.xlim([-boundary,boundary])

            plt.ylim([-boundary,boundary])

        

            # affichage des lignes horizontales et verticales

            plt.plot([-100, 100], [0, 0], color='grey', ls='--')

            plt.plot([0, 0], [-100, 100], color='grey', ls='--')



            # nom des axes, avec le pourcentage d'inertie expliqué

            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))

            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))



            plt.title("Projection des aliments (coloration : "+illustrative_legend+") (sur F{} et F{})".format(d1+1, d2+1))

            plt.show(block=False)



def display_scree_plot(pca):

    scree = pca.explained_variance_ratio_*100

    plt.bar(np.arange(len(scree))+1, scree)

    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')

    plt.xlabel("rang de l'axe d'inertie")

    plt.ylabel("pourcentage d'inertie")

    plt.title("Eboulis des valeurs propres")

    plt.show(block=False)



def plot_dendrogram(Z, names):

    plt.figure(figsize=(10,25))

    plt.title('Hierarchical Clustering Dendrogram')

    plt.xlabel('distance')

    dendrogram(

        Z,

        labels = names,

        orientation = "left",

    )

plt.show()


from sklearn import decomposition

from sklearn import preprocessing



# Import `PCA` from `sklearn.decomposition`

from sklearn.decomposition import PCA



# Build the model

pca = PCA(n_components=2)



# choix du nombre de composantes à calculer

n_comp = 6



# import de l'échantillon

data = food



# selection des colonnes à prendre en compte dans l'ACP

data_pca = food[['nutrition_scoring', 'no_ingredients_scoring',

       'additives_nocive_scoring', 'energy_100g_scoring', 'salt_100g_scoring',

       'sugars_100g_scoring', 'saturated-fat_100g_scoring',

       'fiber_100g_scoring', 'proteins_100g_scoring', 'bio_scoring', 'nova_scoring']]



# préparation des données pour l'ACP

#data_pca = data_pca.fillna(data_pca.mean()) # Il est fréquent de remplacer les valeurs inconnues par la moyenne de la variable

data_pca = data_pca.dropna()



X = data_pca.values

#names = data["idCours"] # ou data.index pour avoir les intitulés



#features = data.columns

features = data_pca.columns



# Centrage et Réduction

std_scale = preprocessing.StandardScaler().fit(X)

X_scaled = std_scale.transform(X)



# Calcul des composantes principales

pca = decomposition.PCA(n_components=n_comp)

pca.fit(X_scaled)



# Eboulis des valeurs propres

display_scree_plot(pca)



# Cercle des corrélations

pcs = pca.components_

#plt.figure(figsize=(16,10))

plt.rcParams["figure.figsize"] = [16,9]

display_circles(pcs, n_comp, pca, [(0,1),(2,3),(4,5)], labels = np.array(features))





# Projection des individus

X_projected = pca.transform(X_scaled)

display_factorial_planes(X_projected, n_comp, pca, [(0,1),(2,3),(4,5)], illustrative_var=data_pca[['nutrition_scoring']].values[:,0], illustrative_legend='nutrition scoring')





plt.show()



display_factorial_planes(X_projected, n_comp, pca, [(0,1),(2,3),(4,5)], illustrative_var=data_pca[['nova_scoring']].values[:,0], illustrative_legend='nova scoring')
display_factorial_planes(X_projected, n_comp, pca, [(0,1),(2,3),(4,5)], illustrative_var=data_pca[['no_ingredients_scoring']].values[:,0], illustrative_legend='no ingredients scoring')
pcs[0]
data_pca
food_scoring_important = food[['code', 'product_name', 'image_url', 'main_category_fr', 'nutrition_scoring', 'no_ingredients_scoring', 'additives_nocive_scoring', 'bio_scoring']].dropna()
food_scoring_important.shape
food_scoring_important = food[(food['nutrition_scoring'] == 5) & (food['additives_nocive_scoring'] == 5) & (food['bio_scoring'] == 5) & (food['no_ingredients_scoring'] == 5)]
food_scoring_important['main_category_fr'].value_counts().plot(kind='bar')
food_scoring_important = food[(food_scoring_important['nutrition_scoring'] == 5) &  (food['bio_scoring'] == 5)]
food_scoring_important['main_category_fr'].value_counts().plot(kind='bar')
food_scoring_important = food[(food['nutrition_scoring'] >= 5) & (food['bio_scoring'] >= 2)& (food['no_ingredients_scoring'] >= 4)]
food_scoring_important['main_category_fr'].value_counts().plot(kind='bar')
food_scoring_important['main_category_fr'].value_counts()[:30]
import plotly.express as px



#from IPython.display import HTML, display

from IPython.display import Image



'''

Cette fonction nécessite une variable globale "scoring_features"

qui contient la liste des noms de colonnes du dataframe df à afficher dans le radar plot

'''

def display_products_radar_image(df):

    max_products_display = 100

    cnt = 0

    

    for i, j in df.iterrows(): 

        if (cnt > max_products_display):

            print('Max products display reached')

            break

            

        radar_values = df.loc[[i]][scoring_features].to_numpy()

        #print(radar_values.tolist()[0])

        radar_values[np.isnan(radar_values)] = 0

        #print(radar_values.tolist()[0])

        

        df_radius = pd.DataFrame(dict(

            r = radar_values.tolist()[0],

            theta = scoring_features))



        fig = px.line_polar(df_radius, r='r', theta='theta', line_close=True, width=600, height=400, title=df.loc[i]['product_name'])

        

        plt.figure(figsize=(10, 10))

        

        print('Produit: ')

        fig.show()

                

        #print('Image du produit: ')  

        image_url = df.loc[i]['image_url']

                        

        '''

        if (type(image_url) == str):

            print(f'image_url = <<{image_url}>>')

            # Commenté car ne fonctionne pas derrière proxy BNP

            

            image_obj = Image(df.loc[i]['image_url'], width=100) 

            try:

                display(image_obj)

                

            except:

                print('Could not display image')

        '''    

                        

        print('\n\n')

        cnt +=1         





display_products_radar_image(food_scoring_important[food_scoring_important['main_category_fr'] == 'Viandes'].head(20))





for cat_name in food_scoring_important['main_category_fr'].value_counts()[:10].iteritems():

    print(f'5 bons produits dans la catégorie {cat_name[0]}')

    

    display_products_radar_image(food_scoring_important[food_scoring_important['main_category_fr'] == cat_name[0]].head(5))