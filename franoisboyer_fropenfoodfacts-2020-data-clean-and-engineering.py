%matplotlib inline



import os

import zipfile

import urllib



import matplotlib.pyplot as plt



import numpy as np



import qgrid



DOWNLOAD_ROOT = ""

FOOD_PATH = os.path.join("/kaggle/input/fropenfoodfactsorg-2020")

FOOD_PATH_OUTPUT = os.path.join("/kaggle/working")

FOOD_PATH_FILE = os.path.join(FOOD_PATH, "fr.openfoodfacts.org.products.csv")



FOOD_PATH_FILE_OUTPUT = os.path.join(FOOD_PATH_OUTPUT, "fr.openfoodfacts.org.products_transformed.csv")





import seaborn as sns

sns.set()

from IPython.display import display, Markdown



def display_freq_table(col_names):

    for col_name in col_names:    

        effectifs = food[col_name].value_counts(bins=5)



        modalites = effectifs.index # l'index de effectifs contient les modalités





        tab = pd.DataFrame(modalites, columns = [col_name]) # création du tableau à partir des modalités

        tab["Nombre"] = effectifs.values

        tab["Frequence"] = tab["Nombre"] / len(food) # len(data) renvoie la taille de l'échantillon

        tab = tab.sort_values(col_name) # tri des valeurs de la variable X (croissant)

        tab["Freq. cumul"] = tab["Frequence"].cumsum() # cumsum calcule la somme cumulée

        

        display(Markdown('#### ' + col_name))

        display(tab)
'''

Cette fonction donne des informations pour aider à décider quelle feature on doit conserver, dans le cas où

on a 2 features qui semblent correspondre à la même notion



Elle remonte 3 informations :

% de cas où la valeur de la colonne 1 est renseignée, mais pas la 2

% de cas où la valeur de la colonne 2 est renseignée, mais pas la 1

% de cas où les valeurs de la colonne 1 et 2 sont renseignées toutes les deux



'''



def compare_na(df, col1, col2):

    num_rows, num_cols = df.shape

    

    col1notnull_col2null = round ( ( ( ( df[ (df[col1].notnull()) & ( df[col2].isna() ) ][[col1,col2]].shape[0] ) / num_rows ) * 100), 5)

    col2notnull_col1null = round ( ( ( ( df[ (df[col2].notnull()) & ( df[col1].isna() ) ][[col1,col2]].shape[0] ) / num_rows ) * 100), 5)

    col1notnull_col2notnull = round ( ( ( ( df[ (df[col1].notnull()) & ( df[col2].notnull() ) ][[col1,col2]].shape[0] ) / num_rows ) * 100), 5)

    

    print(f'Cas où {col1} est renseigné mais pas {col2} : {col1notnull_col2null}%')

    print(f'Cas où {col2} est renseigné mais pas {col1} : {col2notnull_col1null}%')

    print(f'Cas où {col1} et {col2} sont renseignés tous les deux : {col1notnull_col2notnull}%')

def read_raw_file(nblines, food_path = FOOD_PATH):

    csv_path = os.path.join(food_path, "fr.openfoodfacts.org.products.csv")

    

    fp = open(csv_path)

    

    line = ""

    

    for cnt_lines in range(nblines+1):

        line = fp.readline()

        

    print(">>>>>> Line %d" % (cnt_lines))

    print(line)

    

    
read_raw_file(0)
read_raw_file(1)

read_raw_file(2)

read_raw_file(3)

read_raw_file(4)

read_raw_file(5)
import pandas as pd



def load_food_data(food_path=FOOD_PATH):

    csv_path = os.path.join(food_path, "fr.openfoodfacts.org.products.csv")

    return pd.read_csv(csv_path, sep='\t', header=0, encoding='utf-8', low_memory=False)



food = load_food_data()
num_lines = sum(1 for line in open(FOOD_PATH_FILE, encoding='utf-8'))

message = (

f"Nombre de lignes dans le fichier (en comptant l'entête): {num_lines}\n"

f"Nombre d'instances dans le dataframe: {food.shape[0]}"

)

print(message)
food.head()
#pd.options.display.max_columns = 1000

pd.set_option("display.max_columns", 1000)

pd.set_option("display.max_rows",1000)
food[food.duplicated()]
food.drop_duplicates(inplace=True)
food.head()
food.tail()
food.info(verbose=True, null_counts=True)
food.describe()
(food.count()/food.shape[0]).sort_values(axis=0, ascending=False)
pd.options.display.max_colwidth = 100

food['url']
def analyse_donnees_manquantes(df, seuil = .7):

    nb_rows, nb_cols = df.shape



    nb_col_many_nulls = (((df.isnull().sum()) / nb_rows) > seuil)



    percentage_col_many_nulls = round(((nb_col_many_nulls.sum()) / nb_cols) * 100, 2)



    message = ( 

        f"{percentage_col_many_nulls} % des colonnes ont >= {seuil*100:0.0f}% de données manquantes \n"  

        f"Ces colonnes sont : \n"

        f"{nb_col_many_nulls[nb_col_many_nulls].to_string()}"

    )



    print(message)

    

analyse_donnees_manquantes(food)
food['countries_tags'].value_counts()
food['countries'].value_counts()
food['countries_fr'].value_counts()
len(food[food['countries_tags'].str.contains("france")==True])
len(food[ ( food['countries'].str.contains("France", case=False)==True ) | ( food['countries'].str.contains("FR")==True )  ] )
len(food[food['countries_fr'].str.contains("France")==True])
food[( food['countries_fr'].str.contains("France", na=False) == True ) &  

     ( 

         ( food['countries'].str.contains("France", case=False, na=False)==True ) | 

         ( food['countries'].str.contains("FR", na=False)==True ) == False 

     )][['countries', 'countries_fr', 'countries_tags']]
food = food[food['countries_tags'].str.contains("france")==True].copy()
food.info(verbose=True, null_counts=True)
analyse_donnees_manquantes(food)
# This line is commented out, because qgrid seems not to work on kaggle. No data is displayed  (it works on my local computer)

#qgrid.show_grid(food, grid_options={'forceFitColumns': False, 'defaultColumnWidth': 150})
import collections



min_percentage_feature_values_tokeep = collections.defaultdict(lambda : 0.01)

#min_percentage_feature_values_tokeep['vitamin-b2_100g'] = 0.01  # Use this to have specific percentage value 



def drop_lowquality_values(df, min_percentage_feature_values_tokeep):

    num_rows, num_cols = df.shape

    

    for column_name in df.columns:

        if (len(food[food[column_name].notnull()]) < min_percentage_feature_values_tokeep[column_name] * num_rows):

            df.drop([column_name], axis='columns', inplace = True)



drop_lowquality_values(food, min_percentage_feature_values_tokeep)
(food.count()/food.shape[0]).sort_values(axis=0, ascending=False)
food[food['pnns_groups_2'].str.contains("legumes", case=False, na=False)][['product_name', 'main_category_fr', 'pnns_groups_2', 'pnns_groups_1']].head(1000)
food[food['pnns_groups_1'].str.contains("legumes", case=False, na=False)][['product_name', 'main_category_fr', 'pnns_groups_2', 'pnns_groups_1']].head(1000)
food.groupby(['nutriscore_grade'])['nutriscore_grade'].count().plot(kind='pie')
food.groupby(['nutriscore_grade'])['nutriscore_grade'].count().plot(kind='bar')
sns.distplot(food[food['nutrition-score-fr_100g'].notnull()]['nutrition-score-fr_100g'], kde=True)
plt.figure(figsize=(16, 10))

sns.boxplot(x='nutriscore_grade', y='nutrition-score-fr_100g', data=food.sort_values('nutriscore_grade'))
food[( food['nutrition-score-fr_100g'] > -1) & 

     (food['nutriscore_grade'] == 'a')][['product_name', 'nutrition-score-fr_100g', 'nutriscore_grade']]
compare_na(food, 'sodium_100g', 'salt_100g')
(food['sodium_100g'] - food['salt_100g']).hist(bins=50)
compare_na(food, 'ingredients_from_palm_oil_n', 'ingredients_that_may_be_from_palm_oil_n')
(food['ingredients_from_palm_oil_n'] - food['ingredients_that_may_be_from_palm_oil_n']).hist()
food[(food['ingredients_from_palm_oil_n'] - food['ingredients_that_may_be_from_palm_oil_n']) != 0][['product_name', 'ingredients_from_palm_oil_n','ingredients_that_may_be_from_palm_oil_n']].head(1000)
compare_na(food, 'labels_tags', 'labels_fr')
compare_na(food, 'labels_tags', 'labels')
compare_na(food, 'labels_fr', 'labels')
food[['product_name', 'labels', 'labels_fr', 'labels_tags']].head(1000)
food[food['labels_tags'].str.contains("bio|france|organic", case=False, na=False)][['product_name', 'labels', 'labels_fr', 'labels_tags']].head(1000)
food[( food['labels_tags'].str.contains("bio|organic|france", case=False, na=False) == True ) &  

     ( 

         ( food['labels_fr'].str.contains("bio|organic|france", case=False, na=False)==False ) 

     )][['labels_tags', 'labels', 'labels_fr']]
food[( food['labels_fr'].str.contains("bio|organic|france", case=False, na=False) == True ) &  

     ( 

         ( food['labels_tags'].str.contains("bio|organic|france", case=False, na=False)==False ) 

     )][['labels_tags', 'labels', 'labels_fr']]
food[( food['labels_tags'].str.contains("bio|organic|france", case=False, na=False) == True ) &  

     ( 

         ( food['labels'].str.contains("bio|organic|france|AB", case=False, na=False)==False ) 

     )][['labels_tags', 'labels', 'labels_fr']]
food[['product_name', 'additives_tags', 'additives_fr', 'additives_n']].head(1000)
compare_na(food, 'additives_tags', 'additives_fr')
food[['product_name', 'states', 'states_tags', 'states_fr']].head(1000)
food[( food['states_tags'].str.contains("to-be-checked", case=False, na=False) == True ) &  

     ( 

         ( food['states'].str.contains("to-be-checked", case=False, na=False)==False ) 

     )][['states', 'states_tags', 'states_fr']]
food[( food['states'].str.contains("to-be-checked", case=False, na=False) == True ) &  

     ( 

         ( food['states_tags'].str.contains("to-be-checked", case=False, na=False)==False ) 

     )][['states', 'states_tags', 'states_fr']]
food[( food['states_tags'].str.contains("to-be-checked", case=False, na=False) == True ) &  

     ( 

         ( food['states_fr'].str.contains("A vérifier", case=False, na=False)==False ) 

     )][['states', 'states_tags', 'states_fr']]
food[( food['states_tags'].str.contains("A vérifier", case=False, na=False) == True ) &  

     ( 

         ( food['states_fr'].str.contains("to-be-checked", case=False, na=False)==False ) 

     )][['states', 'states_tags', 'states_fr']]
food[['product_name', 'main_category_fr', 'main_category']].head(1000)
food[['product_name', 'ingredients_text']].sample(1000)
food[food['ingredients_text'].notnull()]
food[food['ingredients_text'].notnull()]['ingredients_text'].str.strip().str.split(',').apply(len)
features_list = ['code', 'last_modified_t', 'product_name' , 'states_tags', 'main_category_fr','brands','brands_tags', 'nutriscore_grade','energy_100g','sugars_100g','salt_100g','saturated-fat_100g','fiber_100g','proteins_100g','ingredients_from_palm_oil_n','pnns_groups_2','pnns_groups_1','labels_tags','countries_tags','additives_tags','additives_n','ingredients_text','image_url', 'nova_group']

food = food[features_list]
def convert_category_to_number(cat):

    cat_table = {

        'a' : 5,

        'b' : 4,

        'c' : 3,

        'd' : 2,

        'e' : 1,

        'nan' : None,

    }

    

    return (cat_table.get(cat,None))





food_cat = pd.DataFrame(food['nutriscore_grade'].apply(convert_category_to_number))



food['nutrition_scoring'] = food_cat
food_no_ingredients = pd.DataFrame(food[food['ingredients_text'].notnull()]['ingredients_text'].str.strip().str.split(',').apply(len))



food['no_ingredients'] = food_no_ingredients
pd.set_option('display.max_colwidth', -1)

food[food['ingredients_text'].notnull()][['product_name','ingredients_text', 'no_ingredients']].sample(100)
no_ingredients_mean = food[food['no_ingredients'].notnull()]['no_ingredients'].mean()

no_ingredients_median = food[food['no_ingredients'].notnull()]['no_ingredients'].median()

plt.figure(figsize=(16, 10))

plt.axvline(no_ingredients_mean, 0, 1, color='red')

plt.axvline(no_ingredients_median, 0, 1, color='green')

sns.distplot(food[food['no_ingredients'].notnull()]['no_ingredients'], kde=True)
food[food['no_ingredients'].notnull()]['no_ingredients'].describe()
food[food['no_ingredients'].notnull()][['product_name', 'no_ingredients', 'ingredients_text', 'additives_n','additives_tags']].sample(100)
no_ingredients_scoring_bins = [0, 3, 5, 7, 10, np.inf]

no_ingredients_scoring_labels = [5, 4, 3, 2, 1]

                                 

food['no_ingredients_scoring'] = pd.cut(food['no_ingredients'], bins=no_ingredients_scoring_bins, labels=no_ingredients_scoring_labels)
food[food['ingredients_text'].notnull()][['product_name','ingredients_text', 'no_ingredients', 'no_ingredients_scoring']].sample(100)
# Pour une application réelle, il faudra récupérer la liste des additifs noctifs sur une source de données externe à déterminer

additives_nocive_list = ['e100', 'e101', 'e103','e104', 'e111', 'e124', 'e128', 'e131', 'e132', 'e133', 'e143', 'e171', 'e173', 'e199', 'e214', 'e215', 'e216', 'e217', 'e218', 'e219', 'e240', 'e249', 'e250', 'e251', 'e386', 'e620', 'e621','e622','e623','e624','e625', 'e924', 'e924a', 'e924b', 'e926', 'e950', 'e951', 'e952', 'e952i','e952ii','e952iii','e952iv']



additives_nocive_list_search_exp = '|'.join(additives_nocive_list)



def additives_nocive_score_item(additives_tags):

    additives_tags_list = additives_tags.split(',')

    

    additives_tags_list = [item.strip() for item in additives_tags_list]

    

    for additive_nocive in additives_nocive_list:

        if ('en:'+additive_nocive in additives_tags_list):

            return(1)

    

    return(5)



food['additives_nocive_scoring'] = pd.DataFrame(food[food['additives_tags'].notnull()]['additives_tags'].apply(additives_nocive_score_item))
# Ces valeurs de scoring ont été remplies par rapport au document Nutri-score_reglement_usage_041019.pdf







proportions_scoring_bins = {

    'energy_100g': {

        'bins': [-np.inf, 335, 1005, 1675, 2345, np.inf],

        'labels': [5, 4, 3, 2, 1]

    },

    

    'salt_100g': {

        'bins': [-np.inf, 0.225, 0.675, 1.125, 1.575, np.inf],

        'labels': [5, 4, 3, 2, 1]

    },

    

    'sugars_100g': {

        'bins': [-np.inf, 4.5, 13.5, 22.5, 31,np.inf],

        'labels': [5, 4, 3, 2, 1]

    },



    'saturated-fat_100g': {

        'bins': [-np.inf, 1, 3, 5, 7,np.inf],

        'labels': [5, 4, 3, 2, 1]

    },  

    

    'fiber_100g': {

        'bins': [-np.inf, 1.9, 2.8, 3.7, 4.7,np.inf],

        'labels': [1, 2, 3, 4, 5]

    },  

    

    'proteins_100g': {

        'bins': [-np.inf, 3.2, 4.8, 6.4, 8,np.inf],

        'labels': [1, 2, 3, 4, 5]

    },    

}





for feature_name in proportions_scoring_bins.keys():

    feature_name_scoring = feature_name + '_scoring'

    

    food[feature_name_scoring] = pd.cut(food[feature_name], bins=proportions_scoring_bins[feature_name]['bins'], labels=proportions_scoring_bins[feature_name]['labels'])

food.info()
food.groupby(['energy_100g_scoring'])['energy_100g_scoring'].count().plot(kind='pie')
food.groupby(['salt_100g_scoring'])['salt_100g_scoring'].count().plot(kind='pie')
food.info()
analyse_donnees_manquantes(food, 0.7)
bio_list = ['en:organic']

bio_europeen_list = ['en:eu-organic']

bio_francais_list = ['fr:ab-agriculture-biologique']

madeinfrance_list = ['en:made-in-france', 'fr:cuisine-en-france', 'fr:viande-francaise', 'fr:volaille-francaise ']



def bio_score_item(labels_tags):

    labels_tags_list = labels_tags.split(',')

    

    labels_tags_list = [item.strip() for item in labels_tags_list]

    

    for bio_francais in bio_francais_list:

        if (bio_francais in labels_tags_list):

            return(5)



    for bio_europeen in bio_europeen_list:

        if (bio_europeen in labels_tags_list):

            return(4)



    for bio in bio_list:

        if (bio in labels_tags_list):

            return(3)



    for madeinfrance in madeinfrance_list:

        if (madeinfrance in labels_tags_list):

            return(2)

    

    return(1)



food['bio_scoring'] = pd.DataFrame(food[food['labels_tags'].notnull()]['labels_tags'].apply(bio_score_item))
food[['bio_scoring','labels_tags']].sample(100)
nova_scoring_bins = [-np.inf, 1, 2, 3, 4]

nova_scoring_labels = [5, 4, 3, 2]

                                 

food['nova_scoring'] = pd.cut(food['nova_group'], bins=nova_scoring_bins, labels=nova_scoring_labels)
(food.count()/food.shape[0]).sort_values(axis=0, ascending=False)
food.columns
food.describe()
food.skew()
food.hist(bins=50, figsize=(20,15))
from IPython.display import display, Markdown



def display_freq_table(col_names):

    for col_name in col_names:    

        effectifs = food[col_name].value_counts(bins=5)



        modalites = effectifs.index # l'index de effectifs contient les modalités





        tab = pd.DataFrame(modalites, columns = [col_name]) # création du tableau à partir des modalités

        tab["Nombre"] = effectifs.values

        tab["Frequence"] = tab["Nombre"] / len(food) # len(data) renvoie la taille de l'échantillon

        tab = tab.sort_values(col_name) # tri des valeurs de la variable X (croissant)

        tab["Freq. cumul"] = tab["Frequence"].cumsum() # cumsum calcule la somme cumulée

        

        display(Markdown('#### ' + col_name))

        display(tab)
display_freq_table(['energy_100g','salt_100g','sugars_100g','saturated-fat_100g','fiber_100g','proteins_100g','ingredients_from_palm_oil_n'])
food.to_csv(FOOD_PATH_FILE_OUTPUT, index=False)