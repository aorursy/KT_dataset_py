# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# librairies

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
# import des données

train = pd.read_csv('../input/prudential-life-insurance-assessment/train.csv.zip')

test = pd.read_csv('../input/prudential-life-insurance-assessment/test.csv.zip')
# dimension du df et visualisation des données

print(test.shape)

test.head(3)
# liste des variables du df test

list(test)
# dimension du df et visualisation des données

print(train.shape)

train.sample(3)
train.Response.unique()
# liste des variables du df train

list(train)
# description des variables 'train'

train.info(verbose=True)
# description des variables 'test'

test.info(verbose=True)
print("Il y a {} valeurs prises par la variable Product_Info_2 fichier de test: \n{}".format(len(test.Product_Info_2.unique()),test.Product_Info_2.unique()))

print("Il y a {} valeurs prises par la variable Product_Info_2 fichier d'entraînement: \n{}".format(len(train.Product_Info_2.unique()),train.Product_Info_2.unique()))
# informations sur le jeu d'entrainement

train.info()
# librairie pour Rapport de données



# pour visualisation des données avec la bibliothèque ProfileReport

# /!\ tres gourmand en cpu peut faire planter 

# sûrement à cause du grand nombre de variables 

from pandas_profiling import ProfileReport
# aperçu des données d'entrainement

#train_profile = ProfileReport(train, title="Rapport sur les données d'entraînement", html={'style': {'full_width': True}}, sort="None")

#train_profile
# aperçu des données de test

#test_profile = ProfileReport(test, title="Rapport fichier test", html={'style': {'full_width': True}}, sort="None")

#test_profile
# distribution de la variable cible

plt.figure(figsize=(12,6))

sns.countplot(train.Response).set_title('Distribution de la variable cible "Response"')

plt.grid(linestyle='dotted')

plt.show()
# affichage response et poids trié 

train[['Wt', 'Response']].sort_values(by='Wt', ascending=False).tail(20)

train[['Wt', 'Response']].sort_values(by='Wt', ascending=False).head(20)
# definition des colonnes en fonction du typoe d'information qu'elles portent

Product_info_cols = ['Product_Info_{}'.format(i) for i in range(1,7)]

Insured_info_cols = ['InsuredInfo_{}'.format(i) for i in range(1,7)]

Insurance_hist_cols = ['Insurance_Hist_{}'.format(i) for i in range(1,9)]

Family_hist_cols = ['Family_Hist_{}'.format(i) for i in range(1,5)]

Medical_hist_cols = ['Medical_History_{}'.format(i) for i in range(1,41)]

Medical_key_cols = ['Medical_Keyword_{}'.format(i) for i in range(1,48)]



# regroupe les infos de chaque variables

Product_info_data = pd.concat(train[Product_info_cols], test[Product_info_cols])

Insured_info_data = pd.concat(train[Insured_info_cols], test[Insured_info_cols])

Insurance_hist_data = pd.concat(train[Insurance_hist_cols], test[Insurance_hist_cols])

Family_hist_data = pd.concat(train[Family_hist_cols], test[Family_hist_cols])

Medical_hist_data = pd.concat(train[Medical_hist_cols], test[Medical_hist_cols])

Medical_key_data = pd.concat(train[Medical_key_cols], test[Medical_key_cols])
# fonction de tracé des distributions

def plots_cols(data):

    nb_cols = len(data.columns)

    fig = plt.figsize(6*6,6(nb_cols//6+1))

    for i, col in enumerate(data.columns):

        cpt = Counter(data[col])

        keys = list(cpt.keys())

        vals= list(cpt.values())

        plt.subplot(nb_cols//6+1,6,i+1)

        plt.bar(range(len(keys)), vals, align='center')

        plt.xticks(range(len(keys)), keys)

        plt.xlabel(col)

        plt.ylabel('Distribution')

        plt.gird(linestyle='dotted')

        plt.show()
plots_cols(Product_info_cols)
#fusion de test/train et appercu Id

all_data = train.append(test).sort_values(by='Id')

print(len(all_data.Id.unique()))

all_data.shape
all_data
# age

plt.figure(figsize=(12,6))

sns.distplot(all_data['Ins_Age']).set_title('Distribution âge')

plt.grid(linestyle='dotted')

plt.show()
# taille

plt.figure(figsize=(15,6))

sns.distplot(train['Ht']).set_title('Distribution taille')

plt.grid(linestyle='dotted')

plt.show()
# poids

plt.figure(figsize=(12,6))

sns.distplot(train['Wt']).set_title('Distribution poids')

plt.grid(linestyle='dotted')

plt.show()
# Indice de masse corporelle

plt.figure(figsize=(12,6))

sns.distplot(train['BMI']).set_title('Distribution de la masse corporelle')

plt.grid(linestyle='dotted')

plt.show()
# Employment_info
# Insuranced_info
#Insurance_History
# Family_Hist
# matrice de correlation

correlation = train.corr()



# matrice triangle

mask = np.zeros_like(correlation, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# configuration de la matrice

f, ax = plt.subplots(figsize=(30,30))



# affichage

sns.heatmap(correlation, cmap='seismic', mask=mask, vmax=1, vmin=-1, center=0, square=False)

plt.show()
# fonction vecteur de corrélation

def correlation_vector(df, feature):

    corr_matrix = df.corr()

    corr_vector = corr_matrix[feature]

    corr_vector = corr_vector.sort_values(ascending=False)

    corr_vector = corr_vector.drop(feature)

    corr_vector =pd.DataFrame(corr_vector)

    f, ax = plt.subplots(figsize=(3,30))

    plt.title('Vecteur de corrélation variable ' + feature)

    sns.heatmap(corr_vector, cmap='seismic', vmax=1, vmin=-1, center=0, linewidth=0.5, cbar_kws={'shrink': .5}, annot=True, fmt='1.3f', cbar=False)



    

# Affichage corrélation entre la variables Response et les autres variables

correlation_vector(train, 'Response')


def missing(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Pourcent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return tt
# verification des données manquantes dans le jeu d'entrainement

missing(train)['Pourcent'].sort_values(ascending=False)
# affichage des 10 variables de 'train' pour lesquelles il manque le plus d'information

missing(train)['Pourcent'].sort_values(ascending=False).head(10)
# verification des données manquantes dans le jeu de test

missing(test)['Pourcent'].sort_values(ascending=False)
# affichage des 10 variables de 'test' pour lesquelles il manque le plus d'information

missing(test)['Pourcent'].sort_values(ascending=False).head(10)
train_modified = train[train.columns[train.isnull().mean() <= 0.75]]

test_modified = test[test.columns[test.isnull().mean() <= 0.75]]
# affichage des valeurs pour les données d'entrainement

train_modified.isnull().sum().sort_values(ascending=False)
# On regarde dans quelles variables il manque des informations dans train

train_modified.isnull().sum().sort_values(ascending=False).head(10)
# affichage des valeurs pour les données de test

test_modified.isnull().sum().sort_values(ascending=False)
# On regarde dans quelles variables il manque des informations dans test

test.isnull().sum().sort_values(ascending=False).head(10)
# on liste les colonnes dans lesquelles on a des valeurs égale à 'null'

list_null_train = train.columns[train.isna().any()].tolist()

list_null_test = test.columns[test.isna().any()].tolist()



# affichage des variables concernées

print(list_null_test)

print(list_null_train)

train.info(verbose=True)