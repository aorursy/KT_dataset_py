### 0.IMPORT LIBRAIRIES
# Données et calculs
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
### 0.IMPORT DATA
# UnlabeledTrainData : 25 000 revues de filmes IMDB, avec un label sentimental positif ou négatif
train = pd.read_csv("../input/labeledTrainData.tsv", delimiter="\t", 
                    quoting=3) #delimiter = tabs ; quoting=3 tells Python to ignore doubled quotes
# On peut vérifier les colonnes et le nombre de lignes / colonnes chargées. 
# On peut aussi regarder le début du dataset chargé. 
print("Nom des colonnes :", train.columns.values)
print("Taille :", train.shape)
train.head()
# Exemple de commentaire laissé sur le site, présent dans le dataset 
train["review"][0]
### 1. LE TRAITEMENT DU TEXTE

## NETTOYER LES MARKUPS HTML (ex <br/>)
# On va utiliser la bibliothèque Beautiful Soup pour nettoyer les indications HTML
# Import de la bibliothèque
from bs4 import BeautifulSoup   
# Nous allons initiliser la bibliothèque sur une seule revue, comme exemple    
example1 = BeautifulSoup(train["review"][0], "lxml")  
# Impression du texte original et du texte traité pour comparaison
print("ORIGINAL")
print (train["review"][0])
print("------------------------")
print("NETTOYÉ")
print (example1.get_text()) #get_text donne le texte sans tags ni markups
## NETTOYER PONCTUATION, NUMÉROS, MISE EN FORME ...
# Certaines ponctuations peuvent transmettre des sentiments (!!!, :(, ...), mais par simplicité 
# nous allonns supprimer toute la ponctuation. Nous allons aussi enlever les numéros. 

# Import du package "re" pour le nettoyage 
import re 
# Recherche des symboles et remplacement par des espaces vides
letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      example1.get_text() )  # The text to search
print (letters_only)

# [^a-zA-Z] : [] = appartenance à un groupe, ^= non ==> [^a-zA-z] tout ce qui n'appartient pas au groupe
# des lettres minuscules (a-z) ni majuscules (A-Z). "re.sub" cherche tous les symboles qui ne sont 
# pas des lettres et les remplacent pas des espaces vides. 
# Passer tout le texte en minuscule (uniformité)
lower_case = letters_only.lower()   
# Séparer le texte en mots
words = lower_case.split()
# Note : ce processus s'appelle "tokenization".
## NETTOYER "STOP WORDS" (des mots courants sans fort significat : as, a, the, and, ...)
# Python et la bibliothèque NLTK (Natural Language Toolkit) ont des listes de stop word prêtes 
# Import de la bibliothéque
import nltk
from nltk.corpus import stopwords # Liste de stop words
# Impression des stop words pour l'anglais 
print (stopwords.words("english"))
# Ensuite, on peut enlever ces mots de la liste de mots obtenue de la revue 
words = [w for w in words if not w in stopwords.words("english")]
print (words)
# Allez plus loin : NLTK contient aussi une fonction de Lemmatizing, qui permet de traiter des mots
# semblables comme le même mot (ex : message, messages)
## NETTOYER TOUS LES COMMENTAIRES
# Création d'une fonction qui intégre tous les nettoyages précédants
def review_to_words( raw_review ):
    # 1. Nettoyer HTML
    review_text = BeautifulSoup(raw_review, "lxml").get_text() 
    # 2. Nettoyer caractères spéciaux       
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    # 3. Passer en minuscules et séparer en mots
    words = letters_only.lower().split()                             
    # 4. Python est plus rapide pour chercher un set qu'une liste : 
    #   on convertie la liste de stop words en set 
    stops = set(stopwords.words("english"))                  
    # 5. Nettoyer stop words
    meaningful_words = [w for w in words if not w in stops]   
    # 6. Remettre les mots dans un seul string
    return( " ".join( meaningful_words ))   
# Application de la fonction à tous les commentaires
num_reviews = train["review"].size # nombre de revues
clean_train_reviews = [] # initialisation d'une liste vide pour contenir les revues nettoyées 

for i in range( 0, num_reviews ): # parcourir toutes les revues pour les nettoyer
    # Appelle la fonction pour chaque revue et ajoute le résultat dans la nouvelle liste 
    clean_train_reviews.append( review_to_words( train["review"][i] ) )
### 3. BAG OF WORDS
# Pour limiter les nombre de mots dans l'algo, on choisi de prendre les 5000 les plus fréquentes 
# (stop words déjà exclues)
# On utilise le module feature_extraction, de scikit-learn, pour créer le bag -> import
from sklearn.feature_extraction.text import CountVectorizer 

# Il faut initialiser l'objet "CountVectoriser", l'outil bag of words de scikit-learn 
# Le nettoyage des données peut être fait aussi directement sur CountVectorizer
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 
# Ensuite on utilise fit_transform : il ajuste un modèle et apprend le vocabulaire et puis transforme
# le training data en feature vectors. Il prend en entrée une liste de strings. 
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray() # conversion en numpy arrays pour simplifier manipulation
# Le vector résultant contient une ligne par commentaire sur le site et une colonne par mot retenu
train_data_features.shape
# On peut regarder les mot retenus dans le vocabulaire
vocab = vectorizer.get_feature_names()
print(vocab[0:50]) # uniquement les 50 premiers mots pour ne pas alourdir le notebook
train.head()
pd.DataFrame(train["sentiment"]).shape
### 4. RANDOM FOREST
## SET DE VALIDATION 
# Pour tester la pertinance du modèle, on sépare le train set en deux parties : une pour l'entraînement
# du modèle et l'autre pour la validation des 

# Récupération des données explicatives (X) et des résultats attendus (y)
X = train_data_features
y = train["sentiment"] # conversion de la colonne sentiment en dataframe
print("Shapes - X, y : ", X.shape, y.shape)

# Scikit-learn a des fonctions prêtes pour la répartition des données 
from sklearn.cross_validation import train_test_split # import du package
# On réserve 30% des donées à la validation (test_size = 0.3)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3) 
print("Shapes train / valid :", X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
## ENTRAÎNEMENT DU MODÈLE
print ("Training the random forest...")
from sklearn.ensemble import RandomForestClassifier # import du module RandomForest de scikit-learn
# Initialisation du RF avec 100 arbres, pour limiter le temps de calcul 
forest = RandomForestClassifier(n_estimators = 100) 

# Nous allons construire un modele avec le bag of words comme features (5000 features) 
# et les sentiment labels comme target - 25000 lignes 
forest = forest.fit( X_train, y_train )
print("Now you have a new forest :)")
# Validation des résultats
# La métrique d'erreur demandé par le problème est AUC-ROC, l'aire sous la courbe ROC 
# Courbe ROC : taux de vrai positif versus taux de faux positif
from sklearn.metrics import roc_auc_score # import de la métrique prête 
y_pred = forest.predict(X_valid) # génération des prévisions pour le sous-ensemble de validation
roc_auc_score(y_valid, y_pred) #

## GÉNÉRATION DES PRÉVISIONS
# Import du fichier de test 
test = pd.read_csv("../input/testData.tsv", header=0, delimiter="\t", \
                   quoting=3 )
# Création d'une liste vide et ajout des revues nettoyées une à une
num_reviews = len(test["review"])
clean_test_reviews = [] 
for i in range(0,num_reviews):
    clean_review = review_to_words( test["review"][i] )
    clean_test_reviews.append( clean_review )
    
# Création d'une bag of word pour le test set et conversion en numpy array
# Attention : pour le test set, on utilise uniquement transform() et non fit_transform() -> 
# -> la transformation doit être la même que pour le train set 
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Utilisation du modèle crée pour générer des prévisions des sentiment 
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )