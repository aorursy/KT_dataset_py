# Chargement des librairies nécessaires pour le code.
import pandas as p # pour read_csv(), etc.
import numpy as np # pour random.seed(), where(), etc.
import time # pour time(), afin de mesurer le temps d'exécution du programme.
from sklearn.model_selection import train_test_split # pour train_test_split().
from keras.models import Sequential # pour add(), Sequential(), compile(), etc. 
from keras.layers import Dense # pour Dense(), compile(), etc.

# On initialise la graine du Générateur de Nombre Aléatoire (RNG) afin de rendre le résultat reproductible.
np.random.seed(7)
# Création et initialisation d'une fonction, qui prend une variable de type liste/tableau.
# Cette fonction a pour but d'obtenir chaque moyenne des valeurs de chaque colonne, comme par exemple la colonne Glucose,
# Tout en éliminant toute valeur nulle.
def moyColonne(table):
    # Création d'une liste contenant les noms de chaque colonne qui nous intéresse dans le tableau "table".
    colonesAmoyenner = table.columns
    colonesAmoyenner = colonesAmoyenner.drop('Outcome')
    
    # Parcours des noms présent dans la liste colonesAmoyenner déclarer au-dessus. 
    for c in colonesAmoyenner:
        #Calcul de la moyenne de la colonne "c" du tableau "table", dont les valeurs sont supérieurs à 0.
        moyColonnes = table[table[c] > 0][c].mean()
        
        # np.where(condition, valeur si vrai, valeur si fausse)
        # Ajout d'une colone portant le nom actuel de la colone + '_imputed'.
        # Cette nouvelle colone prend la valeur d'origine si elle est différente de 0, ou la valeur moyenne de cette colone sinon.
        table[c + '_imputed'] = np.where(table[c] != 0, table[c], moyColonnes)


# Lecture du fichier "diabetes.csv"
df = p.read_csv("../input/diabetes.csv")
# Divise notre ensemble de données en 2 parties "Train" et "Test". Ici, train contiendra 70% de l'ensemble de données et test seulement 30%
train, test = train_test_split(df, test_size = 0.3, random_state = 0)

# Les données de test peuvent parfois être faussé, c'est pour cela que l'on remplace les 0 par la valeur moyenné de la colonne.
moyColonne(test)
#moyColonne(train)

# On retire la colonne Outcome.
X_train = train.drop('Outcome', axis = 1, inplace = False)

# On retire les colonnes non-imputés.
#X_train = train.drop(df.columns, axis = 1, inplace = False)
X_test = test.drop(df.columns, axis = 1, inplace = False)

# On récupère l'Outcome de chaque tableau train et test.
Y_train = train[['Outcome']]
Y_test = test[['Outcome']]
# Initialisation du modèle.
# Sequential permet la création d'une pile de couche linéaire vide.
model = Sequential()

# Ajout de types de couches de réseau de neuronnes grâce à "add" au modèle.
# Dense permet de créer un réseau de neuronnes fully-connected. On ajoute donc des couches de réseaux de neuronnes fully-connected.
# Les paramètres de Dense, ici, sont:
#    - La dimension de l'espace de sortie.
#    - La dimension de l'espace entrant ('input_dim').
#    - La fonction d'activation, c-à-d une fonction mathématique appliqué à un signal
#       (cf https://fr.wikipedia.org/wiki/Fonction_d%27activation).
#        o La fonction "relu", ou "ReLU", signifie Rectification Linear Unit (Unité de Rectification Linéaire).
#              En d'autres termes: y = 0 si x <= 0, y = x si x > 0.
#        o La fonction "sigmoid", ou "marche douce", est une exponentielle inversée à puissance négative.
#              En d'autres termes: y = 1 / (1 + exp(-x)).
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compillation du modèle, c-à-d que le modèle est configuré pour l'entraînement.
# Les paramètres de compile, ici, sont:
#    - Un String ou une fonction loss, ou d'optimisation du score. (REQUIS)
#    - L'optimisateur, ou instance de la classe optimizer. (REQUIS)
#    - La mesure du résultat (ici, une probabilité).
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#Récupération du temps pour chronomètrer le temps d'entraînement
start_time = time.time()
# Entraînement du modèle sur les données X et Y "_train".
# Le modèle lira "epochs fois" les données X et Y train, qui sont divisé par 10 pour chaque pente de la courbe d'entraînement.
# Les paramètres de fit, ici, sont:
#    - Les données d'entrées.
#    - Les données de sortie.
#    - Le nombre d'itération sur toutes les données d'entrée et de sortie pour entraîner le réseau de neuronnes.
#    - Le nombres de données par pente mis-à-jour.
#    - L'affichage de l'entraînement à chaque "epoch". Ici, aucun affichage.
model.fit(X_train, Y_train, epochs=1500, batch_size=32, verbose = 0)

# Affichage du temps d'exécution du programme.
temps_exec = time.time() - start_time
print("\nNombre de secondes mis par le programme: %.3f secondes" % temps_exec)
# Test du modèle sur les données X et Y "_test".
# Le modèle sera testé sur la création du courbe dont chaque pente contient 10 données de test.
# Les résultats sont renvoyés dans les tableaux model.metrics_names et scores.
# La première colonne concerne le loss, la deuxième la précision (accuracy).
scores = model.evaluate(X_test, Y_test, batch_size = 10, verbose = 0)

# Affichage du score obtenu.
# On affiche ici la précision du modèle sur le test évalué précédemment,
# l'affichage étant en poucentage avec 2 chiffres après la virgule.
print("Le taux de réussite est de : %.2f%%" % (scores[1] * 100))

#ici le code dans son intégralité avec plusieurs essais sur le nombre d'époques
import pandas as p
import numpy as np
import time
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
p.options.mode.chained_assignment = None  # default='warn'
batch_size = 32

def moyColonne(table):
    colonesAmoyenner = table.columns
    colonesAmoyenner = colonesAmoyenner.drop('Outcome')
    for c in colonesAmoyenner:
        moyColonnes = table[table[c] > 0][c].mean()
        table[c + '_imputed'] = np.where(table[c] != 0, table[c], moyColonnes)
        
np.random.seed(7)
df = p.read_csv("../input/diabetes.csv")

train, test = train_test_split(df, test_size = 0.3, random_state = 0)
moyColonne(test)
X_train = train.drop('Outcome', axis = 1, inplace = False)
X_test = test.drop(df.columns, axis = 1, inplace = False)
Y_train = train[['Outcome']]
Y_test = test[['Outcome']]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

start_time = time.time()
model.fit(X_train, Y_train, epochs=500, batch_size=batch_size, verbose = 0)
temps_exec = time.time() - start_time
print("\nTemps passé à entraîner le modèle pour 500 époques : %.3f secondes" % temps_exec)

scores = model.evaluate(X_test, Y_test, batch_size = batch_size, verbose = 0)
print("Le taux de réussite est de : %.2f%%" % (scores[1] * 100))

start_time = time.time()
model.fit(X_train, Y_train, epochs=1000, batch_size=batch_size, verbose = 0)
temps_exec = time.time() - start_time
print("\nTemps passé à entraîner le modèle pour 1000 époques : %.3f secondes" % temps_exec)

scores = model.evaluate(X_test, Y_test, batch_size = batch_size, verbose = 0)
print("Le taux de réussite est de : %.2f%%" % (scores[1] * 100))

start_time = time.time()
model.fit(X_train, Y_train, epochs=1500, batch_size=batch_size, verbose = 0)
temps_exec = time.time() - start_time
print("\nTemps passé à entraîner le modèle pour 1500 époques : %.3f secondes" % temps_exec)

scores = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose = 0)
print("Le taux de réussite est de : %.2f%%" % (scores[1] * 100))
#ici le code dans son intégralité avec plusieurs essais sur la définition du réseau de neurones
import pandas as p
import numpy as np
import time
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
p.options.mode.chained_assignment = None  # default='warn'
batch_size = 32

def moyColonne(table):
    colonesAmoyenner = table.columns
    colonesAmoyenner = colonesAmoyenner.drop('Outcome')
    for c in colonesAmoyenner:
        moyColonnes = table[table[c] > 0][c].mean()
        table[c + '_imputed'] = np.where(table[c] != 0, table[c], moyColonnes)
        
np.random.seed(7)
df = p.read_csv("../input/diabetes.csv")

train, test = train_test_split(df, test_size = 0.3, random_state = 0)
moyColonne(test)
X_train = train.drop('Outcome', axis = 1, inplace = False)
X_test = test.drop(df.columns, axis = 1, inplace = False)
Y_train = train[['Outcome']]
Y_test = test[['Outcome']]

model = Sequential()
model.add(Dense(32, input_dim=8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

start_time = time.time()
model.fit(X_train, Y_train, epochs=1000, batch_size=batch_size, verbose = 0)
temps_exec = time.time() - start_time
print("\nTemps passé à entraîner le modèle pour 1000 époques : %.3f secondes" % temps_exec)

scores = model.evaluate(X_test, Y_test, batch_size = batch_size, verbose = 0)
print("Le taux de réussite est de : %.2f%%" % (scores[1] * 100))

model2 = Sequential()
model2.add(Dense(12, input_dim=8, activation='relu'))
model2.add(Dense(8, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

start_time = time.time()
model2.fit(X_train, Y_train, epochs=1000, batch_size=batch_size, verbose = 0)
temps_exec = time.time() - start_time
print("\nTemps passé à entraîner le modèle pour 1000 époques : %.3f secondes" % temps_exec)

scores = model2.evaluate(X_test, Y_test, batch_size = batch_size, verbose = 0)
print("Le taux de réussite est de : %.2f%%" % (scores[1] * 100))

model3 = Sequential()
model3.add(Dense(32, input_dim=8, activation='relu'))
model3.add(Dense(16, input_dim=8, activation='relu'))
model3.add(Dense(8, activation='relu'))
model3.add(Dense(1, activation='sigmoid'))
model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

start_time = time.time()
model3.fit(X_train, Y_train, epochs=1000, batch_size=batch_size, verbose = 0)
temps_exec = time.time() - start_time
print("\nTemps passé à entraîner le modèle pour 1000 époques : %.3f secondes" % temps_exec)

scores = model3.evaluate(X_test, Y_test, batch_size = batch_size, verbose = 0)
print("Le taux de réussite est de : %.2f%%" % (scores[1] * 100))
#ici l'entraînement comparé au K Nearest Neighbors

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

knn_uni = KNeighborsClassifier(n_neighbors=11, weights = 'uniform')
start_time=time.time()
knn_uni.fit(X_train, Y_train.values.ravel())
Y_pred = knn_uni.predict(X_test)
temps_exec = time.time() - start_time
print("\nTemps passé à entraîner le modèle : %.3f secondes" % temps_exec)
print("Le taux de réussite est de : %.2f%%" % (accuracy_score(Y_test, Y_pred) * 100))
#ici le code dans son intégralité
import pandas as p
import numpy as np
import time
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
p.options.mode.chained_assignment = None  # default='warn'
batch_size = 32

def moyColonne(table):
    colonesAmoyenner = table.columns
    colonesAmoyenner = colonesAmoyenner.drop('Outcome')
    for c in colonesAmoyenner:
        moyColonnes = table[table[c] > 0][c].mean()
        table[c + '_imputed'] = np.where(table[c] != 0, table[c], moyColonnes)
        
np.random.seed(7)
df = p.read_csv("../input/diabetes.csv")

train, test = train_test_split(df, test_size = 0.3, random_state = 0)
moyColonne(test)
X_train = train.drop('Outcome', axis = 1, inplace = False)
X_test = test.drop(df.columns, axis = 1, inplace = False)
Y_train = train[['Outcome']]
Y_test = test[['Outcome']]

model = Sequential()
model.add(Dense(32, input_dim=8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

start_time = time.time()
model.fit(X_train, Y_train, epochs=1000, batch_size=batch_size, verbose = 0)
temps_exec = time.time() - start_time
print("\nTemps passé à entraîner le modèle pour 1000 époques : %.3f secondes" % temps_exec)

scores = model.evaluate(X_test, Y_test, batch_size = batch_size, verbose = 0)
print("Le taux de réussite est de : %.2f%%" % (scores[1] * 100))
