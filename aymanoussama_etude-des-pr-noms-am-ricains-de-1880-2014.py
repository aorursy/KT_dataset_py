import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Importons tout d'abord notre fichier de données

data = pd.read_csv("../input/us-baby-names/NationalNames.csv")
# Vérifions que le fichier de données a été importé correctement en affichant les premières lignes

data.head()
#Intéressons-nous maintenant aux différentes colonnes porposées

data.info()
# Supprimons la colonne 'Id', qui n'est pas pertinente pour cette étude

names = data.drop(['Id'], axis=1)
# Quelle est l'année min et l'année max des données proposées ?

print ('Valeurs disponibles de {} à {}'.format(min(names['Year']),max(names['Year'])))
# Comptons le nombre de valeurs par colonne afin de déterminer s'il est nécessaire de générer d'éventuelles valeurs manquantes.

names.count()



# Nous remarquons que toutes les colonnes sont toutes alimentées de 1825433 valeurs. Il n'y a donc pas besoin de générer de la donnée.
naissances = names.pivot_table('Count', index='Year', columns='Gender', aggfunc=sum)

naissances.plot(title='Evolution des sex des bébés américains en fonction du temps')
# Défintion des fonctions nécessaires à cette analyse



# Check if the name ends in vowel

def checkVowelEnd(name):

    if name[-1] in "aeiou":

        return 0

    return 1



# Check if name starts in vowel and ends with consonant

def checkVowelStartConsonantEnd(name):

    if name[0] in "AEIOU" and not name[-1] in "aeiou":

        return 0

    return 1



# Check if name starts with a vowel 

def checkVowelStart(name):

    if name[0] in "AEIOU":

        return 0

    return 1



# Check length of name

def checkNameLength(name):

    return len(name)



# check if name is short or long

def checkShortLongName(name):

    if len(name) < 5:

        return 0

    return 1
# Encodage binaire du genre de chaque prénom, puisque la plupart des algorithmes 

# ont besoin de données numériques, et n'acceptent pas les chaînes de caractères.



names.Gender = names.Gender.map({'F':1, 'M':0})
#ng pour names gender

ng = names.drop(['Year','Count'], axis=1) 



ng["VowelEnd"] = ng["Name"].apply(checkVowelEnd)

ng["VowelStart"] = ng["Name"].apply(checkVowelStart)

ng["NameLength"] = ng["Name"].apply(checkNameLength)

ng["Short/Long"] = ng["Name"].apply(checkShortLongName)

ng["ConsonantEnd/VowelStart"] = ng["Name"].apply(checkVowelStartConsonantEnd)



ng = ng.drop(['Name'], axis=1)



ng.head()
from sklearn.model_selection import train_test_split
X = ng.drop(['Gender'], axis=1)

y = ng.Gender



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)
print ('X_train : {}'.format(X_train.shape))

print ('X_test  : {}'.format(X_test.shape))

print ('Total   : {}'.format(names.shape))

# On remarquera que les nouveaux ensembles créés sont bien cohérents
import xgboost as XGB

xgb  = XGB.XGBClassifier()

xgb.fit(X_train, y_train)

y_xgb = xgb.predict(X_test)
from sklearn.metrics import classification_report, auc, roc_curve, confusion_matrix, accuracy_score
# Déterminons la matrice de confusion

print(confusion_matrix(y_test,y_xgb))
# Calculons ainsi la pertinence

print(accuracy_score(y_test,y_xgb))
# Déterminons ensuite les différents estimateurs nous permettant d'avoir des informations plus précises sur les faux positifs et les faux négatifs

# afin de savoir s'il y a une différence de précision entre la qualité de prédiction des prénoms féminins (1) et des prénoms masculins (0).

print(classification_report(y_test, y_xgb))
# Quelles sont les pourcentages de prénoms masculins et de prénoms féminins ?

males = ng.Gender == 0

females = ng.Gender == 1



print ('Prénoms masculins (0) : {}%'.format(str(round((names[males].Name.count()/names.Name.count())*100, 2))))

print ('Prénoms féminins (1) : {}%'.format(str(round((names[females].Name.count()/names.Name.count())*100, 2))))
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(X_train,y_train)

probas = lr.predict_proba(X_test)
dfprobas = pd.DataFrame(probas,columns=['proba_0','proba_1'])

dfprobas['y'] = np.array(y_test)



dfprobas
from matplotlib import pyplot as plt



plt.figure(figsize=(10,10))

sns.distplot(1-dfprobas.proba_0[dfprobas.y==0], bins=50)

sns.distplot(dfprobas.proba_1[dfprobas.y==1], bins=50)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,probas[:, 1])

roc_auc = auc(false_positive_rate, true_positive_rate)

print (roc_auc)
plt.figure(figsize=(12,12))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')        # plus mauvaise courbe

plt.plot([0,0,1],[0,1,1],'g:')     # meilleure courbe

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
def nameEvolution(name,celebrityYear):

    fig = plt.figure(figsize = (12,6))

    ax = fig.add_axes([0,0,1,1])

    ax.axvline(celebrityYear, lw = 2, ls = "--", c = "red")

    names[(names["Name"] == name) & (names["Gender"] == 0)][10:].plot(x = "Year", y = "Count", ax = ax, lw = 5, label='Prénoms masculins')

    names[(names["Name"] == name) & (names["Gender"] == 1)][10:].plot(x = "Year", y = "Count", ax = ax, lw = 5, label='Prénoms féninins')

    plt.title("Evolution du prénom '{}' au fil des années aux USA".format(name), fontsize = 18)

    plt.show()
# Angélina Jolie

# Actrice américaine ayant commencé sa carrière dans les années 1990

nameEvolution("Angelina",1990)
# Theodore Roosevelt

# Président des Etats Unis de 1901 à 1909)

nameEvolution("Theodore",1901)