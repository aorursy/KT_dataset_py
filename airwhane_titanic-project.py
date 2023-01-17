%matplotlib inline

import numpy as np

import pandas as pd

import seaborn as sb



informationsTitanic = pd.read_csv('../input/titanic/train.csv', header = 0, sep = ',')

#print(type(informationsTitanic))

#full_data = [informationsTitanic, test]

informationsTitanic.info(verbose=False)

informationsTitanic.head(n=30)



print('Voici les informations relatives à notre fichier csv : \n\n')

print (informationsTitanic.info())

print('\n\nVoici quelques données présentes dans le fichier csv :\n\n')

print (informationsTitanic)

sb.heatmap(informationsTitanic.isnull(), yticklabels=False, cmap='viridis')
#ajout d'une colonne NewSex qui contient 1 pour Male et O pour Female #################################



dictarget = {'female': 0, 'male': 1}

informationsTitanic['Sex'] = informationsTitanic['Sex'].map(dictarget)

#print (informationsTitanic['New_Sex'])

print (informationsTitanic['Sex'])

    

#######################################################################################################



#ajout d'une colonne New_Age ##########################################################################



informationsTitanic['New_Age'] = informationsTitanic['Age']

    #print (informationsTitanic)

    

#######################################################################################################



#calcul de la moyenne sans prendre en compte les NaN ##################################################



matrice_age_calcul_mean = informationsTitanic['New_Age']



add = 0

passage_if = 0

passage_else = 0

noValue = "NaN"

    

for compteur in range (len(matrice_age_calcul_mean)):

    if(float(matrice_age_calcul_mean[compteur]) > 0 and float(matrice_age_calcul_mean[compteur]) < 200 ):

        passage_if = passage_if + 1

        add = add + matrice_age_calcul_mean[compteur]

    else:

        matrice_age_calcul_mean[compteur] = 0

        add = add + matrice_age_calcul_mean[compteur]

        passage_else = passage_else + 1

        

#print (matrice_age_calcul_mean)

#print (compteur)

#print (add)

#print (passage_if)

#print (passage_else)



mean_matrice_age = (add)/(passage_if)

print('This is the mean of the age : ' + str(mean_matrice_age))



#######################################################################################################



#modifier les NaN par la moyenne calculée précedemment ################################################



compteur2 = 0

#print(informationsTitanic['Age'])

matrice_age_sans_NaN = informationsTitanic['Age']



for compteur2 in range (len(matrice_age_sans_NaN)):

    if(float(matrice_age_sans_NaN[compteur2]) > 0 and float(matrice_age_sans_NaN[compteur2]) < 200 ):

        print('\n')

    else:

        matrice_age_sans_NaN[compteur2] = round (mean_matrice_age, 2)



#######################################################################################################



print (mean_matrice_age)

print (round (mean_matrice_age, 2))



#print (informationsTitanic['Age'])

#print (matrice_age_sans_NaN)



#comparaison du nombre de non null valeur dans Age

#print (informationsTitanic.info())

#print (informationsTitanic)

print (informationsTitanic['Age'])

print (informationsTitanic['New_Age'])

sb.heatmap(informationsTitanic.isnull(), yticklabels=False, cmap='viridis')
import seaborn as sb

import time as time



print ("Faites votre choix : \n Pour le graphe 'Survivant', tapez 1\n Pour le graphe 'Sex', tapez 2\n Pour le graphe 'Class', tapez 3\ Pour quitter, tapez 0")

#choix = input()

#choix_numero = int(choix)



while True:

    choix = int(input())

    if choix == 1:

        print("choix 1")

        time.sleep(10)

        sb.countplot(x='Survived' , data=informationsTitanic)

    elif choix == 2:

        print("choix 2")

        sb.countplot(x='Survived' , hue='Sex' , data=informationsTitanic)

    elif (choix < 0 and choix > 3):

        print("Choix non valide")

    else:

        break

print("Sortie de boucle")
import seaborn as sb

import time as time



print ("Faites votre choix : \n Pour le graphe 'Survivant', tapez 1\n Pour le graphe 'Sex', tapez 2\n Pour le graphe 'Class', tapez 3\ Pour quitter, tapez 0")

#choix = input()

#choix_numero = int(choix)

choix = int(input())

if choix == 1:

    print("choix 1")

    sb.countplot(x='Survived' , data=informationsTitanic)

if choix == 2:

    print("choix 2")

    sb.countplot(x='Survived' , hue='Sex' , data=informationsTitanic)

if choix == 3:

    print("choix 3")

    sb.countplot(x='Survived' , hue='Pclass' , data=informationsTitanic)

if (choix < 0 and choix > 3):

    print("Choix non valide")
print (informationsTitanic[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
print (informationsTitanic[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())
for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[ dataset['SibSp'] + dataset['Parch'] + 1 == 1, 'IsAlone'] = 1

print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
for dataset in full_data:

    age_avg 	   = dataset['Age'].mean()

    age_std 	   = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

    
for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 2, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] >  2) & (dataset['Age'] <=  8), 'Age'] = 1

    dataset.loc[(dataset['Age'] >  8) & (dataset['Age'] <= 16), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 5

    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 6



# Feature Selection

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'SibSp',\

                 'Parch', 'Embarked']

train = train.drop(drop_elements, axis = 1)

test  = test.drop(drop_elements, axis = 1)



print (train.head(10))



train = train.values

test  = test.values



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression



classifiers = [

    KNeighborsClassifier(5),

    SVC(probability=True),

    DecisionTreeClassifier(),

    LogisticRegression()]



log_cols = ["Classifier", "Accuracy"]

log 	 = pd.DataFrame(columns=log_cols)



sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)



X = train[0::, 1::]

y = train[0::, 0]



acc_dict = {}



for train_index, test_index in sss.split(X, y):

	X_train, X_test = X[train_index], X[test_index]

	y_train, y_test = y[train_index], y[test_index]

	

	for clf in classifiers:

		name = clf.__class__.__name__

		clf.fit(X_train, y_train)

		train_predictions = clf.predict(X_test)

		acc = accuracy_score(y_test, train_predictions)

		if name in acc_dict:

			acc_dict[name] += acc

		else:

			acc_dict[name] = acc



for clf in acc_dict:

	acc_dict[clf] = acc_dict[clf] / 10.0

	log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)

	log = log.append(log_entry)



plt.xlabel('Accuracy')

plt.title('Classifier Accuracy')



sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
candidate_classifier = SVC()

candidate_classifier.fit(train[0::, 1::], train[0::, 0])

result = candidate_classifier.predict(test)
