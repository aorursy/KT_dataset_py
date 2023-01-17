#importiamo le librerie necessarie per l'analisi del dataset
import pandas as pd
# pandas è una libreria molto utilizzata in data analysis in quanto permette di gestire grosse quantità di dati in maniera veloce e inuitiva
#inoltre offre molte altre funzionalità integrate, come la creazione di grafici ecc-
import numpy as np
#numpy è una libreria utilizzata per un'insieme di operazioni numeriche
import matplotlib.pyplot as plt
#matplotlib ci permette di visualizzare l'andamento dei dati attraverso grafici e tabelle
import seaborn as sns
#seaborn, come matplotlib, permette di visualizzare grafici e creare istogrammi
#carichiamo il dataset
train= pd.read_csv("../input/train.csv")
#train = pd.read_csv("train.csv")
train.head()
#per rendere più facile la lettura, cambiamo i nomi delle colonne in lowercase
train.columns = [x.lower() for x in train.columns]
train.head(20)
#train.info ci permette di analizzare il dataset e capire quali dati possiede ogni colonna
train.info()
#L'idea principale che si utilizza quando si vanno a riempire valori vuoti, e che i 'segnaposto' che si vanno a inserire 
#non devono togliere o dare informazioni aggiuntive ai dati, ovvero 'snaturare' il dataset: infatti, se noi diamo valori
#che non c'entrano con quelli reali, è molto probabile che la classificazione avvenga non inerente con la realtà.
#Inserendo nei valori nulla la media delle età, ricaviamo un dato che non aggiunge nè toglie valore alla distribuzione dei dati,
#non alterandone l'andamento.
train['age'] = train['age'].fillna(train['age'].median())
train['fare'] = train['fare'].fillna(train['fare'].median())
#stessa cosa per embarked: siccome non sappiamo dove sono saliti i passeggeri, poniamo come segnapost la U di 'Unknown'
train['embarked'] = train['embarked'].fillna('S')
# per le cabine dobbiamo capire come si distribuiscono nel dataset:
train['cabin'] = train['cabin'].fillna('Unknown')
#per farlo utilizziamo una funzione, denominata def_embarked, e andremo a modificare il dataset attraverso l'operatore lambda
def def_embarked(point):
    if point == "S":
        return 1
    elif point == "Q":
        return 2
    elif point == "C":
        return 3
    else:
        return 0

train["embarked_"] = train.apply(lambda row:def_embarked(row["embarked"]),axis=1)

#per cabin il discorso è un po' diverso: ogni cabina è divisa in una lettera identificativa di una parte della nave più
#il numero di stanza reale. è necessario dividere in classi in base alla lettera che vi è davanti.
# identifichiamo la posizione di ogni cabina (se la hanno) all'interno della nave
def def_position(cabin):
    return cabin[:1]
train["Position"] = train.apply(lambda row:def_position(row["cabin"]), axis=1)
#value_counts() ci restituisce quanti valori ci sono all'interno di uan colonna:
train["Position"].value_counts()
#osserviamo 8 possibili classi, che andremo ad aggiungere al nostro dataset:
def def_cabin(pos):
    if pos == "C":
        return 1
    elif pos == "B":
        return 2
    elif pos == "D":
        return 3
    elif pos == "E":
        return 4
    elif pos == "F":
        return 5
    elif pos == "A":
        return 6
    elif pos == "G":
        return 7
    else: 
        return 0
train["cabin_"] = train.apply(lambda row:def_cabin(row["Position"]),axis=1)
#stessa cosa la effettuiamo con male o female
def def_sex(sex):
    if sex=="male":
        return 0
    else:
        return 1
train["sex_"] = train.apply(lambda row: def_sex(row["sex"]),axis = 1)
train = train.drop(columns="passengerid")
train = train.drop(columns="name")
train = train.drop(columns = "embarked")
train = train.drop(columns = "cabin")
train = train.drop(columns= "Position")
train = train.drop(columns="sex")

train = train.drop(columns="ticket") #drop ma c'è da rivedere, perchè non capisco come funziona
train.info()
x = []
for i in train["survived"]:
    if(i == 1):
        x.append("Survived")
    else:
        x.append("Not Survived")
titanic_target_names = np.asarray(x)
titanic_feature_names =  np.asarray(train.columns[1:])
train_ = train.drop(columns="survived")
titanic_data = np.asarray(train_.get_values())
titanic_target = np.asarray(train["survived"])
#con train_test_split dividiamo il nostro dataset in due parti: la prima che la utilizzeremo per il training, grande il 75% del totale,
#mentra la seconda la utilizzeremo per il testing, che è grande il 25% del totale. Ovviamente dividerà anche le etichette relative
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(titanic_data,titanic_target,random_state=1)
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state = 0)
tree.fit(X_train,y_train)
print("Accuracy on training set: {}".format(tree.score(X_train,y_train)))
print("Accuracy on the testing set: {}".format(tree.score(X_test,y_test)))
#graphviz è una libreria che serve per caricare grafici e salvarli. Nel nostro caso andiamo a salvare la raffigurazione
#dell'albero sopra creato e poi andremo a visualizzarlo nel kernel. C'è salvato anche nella cartella
import graphviz
from sklearn.tree import export_graphviz
export_graphviz(tree,out_file="tree.dot",class_names=["Survived","Not Survived"],feature_names=titanic_feature_names,impurity=False,filled=True)
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
import matplotlib.pyplot as plt
def plot_feature_importances(model):
    n_features = titanic_data.shape[1]
    plt.barh(range(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features), titanic_feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.show()
plot_feature_importances(tree)
results = []
importances = []
max_ = [0,0]
for i in range(1,10):
    tree = DecisionTreeClassifier(max_depth=i,random_state = 1)
    tree.fit(X_train,y_train)
    if (tree.score(X_test,y_test) > max_[0]):
        max_ = [tree.score(X_test,y_test),i-1]
    results.append(tree.score(X_test,y_test))
    importances.append(tree.feature_importances_)
plt.plot([max_[1]],[max_[0]],marker='o',color="red")
plt.plot(results)
plt.title("Accuracy on max_depth")
plt.ylabel('Accuracy')
plt.xlabel('max_depth')
plt.legend(["Max Accuracy: {0} with {1} depth".format(round(max_[0],2),max_[1])],loc=(1.04,0.5))
plt.show()
plt.plot(importances)
plt.legend(titanic_feature_names,loc=(1.04,0.05))
plt.title("Feature importances through max_depth")
plt.ylabel('Accuracy')
plt.xlabel('max_depth')
#plt.legend(["Max Accuracy: {0} with {1} depth".format(round(max_[0],2),max_[1])],loc=(1.04,0.5))
plt.show()
plt.plot(importances)
plt.plot([max_[1]],[max_[0]],marker='o',color="red")
plt.plot(results, color="red")
plt.legend(titanic_feature_names,loc=(1.04,0.05))
plt.show()
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=1,random_state=0)
forest.fit(X_train,y_train)

print("Accuracy on training set: {:.3f}".format(forest.score(X_train,y_train)))
print("Accuracy on testing set: {:.3f}".format(forest.score(X_test,y_test)))
plot_feature_importances(forest)
results_forest = []
importances_forest = []
max_forest = [0,0]
for i in range(1,10):
    forest = RandomForestClassifier(n_estimators=i,random_state=0)
    forest.fit(X_train,y_train)
    if (forest.score(X_test,y_test) > max_forest[0]):
        max_forest = [forest.score(X_test,y_test),i-1]
    results_forest.append(forest.score(X_test,y_test))
    importances_forest.append(forest.feature_importances_)
plt.plot([max_forest[1]],[max_forest[0]],marker='o',color="red")
plt.plot(results_forest)
plt.title("Accuracy on max_depth")
plt.ylabel('Accuracy')
plt.xlabel('n_estimators')
plt.legend(["Max Accuracy: {0} with {1} n_estimators".format(round(max_forest[0],2),max_forest[1])],loc=(1.04,0.5))
plt.show()
plt.plot(importances_forest)
plt.plot([max_forest[1]],[max_forest[0]],marker='o',color="red")
plt.plot(results_forest, color="red")
plt.legend(titanic_feature_names,loc=(1.04,0.05))
plt.show()
test = pd.read_csv("../input/test.csv")
test1 = pd.read_csv("../input/test.csv")
test.head()
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
test["Cabin"] = test["Cabin"].fillna("C")
test["Embarked"] = test["Embarked"].fillna("U")
test["embarked_"] = test.apply(lambda row:def_embarked(row["Embarked"]),axis=1)
test["Position"] = test.apply(lambda row:def_position(row["Cabin"]), axis=1)
test["cabin_"] = test.apply(lambda row:def_cabin(row["Position"]),axis=1)
test["sex_"] = test.apply(lambda row: def_sex(row["Sex"]),axis = 1)
test = test.drop(columns="PassengerId")
test = test.drop(columns="Name")
test = test.drop(columns = "Embarked")
test = test.drop(columns = "Cabin")
test = test.drop(columns= "Position")
test = test.drop(columns="Sex")
test = test.drop(columns="Ticket")
test.head()
best_tree = DecisionTreeClassifier(max_depth=2,random_state = 1)
best_tree.fit(X_train,y_train)
pred = best_tree.predict(test)
d =  {'PassengerId' : test1["PassengerId"],'Survived' : pred}
prediction = pd.DataFrame(d,columns=["PassengerId","Survived"])
prediction.to_csv("Kaggle_first_try.csv",index=False)