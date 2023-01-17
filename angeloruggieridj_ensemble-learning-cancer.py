import pandas as pd

import matplotlib.pyplot as plt



col_names=['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']

dataframe = pd.read_csv('../input/breast-cancer-csv/breastCancer.csv', header=0, names = col_names)

array = dataframe.values



dataframe.head()
#Rimozione della feature _Sample code number_

dataframe.drop(['Sample code number'],axis = 1, inplace = True)
dataframe.describe()
dataframe.info()
dataframe.replace('?',0, inplace=True)
from sklearn.impute import SimpleImputer

# Conversione del DataFrame in array NumPy per applicare il metodo Imputer().

values = dataframe.values



imputer = SimpleImputer()

imputedData = imputer.fit_transform(values)
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler(feature_range=(0, 1))

normalizedData = scaler.fit_transform(imputedData)
#Split del dataset in X e Y

X = normalizedData[:,0:9] #Esclude la colonna relativa alla classe

Y = normalizedData[:,9] #Considera esclusivamente la classe come feature target



#Impostazione del seed

seed = 1
num_trees_list = [x for x in range(5,100+1,5)]
def print_graph(num_trees_list, means):

    plt.plot(num_trees_list, means)

    plt.xlabel('Numero alberi')

    plt.ylabel('Media Cross-Validation')

    plt.show()
def print_max_accuracy(num_trees_list, means):

    max_index = means.index(max(means))

    zipped_list = list(zip(num_trees_list, means))

    print(f"Numero alberi: {zipped_list[max_index][0]} -> Accuracy CV: {zipped_list[max_index][1]}")
# Bagged Decision Trees for Classification

from sklearn import model_selection

from sklearn.ensemble import BaggingClassifier

from sklearn.tree import DecisionTreeClassifier
#Dichiarazione del modello

kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle = True)



#Classificatore di base

#criterion='gini' restituisce un numero di alberi pari a 80, con la stessa accuracy

cart = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=1)



means = []

for num_trees in num_trees_list:

    #Dichiarazione del modello Bagging

    model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=1)

    results = model_selection.cross_val_score(model, X, Y, cv=kfold)

    means.append(results.mean())

    

print_graph(num_trees_list, means)

print_max_accuracy(num_trees_list, means)
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split





X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1, stratify=Y)



cart = cart.fit(X_train, y_train)

y_train_pred = cart.predict(X_train)

y_test_pred = cart.predict(X_test)



tree_train = accuracy_score(y_train, y_train_pred)

tree_test = accuracy_score(y_test, y_test_pred)

print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))



model = model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)



model_train = accuracy_score(y_train, y_train_pred) 

model_test = accuracy_score(y_test, y_test_pred) 

print('Bagging train/test accuracies %.3f/%.3f' % (model_train, model_test))
from sklearn import model_selection

from sklearn.ensemble import RandomForestClassifier
#Dichiarazione del modello

max_features = 3

kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle = True)



#Dichiarazione del modello RandomForestClassifier

means = []

for num_trees in num_trees_list:

    model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features, n_jobs = -1)

    results = model_selection.cross_val_score(model, X, Y, cv=kfold)

    means.append(results.mean())

    

print_graph(num_trees_list, means)

print_max_accuracy(num_trees_list, means)
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1, stratify=Y)



model = model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)



model_train = accuracy_score(y_train, y_train_pred) 

model_test = accuracy_score(y_test, y_test_pred) 

print('RandomForestClassifier train/test accuracies %.3f/%.3f' % (model_train, model_test))
from sklearn import model_selection

from sklearn.ensemble import ExtraTreesClassifier
#Definizione parametri del modello

num_trees = 100

max_features = 7



kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle = True)



#Dichiarazione del modello ExtraTreesClassifier

means = []

for num_trees in num_trees_list:

    model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features, n_jobs = -1, random_state=seed)

    results = model_selection.cross_val_score(model, X, Y, cv=kfold)

    means.append(results.mean())

    

print_graph(num_trees_list, means)

print_max_accuracy(num_trees_list, means)
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1, stratify=Y)



model = model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)



model_train = accuracy_score(y_train, y_train_pred) 

model_test = accuracy_score(y_test, y_test_pred) 

print('ExtraTreesClassifier train/test accuracies %.3f/%.3f' % (model_train, model_test))
# AdaBoost Classification

from sklearn import model_selection

from sklearn.ensemble import AdaBoostClassifier



kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle = True)

#Classificatore di base

#tree = DecisionTreeClassifier(criterion='entropy', max_depth=1,random_state=1)



#Dichiarazione del modello AdaBoost

means = []

for num_trees in num_trees_list:

    #Opzionalmente si potrebbe parametrizzare con il classificatore di base l'attributo base_estimator

    #model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed, learning_rate=0.1, base_estimator=tree)

    model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed, learning_rate=0.1)



    results = model_selection.cross_val_score(model, X, Y, cv=kfold)

    means.append(results.mean())

    

print_graph(num_trees_list, means)

print_max_accuracy(num_trees_list, means)
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split





X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1, stratify=Y)



#Se base_estimator è definito, è possibile rimuovere il commento

"""tree = tree.fit(X_train, y_train)

y_train_pred = tree.predict(X_train)

y_test_pred = tree.predict(X_test)



tree_train = accuracy_score(y_train, y_train_pred)

tree_test = accuracy_score(y_test, y_test_pred)

print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))"""



model = model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)



model_train = accuracy_score(y_train, y_train_pred) 

model_test = accuracy_score(y_test, y_test_pred) 

print('AdaBoost train/test accuracies %.3f/%.3f' % (model_train, model_test))
from sklearn import model_selection

from sklearn.ensemble import GradientBoostingClassifier



num_trees = 100

kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle = True)



#Dichiarazione del modello GradientBoostingClassifier

means = []

for num_trees in num_trees_list:

    model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)

    results = model_selection.cross_val_score(model, X, Y, cv=kfold)

    means.append(results.mean())

    

print_graph(num_trees_list, means)

print_max_accuracy(num_trees_list, means)
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split





X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1, stratify=Y)



model = model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)



model_train = accuracy_score(y_train, y_train_pred) 

model_test = accuracy_score(y_test, y_test_pred) 

print('GradientBoostingClassifier train/test accuracies %.3f/%.3f' % (model_train, model_test))
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier





kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle = True)



# Creazione dei singoli algoritmi

sub_models = []

lr = LogisticRegression(max_iter=1000)

sub_models.append(('Logistic Regression', lr))



cart = DecisionTreeClassifier()

sub_models.append(('Decision Tree (CART)', cart))



svc = SVC()

sub_models.append(('Support Vector Machine', svc))



# Creazione dell'Ensemble

ensemble = VotingClassifier(sub_models, voting='hard')

results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)

print("Accuracy: ", results.mean())
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1, stratify=Y)



ensemble = ensemble.fit(X_train, y_train)

y_train_pred = ensemble.predict(X_train)

y_test_pred = ensemble.predict(X_test)



model_train = accuracy_score(y_train, y_train_pred) 

model_test = accuracy_score(y_test, y_test_pred) 

print('VotingClassifiler train/test accuracies %.3f/%.3f' % (model_train, model_test))