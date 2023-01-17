# Importando as bibliotecas necessárias para este projeto

import numpy as np

import pandas as pd

from IPython.display import display # Permite o uso de display() para DataFrames

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt

from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.mixture import GMM

from sklearn.metrics import silhouette_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier

#from sklearn import model_selection

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.ensemble import BaggingClassifier





# Mostre matplotlib no corpo do texto (bem formatado no Notebook)

%matplotlib inline



# ignorar os warnings

import warnings

warnings.filterwarnings('ignore')





# Função para exibir o accuracy score

def show_accuracy_score(algorithm, acc_train, acc_test):

    # Registra o accuracy score do algoritmo

    list_algorithm.append(algorithm)

    list_acc_train.append(acc_train)

    list_acc_test.append(acc_test)

    

    print ("Accuracy Score para {}:".format(algorithm))

    print ("- Treinamento = {}".format(acc_train))

    print ("- Teste = {}".format(acc_test))



# Carregar o conjunto de dados para treino

try:

    ds_treino = pd.read_csv("dataset_treino.csv")

   

    print ("O dataset de treino tem {} amostras com {} atributos cada.".format(*ds_treino.shape))

except:

    print ("O dataset de treino não pode ser carregado ou não foi encontrato.")

    

# Carregar o conjunto de dados para testes

try:

    ds_teste = pd.read_csv("dataset_teste.csv")

   

    print ("O dataset de teste tem {} amostras com {} atributos cada.".format(*ds_teste.shape))

except:

    print ("O dataset d teste não pode ser carregado ou não foi encontrato.")

    
# Verificar se existem valores nulos

print ("Dataset Treino")

ds_treino.isnull().sum()

# Verificar se existem valores nulos

print ("Dataset Teste")

ds_teste.isnull().sum()
# Descrição do conjunto de dados

display(ds_teste.describe())
# Plotar matriz de dispersão para cada um dos pares de atributos dos dados



scatter_matrix = pd.plotting.scatter_matrix(ds_treino, alpha = 0.3, figsize = (14,8), diagonal = 'kde');



for ax in scatter_matrix.ravel():

    ax.set_xlabel(ax.get_xlabel(), fontsize = 12, rotation = 45)

    ax.set_ylabel(ax.get_ylabel(), fontsize = 12, rotation = 45)

    

# Plota o gráfico de calor de correlação entre os atributos



correlacao = np.around(ds_teste.corr(),decimals=3)

plt.figure(figsize=(12, 8))

ax = plt.axes()

sns.heatmap(correlacao, vmin=0.1, vmax=1, annot=True, cmap="Reds")

ax.set_title('Correlacao entre os Atributos')

plt.show()

# Define os valores para Treino e Teste



X_train, X_test, y_train, y_test = train_test_split(ds_treino.iloc[:,0:9].values, ds_treino.iloc[:,9].values, random_state=66)



# Lista para adicionar o accuracy score de cada algoritmo

list_algorithm = []

list_acc_train = []

list_acc_test  = []

# K-NN Algorithm



# Determinar o melhor valor para n_neighbors

training_accs = []

test_accs = []

neighbors_range = range(1, 20)

for n_neighbors in neighbors_range:

    

    clf = KNeighborsClassifier(n_neighbors=n_neighbors,metric = 'minkowski', p = 2)

    clf.fit(X_train, y_train)

    

    

    training_accs.append(clf.score(X_train, y_train))

    

    test_accs.append(clf.score(X_test, y_test))

    

plt.plot(neighbors_range, training_accs, label="Treino Accuracy")

plt.plot(neighbors_range, test_accs, label="Test Accuracy")

plt.ylabel("Accuracy")

plt.xlabel("n_neighbors")

plt.legend()

plt.show()





clf = KNeighborsClassifier(n_neighbors = 8, metric = 'minkowski', p = 2)

clf.fit(X_train, y_train)



predicted = clf.predict(X_train)

acc_train = accuracy_score(y_train, predicted)



predicted = clf.predict(X_test)

acc_test = accuracy_score(y_test, predicted)



show_accuracy_score('K-NN',acc_train,acc_test)
# Random Forest Classification



clf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

clf.fit(X_train, y_train)



predicted = clf.predict(X_train)

acc_train = accuracy_score(y_train, predicted)



predicted = clf.predict(X_test)

acc_test = accuracy_score(y_test, predicted)



show_accuracy_score('Random Forest Classification',acc_train,acc_test)

# Decision Tree Algorithm



clf = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

clf.fit(X_train, y_train)



predicted = clf.predict(X_train)

acc_train = accuracy_score(y_train, predicted)



predicted = clf.predict(X_test)

acc_test = accuracy_score(y_test, predicted)



show_accuracy_score('Decision Tree',acc_train,acc_test)

# Naive Bayes



clf = GaussianNB()

clf.fit(X_train, y_train)



predicted = clf.predict(X_train)

acc_train = accuracy_score(y_train, predicted)



predicted = clf.predict(X_test)

acc_test = accuracy_score(y_test, predicted)



show_accuracy_score('Naive Bayes',acc_train,acc_test)

# K-SVM



clf = SVC(kernel = 'rbf', random_state = 0)

clf.fit(X_train, y_train)



predicted = clf.predict(X_train)

acc_train = accuracy_score(y_train, predicted)



predicted = clf.predict(X_test)

acc_test = accuracy_score(y_test, predicted)



show_accuracy_score('K-SVM',acc_train,acc_test)
# SVC



clf = SVC(kernel = 'linear', random_state = 0)

clf.fit(X_train, y_train) 



predicted = clf.predict(X_train)

acc_train = accuracy_score(y_train, predicted)



predicted = clf.predict(X_test)

acc_test = accuracy_score(y_test, predicted)



show_accuracy_score('SVC',acc_train,acc_test)
# Logistic Regression



clf = LogisticRegression(random_state = 0)

clf.fit(X_train, y_train)



predicted = clf.predict(X_train)

acc_train = accuracy_score(y_train, predicted)



predicted = clf.predict(X_test)

acc_test = accuracy_score(y_test, predicted)



show_accuracy_score('Logistic Regression',acc_train,acc_test)
# Gradient Boosting



clf = GradientBoostingClassifier(random_state = 0,learning_rate=0.01)

clf.fit(X_train, y_train)



predicted = clf.predict(X_train)

acc_train = accuracy_score(y_train, predicted)



predicted = clf.predict(X_test)

acc_test = accuracy_score(y_test, predicted)



show_accuracy_score('Gradient Boosting',acc_train,acc_test)

# Cria dataset para accuracy score de cada algoritmo

accs = pd.DataFrame(data={'algorithm':list_algorithm,'accuracy_score_train': list_acc_train,'accuracy_score_test':list_acc_test}).sort_values('accuracy_score_test', ascending=False)



# Exibe a lista de accuracy score de cada algoritmo em ordem decrescente 

accs.head(-1)

# Bagged Decision Trees for Classification



seed = 7

kfold = model_selection.KFold(n_splits=10, random_state=seed)

cart = DecisionTreeClassifier()

num_trees = 100

model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)



results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)



params = {'n_estimators': [1, 100], 'base_estimator__max_leaf_nodes':[10, 15], 'base_estimator__max_depth':[4, 5, 6]}

dt = DecisionTreeClassifier()

bc = BaggingClassifier(base_estimator=cart, oob_score=True, random_state=1) #n_estimators=70, random_state=1)



# Grid Search to determine best parameters

bc_grid = GridSearchCV(estimator=bc, param_grid=params, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

bc_grid.fit(X_train, y_train)

best_params = bc_grid.best_params_

print("Melhores parâmetros para o Algoritmo")

print(best_params)

# Faz a predicao usando os melhores parâmetros

final_dt = DecisionTreeClassifier(max_leaf_nodes=10, max_depth=4)                   

final_bc = BaggingClassifier(base_estimator=final_dt, n_estimators=100, random_state=7, oob_score=True)



final_bc.fit(X_train, y_train)



final_preds = final_bc.predict(X_test)



acc_oob = final_bc.oob_score_

print("Accuracy score: {}".format(acc_oob))
# Gera o arquivo com o resultado da predicao

X_test = ds_teste.iloc[:].values



predicted = final_bc.predict(X_test)



result = ds_teste[['id']].copy()

classe = pd.Series(predicted)



result['classe'] = classe.astype(int)



result.to_csv("sampleSubmission.csv",index=False)

result.head()