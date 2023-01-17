import matplotlib.pyplot as plt # visualização de dados
import seaborn as sns # visualização de dados

import numpy as np # linear algebra
import pandas as pd # data processing

import os # interface with the operational system
# Lista os arquivos na pasta '../input/' onde foram salvos os dados
print(os.listdir("../input/"))
df = pd.read_csv('../input/Iris.csv')
df.head()
df.info(null_counts=True)
from sklearn.model_selection import train_test_split
variables = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
target = 'Species'
RANDOM_STATE = 1
X_train, X_test, y_train, y_test = train_test_split(df[variables],
                                                    df[target], 
                                                    test_size=0.33, # 33% será amostra de teste
                                                    random_state=RANDOM_STATE,
                                                    stratify=df[target])
# Usando o módulo tree para chamar o classificador de árvore de decisão
from sklearn import tree

# Guardando a semente aleatória
RANDOM_STATE = 1

# Todos esses são os parâmetros da DecisionTree implementada no scikit-learn, busque sua explicação detalhada na documentação
dec_clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, 
                                      min_samples_split=2, min_samples_leaf=1, 
                                      min_weight_fraction_leaf=0.0, max_features=None, 
                                      random_state=RANDOM_STATE, max_leaf_nodes=None, 
                                      min_impurity_decrease=0.0, min_impurity_split=None, 
                                      class_weight=None, presort=False)

# Usando os arquivos de treino com as variáveis e o target, ajustamos um modelo
dec_clf.fit(X_train, y_train)

# Usando a biblioteca graphviz, podemos plotar a árvore gerada pelo nosso modelo
import graphviz
dot_data = tree.export_graphviz(dec_clf, out_file=None, feature_names=variables, 
                                class_names=dec_clf.classes_, filled=True) 
graph = graphviz.Source(dot_data) 
graph
# Aplica o modelo nas variáveis de teste
y_pred = dec_clf.predict(X_test)
from sklearn.metrics import confusion_matrix

# Compara o resultado do modelo com o resultado verdadeiro
cnf_matrix = confusion_matrix(y_test, y_pred)

# Constrói um Dataframe pandas com a matriz de confusão apenas para ficar mais legível
pd.DataFrame(cnf_matrix, columns=('Prev ' + dec_clf.classes_), 
             index=('True ' + dec_clf.classes_))
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    plt.grid(False)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
plot_confusion_matrix(cnf_matrix, dec_clf.classes_, normalize=False, 
                      title='Confusion matrix', cmap=plt.cm.Greens)
def prec_revoc(cnf_matrix, classes):
    prec = []
    revoc = []
    for i in range(len(classes)):
        prec.append(cnf_matrix[i][i]/cnf_matrix[:][i].sum())
        revoc.append(cnf_matrix[i][i]/cnf_matrix[i].sum())
    
    print(pd.DataFrame([prec,revoc], columns=classes, index=['Precisão', 'Revocação']))
prec_revoc(cnf_matrix, dec_clf.classes_)
# Importa o classificador do scikit-learn
from sklearn.ensemble import RandomForestClassifier

# Salva a semente aleatória
RANDOM_STATE = 3

# Ajusta os parâmetros de um classificador do tipo RandomForest. 
# Verificar a descrição de cada um na documentação
rnd_clf = RandomForestClassifier(n_estimators=30, criterion='gini', max_depth=None, 
                                 min_samples_split=4, min_samples_leaf=2,
                                 min_weight_fraction_leaf=0.0, max_features='auto', 
                                 max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                 min_impurity_split=None, bootstrap=True, oob_score=False, 
                                 n_jobs=1, random_state=RANDOM_STATE, 
                                 verbose=0, warm_start=False, class_weight=None)

# Ajusta um modelo aos dados de treino
rnd_clf.fit(X_train, y_train)

# Prevê os valores para as variáveis de teste com o modelo
y_pred = rnd_clf.predict(X_test)

# Gera a matrix de confusão
cnf_matrix = confusion_matrix(y_test, y_pred)

# Plota a matriz de confusão
plot_confusion_matrix(cnf_matrix, rnd_clf.classes_, normalize=False, 
                      title='Confusion matrix', cmap=plt.cm.Greens)
prec_revoc(cnf_matrix, rnd_clf.classes_)
# Chama o módulo export_graphviz da biblioteca do sklearn
from sklearn.tree import export_graphviz

# Escolhe a quarta árvore gerada pelo modelo random forest
tree = rnd_clf.estimators_[3]

# Plota essa árvore que é uma nas "n_estimators" do modelo random forest
view = export_graphviz(tree, out_file=None, feature_names = variables, 
                       class_names = rnd_clf.classes_, filled = True)
graphviz.Source(view)
# Importando o módulo com o classificador
from sklearn import svm

# Salvando a semente aleatória
RANDOM_STATE = 1

# Ajustando os parâmetros do classificador
svm_clf = svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True, 
                        intercept_scaling=1, loss='squared_hinge', max_iter=1000, 
                        multi_class='ovr', penalty='l2', random_state=None, tol=0.0001, 
                        verbose=0)

# Ajustando o modelo aos dados de treino
svm_clf.fit(X_train,y_train)

# Prevendo os valores dos dados de teste com o modelo treinado
y_pred = svm_clf.predict(X_test)

# Comparando os dados previstos com os reais e gerando a matriz de confusão
cnf_matrix = confusion_matrix(y_test, y_pred)

# Plotando a matriz de confusão de forma mais amigável
classes = svm_clf.classes_
plot_confusion_matrix(cnf_matrix, classes, normalize=False, 
                      title='Confusion matrix', cmap=plt.cm.Greens)
prec_revoc(cnf_matrix, svm_clf.classes_)