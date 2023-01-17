import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.tree import DecisionTreeClassifier # importa o classificador da árvore de decisão

from sklearn.model_selection import train_test_split #divide o dataset em conjuntos de testes e treino

from sklearn import metrics #Importa as métricas do scikit-learn para cálculo de acurácia 

from sklearn.model_selection import cross_val_score #validação cruzada

#Leitura do Dataset

arq = '../input/musicas_decision_tree_dataset.csv'

df = pd.read_csv(arq, encoding='utf-8', engine='python', sep=',')

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
def label(n):

    if n == 1:

        return "sertanejo"

    else:

        return "feminejo"
df["label"] = df["label"].apply(label)
df.head()
#Pega o nome das colunas do dataframe

cols = list(df.columns)

cols.pop(-1)

print(cols)
X_train, X_test, y_train, y_test = train_test_split(df[cols], df.label, test_size=0.3,random_state=10) # Se test_size 0.3 então train_size = 0.7
# Instancia o Classificador da árvore de decisão

clf = DecisionTreeClassifier(criterion="gini", max_depth=None)



# Treina o classificador da árvore de decisão

clf = clf.fit(X_train,y_train)



#Prever a resposta para o conjunto de dados de teste

y_pred = clf.predict(X_test)



acuracia = metrics.accuracy_score(y_test, y_pred)
# Modelo de acurácia, qual a frequência que nosso classificador acerta?

print(metrics.classification_report(y_test, y_pred))

print("Matriz de Confusão:")

print(metrics.confusion_matrix(y_test, y_pred,labels = [ "sertanejo" ,  "feminejo"]))
scores = cross_val_score(clf, df[cols], df.label, cv=5)
print("Accuracy: %0.2f" % (scores.mean()))
cols
cols = [ 'Avg word length',

        'Avg syllables per word',

    'Lexical diversity',

 'Content diversity',

 'Rare Words Ratio']
X_train, X_test, y_train, y_test = train_test_split(df[cols], df.label, test_size=0.3,random_state=10) # Se test_size 0.3 então train_size = 0.7
clf = DecisionTreeClassifier(criterion="gini", max_depth=None)



# Treina o classificador da árvore de decisão

clf = clf.fit(X_train,y_train)



#Prever a resposta para o conjunto de dados de teste

y_pred = clf.predict(X_test)



acuracia = metrics.accuracy_score(y_test, y_pred)
# Modelo de acurácia, qual a frequência que nosso classificador acerta?

print(metrics.classification_report(y_test, y_pred))

print("Matriz de Confusão:")

print(metrics.confusion_matrix(y_test, y_pred,labels = [ "sertanejo" ,  "feminejo"]))
#Cross-validation

scores = cross_val_score(clf, df[cols], df.label, cv=5)

print("Accuracy: %0.2f" % (scores.mean()))
X_train, X_test, y_train, y_test = train_test_split(df[cols], df.label, test_size=0.5,random_state=10) 
clf = DecisionTreeClassifier(criterion="gini", max_depth=None)



# Treina o classificador da árvore de decisão

clf = clf.fit(X_train,y_train)



#Prever a resposta para o conjunto de dados de teste

y_pred = clf.predict(X_test)



acuracia = metrics.accuracy_score(y_test, y_pred)
# Modelo de acurácia, qual a frequência que nosso classificador acerta?

print(metrics.classification_report(y_test, y_pred))

print("Matriz de Confusão:")

print(metrics.confusion_matrix(y_test, y_pred,labels = [ "sertanejo" ,  "feminejo"]))
#Cross-validation

scores = cross_val_score(clf, df[cols], df.label, cv=5)

print("Accuracy: %0.2f" % (scores.mean()))