import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
#Leitura do Dataset

arq = '../input/pima-indians-diabetes.csv'

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

dataset = pd.read_csv(arq, header=None, names=col_names)
dataset.head()
# Seleciono features de interesse

feature_cols = ['pregnant', 'insulin', 'bmi', 'age']

X = dataset[feature_cols]

y = dataset.label
# Divido X e Y em conjuntos de testes e treino

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# Treinamos um modelo de regressão logistica com nosso conjunto de treino

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)
# Predita o conjunto de teste e salva as repostas em um vetor

y_pred_class = logreg.predict(X_test)
# calculate accuracy

from sklearn import metrics

print(metrics.accuracy_score(y_test, y_pred_class))
# IMPORTANT: first argument is true values, second argument is predicted values

print(metrics.confusion_matrix(y_test, y_pred_class))
# imprimo os 10 primeiros rótulos reais e os previstos

print('Real:', y_test.values[0:10])

print('Pred:', y_pred_class[0:10])
# podemos salvar a matriz e dividi-la em 4 partes:

confusion = metrics.confusion_matrix(y_test, y_pred_class)

TP = confusion[1, 1]

TN = confusion[0, 0]

FP = confusion[0, 1]

FN = confusion[1, 0]
print((TP + TN) / float(TP + TN + FP + FN))

print(metrics.accuracy_score(y_test, y_pred_class))
print((FP + FN) / float(TP + TN + FP + FN))

print(1 - metrics.accuracy_score(y_test, y_pred_class))
print(TP / float(TP + FN))

print(metrics.recall_score(y_test, y_pred_class))
print(TN / float(TN + FP))
print(FP / float(TN + FP))
print(TP / float(TP + FP))

print(metrics.precision_score(y_test, y_pred_class))
from sklearn.model_selection import cross_val_score

cols = list(dataset.columns)

cols.pop(-1)
scores = cross_val_score(logreg, dataset[cols], dataset.label, cv=10, scoring='accuracy')

print(scores)
# usar a precisão média como uma estimativa da precisão

print(scores.mean())
# imprime as primeiras 10 respostas preditas

logreg.predict(X_test)[0:10]
# imprima as primeiras 10 probabilidades previstas de associação de classe

logreg.predict_proba(X_test)[0:10, :]
# imprima as primeiras 10 probabilidades previstas para a classe 1

logreg.predict_proba(X_test)[0:10, 1]
# salva probabilidades previstas para a classe 1

y_pred_prob = logreg.predict_proba(X_test)[:, 1]
import matplotlib.pyplot as plt


# histograma das probabilidas preditas

plt.hist(y_pred_prob, bins=8)

plt.xlim(0, 1)

plt.title('Histograma das probabilidas preditas')

plt.xlabel('Probabilidade prevista de diabetes')

plt.ylabel('Frequência')
# prever diabetes se a probabilidade prevista for maior que 0.3

from sklearn.preprocessing import binarize

y_pred_class = binarize([y_pred_prob], 0.3)[0]
# imprime as 10 primeiras probabilidades preditas

y_pred_prob[0:10]


# imprime as primeiras 10 classes previstas com o limite inferior

y_pred_class[0:10]
# Matriz de confusão anterior (threshold de 0.5)

print(confusion)
# nova matriz de confusão(threshold de 0.3)

print(metrics.confusion_matrix(y_test, y_pred_class))
# sensibilidade aumentou (costumava ser 0.24)

print(46 / float(46 + 16))
# especificidade diminiu (costumava ser 0.91)

print(80 / float(80 + 50))
# IMPORTANTE: primeiro argumento são os valores verdadeiros e segundo argumento são as probabilidades previstas

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)

plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.title('Curva ROC para classificador de diabetes')

plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)')

plt.ylabel('Taxa de Verdadeiros Positivo (Sensibilidade)')

plt.grid(True)