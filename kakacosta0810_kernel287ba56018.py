# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt
# Importando o dataframe e o train_test_split

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
# Carregar o Dataframe

cancer = load_breast_cancer()

df = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns= np.append(cancer['feature_names'], ['target']))
# Informações básicas sobre o df

df.info()
# visualizando os 5 primeiros elementos do df 

df.head()
# Distribuição da variável target, onde 1 trata de um cancer benigno e 0 um maligno.

df['target'].value_counts()
df['target'].value_counts().plot(kind='bar', title='Count (target)')
X = df.drop('target',axis=1) # df sem variavél target

y = df.target # df com variável target



    # Dividindo em dois novos dfs, treino e teste

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

X_train.shape, X_test.shape # resultado em linhas/colunas de cada novo df
# devemos definir um modelo de classificação e foi escolhido o XGBOOST para o treinamento 

from xgboost import XGBClassifier

xgb = XGBClassifier() 



# Devemos também importar as métricas que serão utilizadas #

# As métricas são responsáveis por avaliar o desempenho do modelo

from sklearn.metrics import accuracy_score # acurácia

from sklearn.metrics import precision_score # precisão

from sklearn.metrics import recall_score # revocação

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix # matriz de confusão

from sklearn.metrics import classification_report
xgb.fit(X_train,y_train)

pred_test = xgb.predict(X_test)

print(accuracy_score(y_test,pred_test))
print(classification_report(y_test, pred_test))
conf_mat = confusion_matrix(y_true= y_test, y_pred= pred_test)

print('Confusion matrix:\n', conf_mat)

labels = ['Class 0', 'Class 1']

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Classe Predita')

plt.ylabel('Classe Real')

plt.show()
import imblearn

from imblearn.over_sampling import RandomOverSampler, SMOTE

from imblearn.under_sampling import RandomUnderSampler, TomekLinks
# RandomOverSampler

ros = RandomOverSampler()

X_ros, y_ros = ros.fit_resample(X_train,y_train)

xgb.fit(X_ros, y_ros)

preds_test = xgb.predict(X_test)

print(accuracy_score(y_test, preds_test))
print(classification_report(y_test, preds_test))
conf_mat1 = confusion_matrix(y_true=y_test, y_pred=preds_test)

print('Confusion matrix:\n', conf_mat1)

labels = ['Class 0', 'Class 1']

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(conf_mat1, cmap=plt.cm.Blues)

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Classe Predita')

plt.ylabel('Classe Real')

plt.show()
# SMOTE

sm = SMOTE()

X_sm, y_sm = sm.fit_resample(X_train,y_train)

xgb.fit(X_sm, y_sm)

p_test = xgb.predict(X_test)

print(accuracy_score(y_test, p_test))
print(classification_report(y_test, p_test))
conf_mat2 = confusion_matrix(y_true=y_test, y_pred=p_test)

print('Confusion matrix:\n', conf_mat2)

labels = ['Class 0', 'Class 1']

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(conf_mat2, cmap=plt.cm.Blues)

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Classe Predita')

plt.ylabel('Classe Real')

plt.show()
# TomekLinks

tl = TomekLinks()

X_tl,y_tl = tl.fit_resample(X_train,y_train)

y_tl.value_counts()

xgb.fit(X_tl, y_tl)

pr_test = xgb.predict(X_test)

print(accuracy_score(y_test, pr_test))
print(classification_report(y_test, pr_test))
conf_mat3 = confusion_matrix(y_true=y_test, y_pred=pr_test)

print('Confusion matrix:\n', conf_mat3)

labels = ['Class 0', 'Class 1']

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Classe Predita')

plt.ylabel('Classe Real')

plt.show()
# RandomUnderSampler

rus = RandomUnderSampler()

X_rus,y_rus = rus.fit_resample(X_train,y_train)



xgb.fit(X_rus, y_rus)

predd_test = xgb.predict(X_test)

print(accuracy_score(y_test, predd_test))
print(classification_report(y_test, predd_test))
conf_mat4 = confusion_matrix(y_true=y_test, y_pred=predd_test)

print('Confusion matrix:\n', conf_mat4)

labels = ['Class 0', 'Class 1']

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Classe Predita')

plt.ylabel('Classe Real')

plt.show()