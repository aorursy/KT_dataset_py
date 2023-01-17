# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np



import matplotlib.pyplot as pyplot



from sklearn.preprocessing import StandardScaler ## Talvez não utilizemos isto



from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import metrics

from sklearn.metrics import roc_curve, auc
df = pd.read_excel('../input/covid19/dataset.xlsx')
df.to_csv ('dataset.csv', index = None, header=True)
df.shape
pd.set_option("display.max_columns", 120)
df.head()
df.dtypes
df.describe()
df["SARS-Cov-2 exam result"].value_counts(dropna= 'Positive')
df['SARS-Cov-2 exam result'].value_counts().plot.barh()
pd.set_option("display.max_rows", 200)

df.isnull().sum()

df = df.drop(['Patient ID'], axis = 1)
df_numeric = df.select_dtypes(include=[np.number]) 

numericas = list(df_numeric)
# Nao podemos desprezar  dados devido ao poucos registros disponivieis

# selecionando as colunas numéricas e preenchendo com a média

df[numericas] = df[numericas].fillna(df.mean())
# substitui os dados faltantes por 'na'

df = df.fillna('na') 

# se sobrar alguma linha (nao deveria), dropamos essas linhas.

df = df.dropna() 

# verificando o tamanho do DataFrame

df.shape
df.isnull().sum()
df.head(50)
df['Patient Age Quantile'] = df['Patient age quantile'].astype("category")
# aplicação do get_dummies do pandas para dummização de variáveis categóricas

df = pd.get_dummies(df, drop_first=True)
# verificação do tamamno do DataSet

df.shape

df.head()
df = df.drop(['Patient age quantile'], axis = 1)
df.dtypes
X=df
y = X['SARS-Cov-2 exam result_positive']

X= X.drop(['SARS-Cov-2 exam result_positive'],axis=1)
# importando a biblioteca de split de dados do Sklearn

from sklearn.model_selection import train_test_split

# separando os dados em treino e teste

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
# import logistic regression model and accuracy_score metric

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score
# instanciando o modelo

clf = LogisticRegression()



# ajustando o modelo com os dados de treino

clf.fit(X_train, y_train)



# fazendo predições com os dados de teste

y_pred = clf.predict(X_test)



# imprimindo as principais métricas

print("Accuracy:",accuracy_score(y_test, y_pred))

print("Precision:",precision_score(y_test, y_pred))

print("Recall:",recall_score(y_test, y_pred))

print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))

print('Confusion Matrix ')

# importando a biblioteca de métricas

from sklearn import metrics

# plotando uma matriz de confusão

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

cnf_matrix
# importando as bibliotecas com os modelos classificadores

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt

# definindo uma lista com todos os classificadores

classifiers = [

    KNeighborsClassifier(3),

    GaussianNB(),

    SVC(),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    GradientBoostingClassifier()]





# definindo o tamanho da figura para o gráfico

plt.figure(figsize=(12,8))



# rotina para instanciar, predizer e medir os rasultados de todos os modelos

for clf in classifiers:

    # instanciando o modelo

    clf.fit(X_train, y_train)

    # armazenando o nome do modelo na variável name

    name = clf.__class__.__name__

    # imprimindo o nome do modelo

    print("="*30)

    print(name)

    # imprimindo os resultados do modelo

    print('****Results****')

    y_pred = clf.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    print("Precision:", metrics.precision_score(y_test, y_pred))

    print("Recall:", metrics.recall_score(y_test, y_pred))

    print ('F1 score:', f1_score(y_test,y_pred,average='weighted'))

    # plotando uma matriz de confusão

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

    print (cnf_matrix)
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2200, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

print(random_grid)
# Use the random grid to search for best hyperparameters

# First create the base model to tune

model = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

model = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

model.fit(X_train, y_train)
model.best_params_
model.best_estimator_
model_random = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,

                       criterion='gini', max_depth=20, max_features='sqrt',

                       max_leaf_nodes=None, max_samples=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, n_estimators=1088,

                       n_jobs=None, oob_score=False, random_state=None,

                       verbose=0, warm_start=False)

model_random.fit(X_train, y_train)

name = model_random.__class__.__name__

#imprimindo os resultados do modelo

print('****Results****')

y_pred = model_random.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("Precision:", metrics.precision_score(y_test, y_pred))

print("Recall:", metrics.recall_score(y_test, y_pred))

print ('F1 score:', f1_score(y_test,y_pred,average='weighted'))

#plotando uma matriz de confusão

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

print (cnf_matrix)
cabecalho = ["campos_originais", "importancia"]

ls_imp = sorted(zip(X_train.columns, model_random.feature_importances_), key=lambda x: x[1] * 1)

df_imp = pd.DataFrame(ls_imp, columns=cabecalho)
df_imp["importancia"].sum()
df_imp.head(204)
df_imp.isnull().sum()
df_importancia = df_imp.groupby("campos_originais", as_index=False)["importancia"].sum()
df_importancia = df_importancia.sort_values(by="importancia")
df_importancia = df_importancia[df_importancia.importancia >= 0.015]
pyplot.barh(df_importancia["campos_originais"], df_importancia["importancia"], color="r", align="center")
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

scores = cross_val_score(model_random, X_train, y_train, cv=10, scoring ='accuracy')

list(scores)
scores.mean()
y_pred_proba = model.predict_proba(X_test)[::,1] 

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label=name+", auc="+str(auc))

plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')

plt.legend(loc=4)
# Cálculo KS

y_pred_proba = model_random.predict_proba(X_test)

y_pred_proba[0:100]
df_test = pd.DataFrame(X_test)

df_test["NoRisk"] = 1 - y_test

df_test["Risk"] = y_test

df_test["prob_0"] = y_pred_proba[:,0]

df_test["prob_1"] = y_pred_proba[:,1]
df_test["prob_1"].plot(kind="hist", bins=50)
df_test["decil"] = pd.qcut(df_test["prob_1"], 10, labels=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
decil_agrupado = df_test.groupby("decil", as_index=False)

df_decil = pd.DataFrame(decil_agrupado.min().decil)
df_decil
df_decil["Prediction"] = decil_agrupado.mean().prob_1

df_decil["min_scr"] = decil_agrupado.min().prob_1

df_decil["max_scr"] = decil_agrupado.max().prob_1

df_decil["NoRisk"] = decil_agrupado.sum().NoRisk

df_decil["Risk"] = decil_agrupado.sum().Risk

df_decil["total"] = df_decil["Risk"] + df_decil["NoRisk"]
df_decil
df_decil = (df_decil.sort_values(by="min_scr", ascending=False)).reset_index(drop=True)

df_decil
df_decil["odds"] = df_decil["NoRisk"] / df_decil["Risk"]



df_decil["Risk_rate"] = df_decil["Risk"] / df_decil["total"]



df_decil["ks"] = np.round(((df_decil["Risk"] / df_test["Risk"].sum()).cumsum() - \

                              (df_decil["NoRisk"] / df_test["NoRisk"].sum()).cumsum()), 4) * 100



df_decil["max_ks"] = df_decil["ks"].apply(lambda x: "**" if x == df_decil["ks"].max() else "")



df_decil
pyplot.plot(df_decil["decil"].astype(int), df_decil["Prediction"], marker="", color="orange", linewidth=4, label="Prediction")

pyplot.plot(df_decil["decil"].astype(int), df_decil["Risk_rate"], marker="o", markerfacecolor="blue", markersize=6, \

            color="blue", linewidth=2, label="Real")

pyplot.xlabel("Decil")

pyplot.legend()