from sklearn.ensemble import RandomForestClassifier



from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO 

from IPython.display import Image 

#from pydot import graph_from_dot_data

import pandas as pd

import numpy as np



import sklearn.tree 

import time



import warnings

warnings.filterwarnings("ignore")



# importando as bilbiotecas

import seaborn as sns



#importando o modelo de Regressão Logística

from sklearn.linear_model import LogisticRegression



import matplotlib.pyplot as plt



# importando a biblioteca de métricas

from sklearn import metrics
df = pd.read_csv('/kaggle/input/paysim1/PS_20174392719_1491204439457_log.csv')

df.head(10)
df.columns
df = df[['isFraud','isFlaggedFraud','step','type','amount','nameOrig','oldbalanceOrg','newbalanceOrig',

         'nameDest','oldbalanceDest','newbalanceDest']]
df.head(10)
df.describe()
df.describe().T
df.shape
df.info()
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()



df = df.apply(label_encoder.fit_transform)



df.head(10)
df.info()
# verificando o tamanho das classes por groupby

df.groupby('isFraud').step.count()
import seaborn as sns
# dividindo o DataSet em dois componentes

# X: todas as variáveis exceto class

# Y: variável target (isFraud)



X = df.iloc[:, 1:11].values 

y = df.iloc[:, 0].values 
# dividindo os dados em treino e teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# faz o dummie para transformar se é fraude ou nõ em 0 ou 1



y = pd.get_dummies(y)
# instanciando o modelo

clf = LogisticRegression()



# ajustando o modelo com os dados de treino

clf.fit(X_train, y_train)



# fazendo predições com os dados de teste

y_pred = clf.predict(X_test)



# imprimindo as principais métricas

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))

print("F1:",metrics.f1_score(y_test, y_pred))
# plotando uma matriz de confusão

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

cnf_matrix



# TN FP

# FN TP
# plotando a curva ROC

y_pred_proba = clf.predict_proba(X_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

auc = metrics.roc_auc_score(y_test, y_pred_proba)



plt.rcParams['figure.figsize'] = (12., 8.)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')

plt.legend(loc=4)
df.columns
# definindo variáveis para cada uma das classes

df_0 = df[df.isFraud==0]

df_1 = df[df.isFraud==1]
# verificando o desbalanceamento

len(df_0),len(df_1)
# fazendo um undersampling da classe com output zero (em maior número)

df_0=df_0.sample(n=8213)

len(df_0)
# concatenando os dois DataSets com o mesmo tamanho

df = pd.concat([df_0,df_1])

df.isFraud.value_counts()
df.columns
feature_cols = ['isFlaggedFraud', 'step', 'type', 'amount', 'nameOrig',

       'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest',

       'newbalanceDest']



to_sc = df[feature_cols] 
# "Standalizando"



from sklearn.preprocessing import StandardScaler



sc = StandardScaler()



sc_df = sc.fit_transform(to_sc)
# ignorando os warnings

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)





X = sc_df



# variável target

y = df.isFraud



# dividindo oa dados em treino e teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



# importnado as bibliotecas com os modelos classificadores

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier



# definindo uma lista com todos os classificadores

classifiers = [

    KNeighborsClassifier(3),

    GaussianNB(),

    LogisticRegression(),

    #SVC(),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    GradientBoostingClassifier(),

    MLPClassifier()]



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

    print("F1:", metrics.f1_score(y_test, y_pred))

    

    # plotando a curva ROC

    y_pred_proba = clf.predict_proba(X_test)[::,1]

    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    plt.plot(fpr,tpr,label=name+", auc="+str(auc))

    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')

    plt.legend(loc=4)
# instanciando o modelo

clf = LogisticRegression()



# ajustando o modelo com os dados de treino

clf.fit(X_train, y_train)



# fazendo predições com os dados de teste

y_pred = clf.predict(X_test)



# imprimindo as principais métricas

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))

print("F1:",metrics.f1_score(y_test, y_pred))
# plotando uma matriz de confusão

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

cnf_matrix



# TN FP

# FN TP
# importando a biblioteca

from sklearn import tree

# instanciando e ajustando o modelo

clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=3,min_samples_split=10)

clf = clf.fit(X,y)



# fazendo predições

y_pred = clf.predict(X_test)

# calculando e imprimindo as métricas

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("Precision:", metrics.precision_score(y_test, y_pred))

print("Recall:", metrics.recall_score(y_test, y_pred))

print("F1:",metrics.f1_score(y_test, y_pred))
# plotando uma matriz de confusão

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

cnf_matrix



# TN FP

# FN TP
df = pd.read_csv('/kaggle/input/paysim1/PS_20174392719_1491204439457_log.csv')

df.head()
df.info()
# apagando as Colunas nameOrig e nameDest



df.drop(["nameOrig", "nameDest"], axis = 1, inplace = True)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()



# applying label encoder to whole dataset...

df = df.apply(label_encoder.fit_transform)



# checking the result

df.head()
# verificando o tamanho das classes por groupby

df.groupby('isFraud').step.count()
df.isFraud.value_counts()
X = df.drop("isFraud", axis = 1)

y = df.isFraud
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, stratify=y)

y_train.value_counts()
from sklearn.tree import DecisionTreeClassifier



# importando a biblioteca

from sklearn import tree



# instanciando o modelo

clf = tree.DecisionTreeClassifier(criterion='gini',max_depth=100,min_samples_split=10)



# ajustando o modelo com os dados de treino

clf = clf.fit(X_train, y_train)



# fazendo predições com os dados de teste

y_pred = clf.predict(X_test)



# imprimindo as principais métricas

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))

print("F1:",metrics.f1_score(y_test, y_pred))
# plotando uma matriz de confusão

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

cnf_matrix



# TN FP

# FN TP
#!pip install -U scikit-learn scipy matplotlib



#!pip install imblearn



#!pip install -U scikit-learn



#!pip install -U imbalanced-learn



#!pip install tensorflow



import imblearn

from imblearn import undersampling, oversampling
from imblearn.over_sampling import SMOTE

from imblearn.oversampling import SMOTENC
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, stratify=y)
smt = SMOTE()

X_train, y_train = smt.fit_sample(X_train, y_train)

np.bincount(y_train)
from sklearn.tree import DecisionTreeClassifier



# importando a biblioteca

from sklearn import tree



# instanciando o modelo

clf = tree.DecisionTreeClassifier(criterion='gini',max_depth=100,min_samples_split=10)



# ajustando o modelo com os dados de treino

clf = clf.fit(X_train, y_train)



# fazendo predições com os dados de teste

y_pred = clf.predict(X_test)



# imprimindo as principais métricas

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))

print("F1:",metrics.f1_score(y_test, y_pred))
# plotando uma matriz de confusão

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

cnf_matrix



# TN FP

# FN TP
from imblearn.under_sampling import NearMiss
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, stratify=y)
nr = NearMiss()

X_train, y_train = nr.fit_sample(X_train, y_train)

np.bincount(y_train)
# instanciando o modelo

clf = tree.DecisionTreeClassifier(criterion='gini',max_depth=100,min_samples_split=10)



# ajustando o modelo com os dados de treino

clf = clf.fit(X_train, y_train)



# fazendo predições com os dados de teste

y_pred = clf.predict(X_test)



# imprimindo as principais métricas

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))

print("F1:",metrics.f1_score(y_test, y_pred))
# plotando uma matriz de confusão

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

cnf_matrix



# TN FP

# FN TP