# Importando as bibliotecas



import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import metrics





# plotando os gráficos do matplotlib

%matplotlib inline
# Lendo os dados de treino

%time df_train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')

df_train.head()
df_train.shape
df_train.describe().T
# Plotando o histograma da variável 'target'

df_train.target.value_counts()
df_train.target.value_counts().plot.bar()
import missingno as msno

msno.matrix(df_train)
# Lendo os dados de teste

df_test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')

df_test.head()
df_test.describe().T
import missingno as msno

msno.matrix(df_test)
# Correlação entre os valores



corrs = df_train.corr().abs().unstack().sort_values().reset_index()

corrs = corrs[corrs['level_0'] != corrs['level_1']]

corrs.head(10)
corrs.tail(10)
# Colocando os dados na mesma escala

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



X = pd.DataFrame(scaler.fit_transform(df_train.drop(['ID_code','target' ], axis=1)))

X_test = pd.DataFrame(scaler.transform(df_test.drop(['ID_code'], axis=1)))

y = df_train.target
X.head()
# Aplicando o PCA com dois componentes

from sklearn.decomposition import PCA

# 2 para testar

pca2 = PCA(n_components = 2)

X_pca2 = pca2.fit_transform(X) # Ajustando e transformando os dados de treino
pca_df_train = pd.DataFrame(X_pca2, columns=['pca_1', 'pca_2'])

pca_df_train['target'] = df_train['target']

pca_df_train.head(10)
pca_df_train.target.value_counts()
# Balanceando as classes por undersampling e criando o dataframe 'df_train_copy'

df_train_0 = df_train[df_train.target==0]

df_train_1 = df_train[df_train.target==1]

df_train_0 = df_train_0.sample(df_train_1.shape[0], replace=True)

df_train_copy = pd.concat([df_train_0, df_train_1], ignore_index=True)
df_train_copy.target.value_counts()
# Colocando os dados na mesma escala

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



X_copy = pd.DataFrame(scaler.fit_transform(df_train_copy.drop(['ID_code','target' ], axis=1)))

y_copy = df_train_copy.target
# separando os dados do data set de teste em treino e teste

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_copy, y_copy, test_size=0.2, random_state=42, stratify=y_copy)
# Regressão Logística

pipeline_lr=Pipeline([('scalar1',StandardScaler()),

                     ('pca1',PCA(n_components=10)),

                     ('lr_classifier',LogisticRegression(random_state=0))])
# Árvore de Decisão

pipeline_dt=Pipeline([('scalar2',StandardScaler()),

                     ('pca2',PCA(n_components=10)),

                     ('dt_classifier',DecisionTreeClassifier())])
# Random Forest

pipeline_randomforest=Pipeline([('scalar3',StandardScaler()),

                     ('pca3',PCA(n_components=10)),

                     ('rf_classifier',RandomForestClassifier())])
# GaussianNB

pipeline_gnb=Pipeline([('scalar4',StandardScaler()),

                     ('pca4',PCA(n_components=10)),

                     ('gn_classifier',GaussianNB())])
## Lista com os diferentes Pipelines

pipelines = [pipeline_lr, pipeline_dt, pipeline_randomforest, pipeline_gnb]
# inicialização dos dados

best_accuracy=0.0

best_classifier=0

best_pipeline=""
# Dicionário com os pipelines para facilitar o controle

pipe_dict = {0: 'Logistic Regression', 1: 'Decision Tree', 2: 'RandomForest', 3: 'GaussianNB'}



# Execução do FIT - roda os pipelines

for pipe in pipelines:

    pipe.fit(X_train, y_train)
# resultados

for i,model in enumerate(pipelines):

    print("{} Test Accuracy: {}".format(pipe_dict[i],model.score(X_test,y_test)))
# criando uma lista com todos os modelos

classifiers = [

    KNeighborsClassifier(2),

    LogisticRegression(),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),]



plt.figure(figsize=(12,8))



# criando uma funçào para rodas o pipeline 

for clf in classifiers:

    # ajustando o modelo

    clf.fit(X_train, y_train)

    # armazenando o nome do modelo

    name = clf.__class__.__name__

    # imprimindo o nome do modelo

    print("="*30)

    print(name)

    # imprimindo os resultados

    print('****Results****')

    # fazendo predições

    # calculando as métricas

    y_pred = clf.predict(X_test)

    # imprimindo as métricas

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    print("Precision:", metrics.precision_score(y_test, y_pred))

    print("Recall:", metrics.recall_score(y_test, y_pred))

    

    

    

    # plotando a curva ROC

    y_pred_proba = clf.predict_proba(X_test)[::,1]

    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    plt.plot(fpr,tpr,label=name+", auc="+str(auc))

    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')

    plt.legend(loc=4)
X_test_sub = pd.DataFrame(scaler.transform(df_test.drop(['ID_code'], axis=1)))
clf = GaussianNB()

y_pred_gnb = clf.fit(X_train, y_train).predict(X_test_sub)
sub_df = pd.DataFrame({"ID_code":df_test["ID_code"].values})

sub_df["target"] = y_pred_gnb

sub_df.to_csv("submission1.csv", index=False)