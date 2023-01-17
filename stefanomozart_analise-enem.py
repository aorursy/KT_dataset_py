!pip install pingouin
# Carregamos as bibliotecas que serão utilizadas para manipulação e visualização dos dados
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pingouin as pg
import seaborn as sns
# Criamos um pipeline de pré-processamento. A ideia é utilizar essa função para microdados de
# diferentes anos 
def pipeline_notas_Enem(arquivo):
    # Colunas a serem lidas no arquivo
    features = [
        'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT','NU_NOTA_REDACAO','TP_ESCOLA'
    ]
    
    # Lemos o arquivo, retirando os registros em que um dos valores não estivesse presente.
    df = pd.read_csv(
        arquivo,
        #nrows = 5000, # 5k linhas para desenvolvimento inicial
        encoding = 'latin1',
        usecols = features,
        sep = ';'
    ).dropna()
    
    df['NOTA_MEDIA'] = (df['NU_NOTA_CN'] + df['NU_NOTA_CH'] + df['NU_NOTA_LC'] + df['NU_NOTA_MT']) / 4
    
    # Filtramos os registros de com alunos de escolas públicas e privadas (valores 2 e 3 no 
    # campo TP_ESCOLA)
    df = df.loc[df['TP_ESCOLA'].isin([2, 3])]
    df.loc[df['TP_ESCOLA']==2, 'TP_ESCOLA'] = 'Pública'
    df.loc[df['TP_ESCOLA']==3, 'TP_ESCOLA'] = 'Privada'
    
    return df

# Carregamos o dataset a partir da cópia do Kaggle
notas = pipeline_notas_Enem('/kaggle/input/enem-2019/DADOS/MICRODADOS_ENEM_2019.csv')

notas.head()
notas.TP_ESCOLA.value_counts().plot(kind='bar')
plt.show()
notas[['TP_ESCOLA', 'NU_NOTA_REDACAO']].groupby('TP_ESCOLA').describe()
pub = notas.loc[notas.TP_ESCOLA=='Pública', 'NU_NOTA_REDACAO']
priv = notas.loc[notas.TP_ESCOLA=='Privada', 'NU_NOTA_REDACAO']

import warnings
warnings.filterwarnings('ignore')

pg.ttest(priv, pub)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))

p1=sns.distplot(
    pub,
    ax=axes[0],
    axlabel=f'Média: {pub.mean():.2f}\nDesvio padrao: {pub.std():.2f}'
).set_title("Notas de alunos de escola pública")
p2=sns.distplot(
    priv,
    axlabel=f'Média: {priv.mean():.2f}\nDesvio padrao: {priv.std():.2f}'
).set_title("Notas de alunos de escola privada")
plt.show()
notas[['TP_ESCOLA', 'NOTA_MEDIA']].groupby('TP_ESCOLA').describe()
pub = notas.loc[notas.TP_ESCOLA=='Pública', 'NOTA_MEDIA']
priv = notas.loc[notas.TP_ESCOLA=='Privada', 'NOTA_MEDIA']
pg.ttest(x=priv, y=pub, correction=False).round(2)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))

p1=sns.distplot(
    pub,
    ax=axes[0],
    axlabel=f'Média: {pub.mean():.2f}\nDesvio padrao: {pub.std():.2f}'
).set_title("Nota de alunos de escola pública")
p2=sns.distplot(
    priv,
    axlabel=f'Média: {priv.mean():.2f}\nDesvio padrao: {priv.std():.2f}'
).set_title("Nota de alunos de escola privada")
plt.show()
std_features = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO', 'NOTA_MEDIA']
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
notas[std_features] = std.fit_transform(notas[std_features])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    notas.drop(['TP_ESCOLA', 'NOTA_MEDIA'], axis = 1), notas.TP_ESCOLA, test_size=0.2, random_state=42
)
# Instanciamos o modelo KNN com dois núcleos
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2)

# Definimos uma função para treinamento e exbição dos resultados de um modelo
def pipeline_treino_teste(model):
    # Ajustamos o modelo
    model.fit(X_train, y_train)
    # Submetemos os dados de teste ao classificador 
    y_pred = model.predict(X_test)
    
    # E observamos algumas métricas de desempenho desse modelo: acurácia, F1-score e matriz de confusão
    from sklearn import metrics
    print(f'Acurácia: {metrics.accuracy_score(y_test, y_pred)}')
    print(f'F1-score médio: {metrics.f1_score(y_test, y_pred, average="weighted")}')
    print(f"F1-score da classe minoritária: {metrics.f1_score(y_test, y_pred, pos_label='Privada')}")
    metrics.plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues, normalize='true')
    
# Agora executamos o pipeline
pipeline_treino_teste(knn)
# Instanciamos um modelo SGD (descida do gradiente estocástica)
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(class_weight="balanced", loss='modified_huber', penalty="elasticnet", random_state=42)

# Executamos o pipeline de treino e teste
pipeline_treino_teste(sgd)
# Instanciamos o modelo de regressão logística
from sklearn.linear_model import LogisticRegression
rlog = LogisticRegression(class_weight="balanced")

# E submetemos ao pipeline de treino e teste
pipeline_treino_teste(rlog)
# Instanciamos uma árvore de decisão, com altura máxima de 3 nós
from sklearn.tree import DecisionTreeClassifier, plot_tree
tree = DecisionTreeClassifier(class_weight="balanced", max_depth=7, random_state=42)

# Executamos o pipeline de treino e teste
pipeline_treino_teste(tree)
# Instanciamos o classificador Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(class_weight="balanced", n_estimators=300, random_state=42)

# Executamos o pipeline de treino e teste
pipeline_treino_teste(rf)
# Instanciamos o classificador SVM
from sklearn.svm import LinearSVC
lsvm = LinearSVC(class_weight="balanced", max_iter=3000, random_state=42, tol=5e-4)

# Submetemos o classificador ao treino e teste
pipeline_treino_teste(lsvm)
from xgboost import XGBClassifier
xgb = XGBClassifier(
    objective = 'multi:softmax',
    booster = 'gbtree',
    num_class = 2,
    eval_metric = 'logloss',
    eta = .1,
    max_depth = 14,
    colsample_bytree = .4,
    n_jobs=-1
)

pipeline_treino_teste(xgb)
# Instanciamos o classficador por votos, passando os classificadores já treinados
from mlxtend.classifier import EnsembleVoteClassifier

vote = EnsembleVoteClassifier(
    clfs=[knn, sgd, rlog, tree, rf, lsvm, xgb],
    weights=[1, 1, 1, 1, 1, 1, 1],
    refit=False
)

# Submetemos esse classificador ao teste
pipeline_treino_teste(vote)
from fastai.tabular import *

dep_var = 'TP_ESCOLA'
cont_names = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO', 'NOTA_MEDIA']

start = int(len(notas)*.7)
end = int(len(notas)*.1) + start

              
test = TabularList.from_df(notas.iloc[start:end], cont_names=cont_names)

data = (
    TabularList
        .from_df(notas, cont_names=cont_names)
        .split_by_idx(list(range(start,end)))
        .label_from_df(cols=dep_var)
        .add_test(test)
        .databunch()
)

data.show_batch(rows=10)
learn = tabular_learner(data, layers=[200,100], metrics=accuracy)
learn.fit_one_cycle(1, 5e-3)
ClassificationInterpretation.from_learner(learn).plot_confusion_matrix(normalize=True)
nX = notas[['NOTA_MEDIA', 'NU_NOTA_REDACAO']]
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ny = le.fit_transform(notas.TP_ESCOLA)
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools

gs = gridspec.GridSpec(2, 2)
fig = plt.figure(figsize=(10,8))

labels = ['K-nearest Neighbour', 'Stochastic Gradient Descent', 'Logistic Regression', 'Decision Tree']

for clf, lab, grd in zip([knn, sgd, rlog, tree],
                         labels,
                         itertools.product([0, 1], repeat=2)):
    clf.fit(nX, ny)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=nX.to_numpy(), y=ny, clf=clf)
    plt.title(lab)
# Criamos um pipeline de pré-processamento. A ideia é utilizar essa função para microdados de
# diferentes anos 
def pipeline_SocioEconomico_Enem(arquivo):
    # Colunas a serem lidas no arquivo
    features = [
        'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO',  'TP_ESCOLA', 
        'TP_ENSINO', 'SG_UF_ESC', 'TP_COR_RACA', 'TP_SEXO', 'Q001', 'Q002', 'Q005', 'Q006', 'Q007',
        'NU_IDADE'
    ]

    # Carregamos o dataset a partir do arquivo
    df = pd.read_csv(
        arquivo,
        nrows = 5000, # 5k linhas para desenvolvimento inicial
        encoding = 'latin1',
        usecols = features,
        sep = ';'
    )#.dropna()
    
    # Filtramos os registros de com alunos de escolas públicas e privadas (valores 2 e 3 no campo TP_ESCOLA)
    df = df.loc[df['TP_ESCOLA'].isin([2, 3])]
    df.loc[df['TP_ESCOLA']==2, 'TP_ESCOLA'] = 'Pública'
    df.loc[df['TP_ESCOLA']==3, 'TP_ESCOLA'] = 'Privada'

    # Vamos atribuir o tipo de ensino com base na idade do aluno
    df = df.loc[df['NU_IDADE'].notna()]
    df.loc[df['TP_ENSINO'].isna() & df['NU_IDADE']>21, 'TP_ENSINO'] = 3
    df.loc[df['TP_ENSINO'].isna(), 'TP_ENSINO'] = 1
    
    # Filtramos os demais valores ausentes
    df.dropna(inplace=True)
    
    # Realizamos uma normalização dos valores das notas
    std_features = ['NOTA_MEDIA', 'NU_NOTA_REDACAO']
    df['NOTA_MEDIA'] = (df['NU_NOTA_CN'] + df['NU_NOTA_CH'] + df['NU_NOTA_LC'] + df['NU_NOTA_MT']) / 4
    
    
    from sklearn.preprocessing import StandardScaler
    std = StandardScaler()
    df[std_features] = std.fit_transform(df[std_features])
    
    # Usamos um encoder ordinal para transformar os valores presentes na coluna 'Q006' (faixa de renda) em valores 
    # numéricos, crescentes.
    ord_enc_features = ['Q001', 'Q002', 'Q001']
    from sklearn.preprocessing import OrdinalEncoder
    ord_enc = OrdinalEncoder()
    df[ord_enc_features] = ord_enc.fit_transform(df[ord_enc_features])

    ## - Colunas que passarão por um processo de codificação
    #onehot_enc_features = ['TP_COR_RACA', 'TP_SEXO', 'TP_ENSINO', 'SG_UF_ESC']
    #from sklearn.preprocessing import OneHotEncoder
    #onehot_enc = OneHotEncoder()
    #df.enc = onehot_enc.fit_transform(df[onehot_enc_features])

    # Retira as colunas usadas para cálculos intermediários
    return df.drop(['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT'], axis = 1)

# Carregamos o dataset a partir da cópia do Kaggle, retirando os registros em que um dos valores não estivesse presente.
se = pipeline_SocioEconomico_Enem('./dados/MICRODADOS_ENEM_2019.csv')

se.head()
ord_enc_features = ['Q005', 'Q006', 'Q007']
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
se[ord_enc_features] = enc.fit_transform(se[ord_enc_features])
se.head()
onehot_enc_features = ['TP_ESCOLA', 'TP_COR_RACA', 'TP_SEXO', 'TP_ENSINO', 'SG_UF_ESC']
se = pd.get_dummies(se, prefix=onehot_enc_features, columns=onehot_enc_features, drop_first=True)
se.head()
X_train, X_test, y_train, y_test = train_test_split(
    se.drop(['NOTA_MEDIA'], axis = 1), se.NOTA_MEDIA, test_size=0.2, random_state=42
)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, lr.predict(X_test))
X_train, X_test, y_train, y_test = train_test_split(
    se.drop(['NU_NOTA_REDACAO'], axis = 1), se.NU_NOTA_REDACAO, test_size=0.2, random_state=42
)

lr2 = LinearRegression()
lr2.fit(X_train, y_train)

mean_squared_error(y_test, lr.predict(X_test))