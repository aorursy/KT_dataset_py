
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE,ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import itertools
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


###Carregando a base e verificando a forma da base. 
df = pd.read_csv ('/kaggle/input/hmeq-data/hmeq.csv')
##Verificando as dimensões da base
df.shape
###Observa-se que a base possui 5960 observações e 13 variáveis.
###Verificando os nomes das colunas
df.columns
###Verificando as variáveis na forma transposta
df.head().T
###É possivel observar as 13 variáveis e as 4 primeiras observações. 
##E já se observa a presença de NAN
###Verificando de forma aleatória para verificar outros detalhes
df.sample(5).T
###Coletando informações dos dados
df.info()
# columns of dataset
df.columns
####Nesses códigos abaixo é possível observar os valores de alguns casos 
def rstr(df): return df.shape, df.apply(lambda x: [x.unique()])
print(rstr(df))
###VERIFICANDO SE AS VARIÁVEIS DUMMIES FORAM CRIADAS
df.shape
# Observa-se um aumento no múmero de variáveis de 13 para 19.
##Agora verifiacando as variáveis dummies na base
df.sample(5).T
####AGORA, VAMOS VERIFICAR A ESTATÍSTICA DESCRITIVA DAS VARIÁVEIS 
print(df.describe().T)
## vale ressaltar que as variáveis qualitativas estão por padrão nessa análise.
#então a média e desvios-padrão não são adequados.
###############talvez retirar
###Verificando a estatística descritiva das variáveis qualitativas
# categorical features
categorical_cols = [cname for cname in df.columns if
                    df[cname].dtype in ['object']]
cat = df[categorical_cols]
cat.columns
##Verificando se existem MISSING VALUES. Foi também calculado o percentual de missing cases

feat_missing = []

for f in df.columns:
    missings = df[f].isnull().sum()
    if missings > 0:
        feat_missing.append(f)
        missings_perc = missings/df.shape[0]
        
###Verificando o percentual de missing
        print('Variable {} has {} records ({:.2%}) with missing values'.format(f, missings, missings_perc))
    
###Verificando quantas variáveis apresentam os casos faltosos

print()
print('In total, there are {} variables with missing values'.format(len(feat_missing)))
#dropping rows that have missing data
df.dropna(axis=0, how='any', inplace=True)
df.info()
##Verificando se existem MISSING VALUES. Foi também calculado o percentual de missing cases
###Confirmando se restou valor faltoso
feat_missing = []

for f in df.columns:
    missings = df[f].isnull().sum()
    if missings > 0:
        feat_missing.append(f)
        missings_perc = missings/df.shape[0]
        
###Verificando o percentual de missing
        print('Variable {} has {} records ({:.2%}) with missing values'.format(f, missings, missings_perc))
    
###Verificando quantas variáveis apresentam os casos faltosos

print()
print('In total, there are {} variables with missing values'.format(len(feat_missing)))
###Criando uma nova coluna para iniciar a previsão de forma mais acurada
df['VALUE_MORTDUE'] = df['VALUE'] - df['MORTDUE']
df.info()
###Agora verificando novamente a estatística descritiva das variáveis quantitativas
print(df.describe().T)
######DEFININDO A "TARGET" "BAD"
y=df.BAD
####
import matplotlib.pyplot as plt
ax = sns.countplot(y='BAD', data=df).set_title("Clientes inadimplentes ou não")
####Verificando a estatística das variáveis qualitativas
y = y.astype(object)
count = pd.crosstab(index = y, columns="count")
percentage = pd.crosstab(index = y, columns="frequency")/pd.crosstab(index = y, columns="frequency").sum()
pd.concat([count, percentage], axis=1)
###Verificando as outras variáveis categóricas
categorical_cols = [cname for cname in df.columns if
                    df[cname].dtype in ['object']]
cat = df[categorical_cols]
cat.columns


###Iportando bibliotecas para o modelo
import pandas
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
# GRáficos de contagem de cada categoria
sns.set( rc = {'figure.figsize': (5, 5)})
fcat = ['REASON','JOB']

for col in fcat:
    plt.figure()
    sns.countplot(x=cat[col], data=cat, palette="Set3")
    plt.show()
###Criando variáveis dummies
df_dum = pd.get_dummies(df)
###Verificando as novas variáveis dummies na base transposta
df_dum.sample(5).T
###Verificando as características das variáveis numéricas
numerical_cols = [cname for cname in df.columns if
                 df[cname].dtype in ['float']]
num = df[numerical_cols]
num.columns


###Analisando os histogramas das variáveis quantitativas
##Os histogramas abaixo demonstram distribuições assimétricas.
f, axes = plt.subplots(3,3, figsize=(20,20))
sns.distplot( df_dum["LOAN"] , color="skyblue", bins=15, kde=False, ax=axes[0, 0])
sns.distplot( df_dum["DEBTINC"] , color="olive", bins=15, kde=False, ax=axes[0, 1])
sns.distplot( df_dum["MORTDUE"] , color="orange", bins=15, kde=False, ax=axes[0, 2])
sns.distplot( df_dum["YOJ"] , color="yellow", bins=15, kde=False, ax=axes[1, 0])
sns.distplot( df_dum["VALUE"] , color="pink", bins=15, kde=False, ax=axes[1, 1])
sns.distplot( df_dum["CLAGE"] , color="gold", bins=15, kde=False, ax=axes[1, 2])
sns.distplot( df_dum["CLNO"] , color="teal", bins=15, kde=False, ax=axes[2, 1])
sns.distplot( df_dum['DEROG'], color="blue", bins=15, kde=False, ax=axes[2, 2])
sns.distplot( df_dum['DELINQ'], color="green", bins=15, kde=False, ax=axes[2, 0])
#VALIDAÇÃO CRUZADA ESTRATIFICADA + REAMOSTRAGEM
###Uma abordagem para lidar com conjuntos de dados desequilibrados é sobreamostrar a classe minoritária.
###A abordagem mais simples envolve a duplicação de exemplos na classe minoritária, embora esses exemplos não adicionem novas informações ao modelo. 
###Em vez disso, novos exemplos podem ser sintetizados a partir dos exemplos existentes.
###Esse é um tipo de aumento de dados para a classe minoritária e é chamado de Técnica de superamostragem por minoria sintética, ou SMOTE, para abreviar
##Synthetic Minority Oversampling Technique = SMOT
y = y.astype('int') 
smo = SMOTE(random_state=0)
X_resampled, y_resampled = smo.fit_resample(df_dum, y)
print(sorted(Counter(y_resampled).items()))
# Dividindo o DataFrame
from sklearn.model_selection import train_test_split
##### Treino e teste
train, test = train_test_split(df_dum, test_size=0.15, random_state=42)

# Veificando o tanho dos DataFrames
train.shape, test.shape
#######
df_dum.info()
#####features
feats = [c for c in df_dum.columns if c not in ['BAD']]

df_dum.head()
###Cross validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(rf, train[feats], train['BAD'], n_jobs=-1, cv=5)

scores, scores.mean()
####Dividindo o dataset

X_train, X_valid, y_train, y_valid = train_test_split(X_resampled, y_resampled, train_size=0.8, test_size=0.2,
                                                                random_state=0)
#####features
feats = [c for c in df_dum.columns if c not in ['BAD']]
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

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

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
###Gerando modelos da linha de base
##oPÇÕES DE TESTE E MÉTRICA DA AVALIAÇÃO
##Algoritmos de verificação

models = []
models.append(('LogisticRegression', LogisticRegression(random_state=0)))
models.append(('Bagging', BaggingClassifier(random_state=0)))
models.append(('RandomForest', RandomForestClassifier(random_state=0)))
models.append(('AdaBoost', AdaBoostClassifier(random_state=0)))
models.append(('GBM', GradientBoostingClassifier(random_state=0)))
models.append(('XGB', XGBClassifier(random_state=0)))
results_t = []
results_v = []
names = []
score = []
skf = StratifiedKFold(n_splits=5)
for (name, model) in models:
    param_grid = {}
    my_model = GridSearchCV(model,param_grid,cv=skf)
    my_model.fit(X_train, y_train)
    predictions_t = my_model.predict(X_train) 
    predictions_v = my_model.predict(X_valid)
    accuracy_train = accuracy_score(y_train, predictions_t) 
    accuracy_valid = accuracy_score(y_valid, predictions_v) 
    results_t.append(accuracy_train)
    results_v.append(accuracy_valid)
    names.append(name)
    f_dict = {
        'model': name,
        'accuracy_train': accuracy_train,
        'accuracy_valid': accuracy_valid,
    }
   #computando a matrix de confusão para o algoritmo acima
    cnf_matrix = confusion_matrix(y_valid, predictions_v)
    np.set_printoptions(precision=2)

#Gráfico da matriz de confusão não-normalizado
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=["BAD"], title="Confusion Matrix - "+str(name))
    score.append(f_dict)
plt.show()    
score = pd.DataFrame(score, columns = ['model','accuracy_train', 'accuracy_valid'])
###Gerando as pontuações
print(score)
###Modelos de base escalonadas
# Algoritmos de verificação pontual com conjunto de dados padronizado
pipelines = []
pipelines.append(('Scaled_LogisticRegression', Pipeline([('Scaler', StandardScaler()),('LogisticRegression', LogisticRegression(random_state=0))])))
pipelines.append(('Scaled_Bagging', Pipeline([('Scaler', StandardScaler()),('Bagging', BaggingClassifier(random_state=0))])))
pipelines.append(('Scaled_RandomForest', Pipeline([('Scaler', StandardScaler()),('RandomForest', RandomForestClassifier(random_state=0))])))
pipelines.append(('Scaled_AdaBoost', Pipeline([('Scaler', StandardScaler()),('AdaBoost', AdaBoostClassifier(random_state=0))])))
pipelines.append(('Scaled_GBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingClassifier(random_state=0))])))
pipelines.append(('Scaled_XGB', Pipeline([('Scaler', StandardScaler()),('XGB', XGBClassifier(random_state=0))])))
pipelines.append(('Scaled_NeuralNetwork', Pipeline([('Scaler', StandardScaler()),('NeuralNetwork', MLPClassifier(random_state=0))])))
results_t = []
results_v = []
names = []
score_sd = []
skf = StratifiedKFold(n_splits=5)
for (name, model) in pipelines:
    param_grid = {}
    my_model = GridSearchCV(model,param_grid,cv=skf)
    my_model.fit(X_train, y_train)
    predictions_t = my_model.predict(X_train) 
    predictions_v = my_model.predict(X_valid)
    accuracy_train = accuracy_score(y_train, predictions_t) 
    accuracy_valid = accuracy_score(y_valid, predictions_v) 
    results_t.append(accuracy_train)
    results_v.append(accuracy_valid)
    names.append(name)
    f_dict = {
        'model': name,
        'accuracy_train': accuracy_train,
        'accuracy_valid': accuracy_valid,
    }
    # Computing Confusion matrix for the above algorithm
    cnf_matrix = confusion_matrix(y_valid, predictions_v)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=["BAD"], title="Confusion Matrix - "+str(name))
    score_sd.append(f_dict)
plt.show()   
score_sd = pd.DataFrame(score_sd, columns = ['model','accuracy_train', 'accuracy_valid'])

###Verificando os scores dos modelos
print(score_sd)
####Aplicando do método ADASYN
y = y.astype('int') 
ada = ADASYN(random_state=0)
X_resampled_, y_resampled_ = ada.fit_resample(df_dum, y)
print(sorted(Counter(y_resampled_).items()))

# Separando o treino e validação dos dados de treino

X_train_, X_valid_, y_train_, y_valid_ = train_test_split(X_resampled_, y_resampled_, train_size=0.8, test_size=0.2,
                                                                random_state=0)
###MOdelos de linha de base
# Opções de teste e métrica da avaliação
#Algoritmos de verificação pontual
models = []
models.append(('LogisticRegression', LogisticRegression(random_state=0)))
models.append(('Bagging', BaggingClassifier(random_state=0)))
models.append(('RandomForest', RandomForestClassifier(random_state=0)))
models.append(('AdaBoost', AdaBoostClassifier(random_state=0)))
models.append(('GBM', GradientBoostingClassifier(random_state=0)))
models.append(('XGB', XGBClassifier(random_state=0)))
results_t = []
results_v = []
names = []
score = []
skf = StratifiedKFold(n_splits=5)
for (name, model) in models:
    param_grid = {}
    my_model = GridSearchCV(model,param_grid,cv=skf)
    my_model.fit(X_train_, y_train_)
    predictions_t = my_model.predict(X_train_) 
    predictions_v = my_model.predict(X_valid_)
    accuracy_train = accuracy_score(y_train_, predictions_t) 
    accuracy_valid = accuracy_score(y_valid_, predictions_v) 
    results_t.append(accuracy_train)
    results_v.append(accuracy_valid)
    names.append(name)
    f_dict = {
        'model': name,
        'accuracy_train': accuracy_train,
        'accuracy_valid': accuracy_valid,
    }
    #Matriz de confusão do algoritmo acima 
    cnf_matrix = confusion_matrix(y_valid_, predictions_v)
    np.set_printoptions(precision=2)

  #Gráfico não normalizado da matriz 
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=["BAD"], title="Confusion Matrix - "+str(name))
    score.append(f_dict)
plt.show()    
score = pd.DataFrame(score, columns = ['model','accuracy_train', 'accuracy_valid'])
##Verificando os scores
print(score)
####MODELOS DE LINHA DE BASE ESCALONADOS
# Algoritmos de verificação pontual com conjunto de dados padronizado
pipelines = []
pipelines.append(('Scaled_LogisticRegression', Pipeline([('Scaler', StandardScaler()),('LogisticRegression', LogisticRegression(random_state=0))])))
pipelines.append(('Scaled_Bagging', Pipeline([('Scaler', StandardScaler()),('Bagging', BaggingClassifier(random_state=0))])))
pipelines.append(('Scaled_RandomForest', Pipeline([('Scaler', StandardScaler()),('RandomForest', RandomForestClassifier(random_state=0))])))
pipelines.append(('Scaled_AdaBoost', Pipeline([('Scaler', StandardScaler()),('AdaBoost', AdaBoostClassifier(random_state=0))])))
pipelines.append(('Scaled_GBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingClassifier(random_state=0))])))
pipelines.append(('Scaled_XGB', Pipeline([('Scaler', StandardScaler()),('XGB', XGBClassifier(random_state=0))])))
pipelines.append(('Scaled_NeuralNetwork', Pipeline([('Scaler', StandardScaler()),('NeuralNetwork', MLPClassifier(random_state=0))])))
results_t = []
results_v = []
names = []
score_sd = []
skf = StratifiedKFold(n_splits=5)
for (name, model) in pipelines:
    param_grid = {}
    my_model = GridSearchCV(model,param_grid,cv=skf)
    my_model.fit(X_train_, y_train_)
    predictions_t = my_model.predict(X_train_) 
    predictions_v = my_model.predict(X_valid_)
    accuracy_train = accuracy_score(y_train_, predictions_t) 
    accuracy_valid = accuracy_score(y_valid_, predictions_v) 
    results_t.append(accuracy_train)
    results_v.append(accuracy_valid)
    names.append(name)
    f_dict = {
        'model': name,
        'accuracy_train': accuracy_train,
        'accuracy_valid': accuracy_valid,
    }
    #Matriz de confusão do algoritmo acima 
    cnf_matrix = confusion_matrix(y_valid_, predictions_v)
    np.set_printoptions(precision=2)

    #Gráfico não normalizado da matriz
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=["BAD"], title="Confusion Matrix - "+str(name))
    score_sd.append(f_dict)
plt.show()   
score_sd = pd.DataFrame(score_sd, columns = ['model','accuracy_train', 'accuracy_valid'])
#Verificando os scores
print(score)

# Importando o k-means
# Determinando a quantidade de clusters

# Importando o k-means
from sklearn.cluster import KMeans

# Selecionando as variaveis para utilizar no modelo.
X= df_dum[['MORTDUE','LOAN', 'YOJ']]

# Cálculo do SSE - Sum of Squared Erros
sse = []

for k in range (1, 12):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    sse.append(kmeans.inertia_)
print(sse)
###Agora verificando a sugestão de quantos cluster deve ser formados 
import matplotlib.pyplot as plt

plt.plot(range(1, 12), sse, 'bx-')
plt.title('Elbow Method')
plt.xlabel('Quantidade de Clusters')
plt.ylabel('SSE')
plt.show()
###Formando os cluster
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
cluster_id = kmeans.fit_predict(X)
cluster_id
##Salvando os resultados do dataset e  conferindo o tamanho

X['cluster_id'] = cluster_id


X.sample(10)
X.head()
##Gráfico dos cluster e os centróides
fig = plt.figure(figsize=(14,10))

plt.scatter(X.values[:,0], X.values[:,1], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='blue', marker="x", s=200)
plt.show()

