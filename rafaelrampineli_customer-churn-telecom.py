# Biblioteca para manipulação de Dados
import pandas as pd

# Biblioteca para plotar os gráficos
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Biblioteca para operação matemáticas de matrizes e arrays
import numpy as np

# Biblioteca para filtrar warnings e não apresentar na tela
import warnings
warnings.filterwarnings("ignore")

# Biblioteca utilizada durante as operações de Feature Selection e Treinamento do Modelo
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

# Biblioteca utilizada durante a padronização dos dados
from sklearn.preprocessing import StandardScaler

# Biblioteca utilizada para realizar o balanceamento dos dados
#!pip install imblearn (instalação do pacote caso não exista)
from imblearn.over_sampling import SMOTE

# Bibilioteca utilizada durante o split dos dados em treino e teste
from sklearn.model_selection import train_test_split

# Biblioteca utilizada para avaliação do modelo criado
from sklearn import metrics

# Biblioteca utilizada para realizar o cross-validation com os dados teste
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Biblioteca utilizada para otimização de hyper-parametros
from sklearn.model_selection import GridSearchCV
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Carregando o dataset de treino utilizando Pandas
ds_train = pd.read_csv("/kaggle/input/projeto4_telecom_treino.csv", sep = ",")
# Visualizando os primeiros 15 registros do Dataset.
ds_train.head(15)
# Visualizando as informações do Dataset
ds_train.info()
# Removendo a Coluna {Unnamed: 0} que está caracterizando a sequência de registros do Dataset.
ds_train = ds_train.drop('Unnamed: 0', axis=1)
# Verificando os valores distintos de cada coluna do dataset
for i in ds_train.columns:
    x = ds_train[i].unique()
    print(x)
# Função que será utilizada para transforma algumas variáveis string em Numéricas através de um mapeamento de dict.
def transformColumns(dataset):
    
    # Transforma as strings de valores Yes/No em 1/0 e atribui em um nvo campo, exceto a variável target Churn
    dictmapYesNo = {"yes": 1, "no": 0}
    
    dataset['international_plan_num'] = dataset['international_plan'].map(dictmapYesNo)
    dataset['voice_mail_plan_num'] = dataset['voice_mail_plan'].map(dictmapYesNo)    
    dataset['churn'] = dataset['churn'].map(dictmapYesNo)
        
    # Remove as strings dos códigos de Area, preservando somente o código em um novo campo
    dictmapAreaCode = {"area_code_415": 0, "area_code_408": 1, "area_code_510": 2}

    dataset['area_code_num'] = dataset['area_code'].map(dictmapAreaCode)
    
    # Transforma o valor dos status em Numeros em um novo campo
    dictmapState = {'KS': 1000, 'OH': 1001, 'NJ': 1002, 'OK': 1003, 'AL': 1004, 'MA': 1005, 
                    'MO': 1006, 'LA': 1007, 'WV': 1008, 'IN': 1009, 'RI': 1010, 
                    'IA': 1011, 'MT': 1012, 'NY': 1013, 'ID': 1014, 'VT': 1015, 
                    'VA': 1016, 'TX': 1017, 'FL': 1018, 'CO': 1019, 'AZ': 1020, 
                    'SC': 1021, 'NE': 1022, 'WY': 1023, 'HI': 1024, 'IL': 1025, 
                    'NH': 1026, 'GA': 1027, 'AK': 1028, 'MD': 1029, 'AR': 1030, 
                    'WI': 1031, 'OR': 1032, 'MI': 1033, 'DE': 1034, 'UT': 1035, 
                    'CA': 1036, 'MN': 1037, 'SD': 1038, 'NC': 1039, 'WA': 1040, 
                    'NM': 1041, 'NV': 1042, 'DC': 1043, 'KY': 1044, 'ME': 1045, 
                    'MS': 1046, 'TN': 1047, 'PA': 1048, 'CT': 1049, 'ND': 1050
                   }
    
    dataset['state_num'] = dataset['state'].map(dictmapState)
  

    # Transforma as variáveis que eram strings e foram manipuladas em Categóricas
    dataset['international_plan'] = dataset['international_plan'].astype('category')
    dataset['voice_mail_plan']    = dataset['voice_mail_plan'].astype('category')    
    dataset['area_code']          = dataset['area_code'].astype('category')
    dataset['state']              = dataset['state'].astype('category')

    # Reordenando as colunas para que a coluna TARGET (churn) seja a última coluna do Dataframe
    dataset = dataset[[col for col in dataset if col not in ['churn']] + ['churn']]    
    
    return dataset
ds_train = transformColumns(ds_train)
ds_train.info()
ds_train.head(10)
ds_train.describe()
# Verificando se existem valores NULL
ds_train.isnull().values.any()
# Analisando como está a distribuição dos Dados na variável churn (nossa TARGET). 
# 0 -> Não 1 -> Sim
ds_train.groupby('churn').size()
Count_No = len(ds_train[ds_train['churn']==0])
Count_Yes = len(ds_train[ds_train['churn']==1])

print ("Percentual de Clientes que Cancelaram: ", round((Count_Yes / (Count_Yes+Count_No))*100,2))
print ("Percentual de Clientes que Não Cancelaram: ", round((Count_No / (Count_Yes+Count_No))*100,2))
columns = ds_train.select_dtypes(exclude='category').columns

for i in columns:
    #ds_train[i].plot(kind = 'hist')
    sns.distplot(ds_train[i], rug=True)
    plt.title('Histograma Variável: ' + i)
    plt.xlabel(i)    
    plt.show()
cols = [col for col in ds_train.columns if col not in ['international_plan_num','voice_mail_plan_num','area_code_num','state_num','churn']]
for i in cols:
    pd.crosstab(ds_train[i], ds_train.churn).plot(kind = 'bar')
    plt.show()
# Removendo os dados categóricos
columns = ds_train.select_dtypes(exclude='category').columns

for i in columns:
    #ds_train[i].plot(kind = 'box')
    sns.boxplot(ds_train[i])
    plt.title('BoxPlots Variável: ' + i)  
    plt.show()
ds_train.corr()
correlations = ds_train.corr()

fig = plt.figure() # Cria uma figura em branco
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin = -1, vmax = 1) #Mostrar as correlações
fig.colorbar(cax)
# Definir o tamanho do array. Tamanho escolhido com base no # de variaveis
ticks = np.arange(1, 20, 1) 
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(ds_train.columns)
ax.set_yticklabels(ds_train.columns)
plt.show()
ds_train.corr()['churn']
# Remove as colunas Categóricas.
ds_train_FeatureSelect = ds_train.drop(['voice_mail_plan','area_code','state','international_plan'], axis=1)

# Transforma o Dataframe do Pandas em um Array Numpy (Array é um formato esperado para utilização do RFE)
ds_train_array = ds_train_FeatureSelect.values

# Separando os dados em inputs(features) e outputs (target)
features_x = ds_train_array[:,:-1] # selecionando todas as colunas exceto a ultima (que é a TARGET)
target_y = ds_train_array[:,19:] # selecionando somente a ultima coluna Target

# Criando o modelo utilizando Regressão Logistica
LogReg = LogisticRegression()

# Aplicando o modelo RFE
rfe = RFECV(LogReg, cv=4, scoring='accuracy')
fit = rfe.fit(features_x, target_y)

plt.figure()
plt.xlabel("Numero de Features Seleciondas")
plt.ylabel("Cross validation score (# de classificações corretas)")
plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
plt.show()

# Print dos resultados
print("Variáveis Preditoras:", ds_train_FeatureSelect.columns[:-1])
print("Variáveis Selecionadas: %s" % fit.support_)
print("Ranking dos Atributos: %s" % fit.ranking_)
print("Número de Melhores Atributos: %d" % fit.n_features_)

del ds_train_FeatureSelect
del features_x
del target_y
del ds_train_array
# Selecionando somente as colunas escolhidas pelo algoritmo RFECV como mais relevantes e melhor acurácia para treinamento.
def SelectVariablesRFECV(dataset):
    datasetReturn = dataset[['total_intl_calls', 'total_intl_charge','number_customer_service_calls', 'international_plan_num',
                                                       'voice_mail_plan_num', 'churn']]
    
    return datasetReturn
ds_train_Selected = SelectVariablesRFECV(ds_train)
ds_train_Selected.describe()
columns = [col for col in ds_train_Selected.columns if col not in ['churn']]
for i in columns:
    #ds_train_Selected[i].plot(kind = 'hist')
    sns.distplot(ds_train_Selected[i], rug=True)
    plt.title('Histograma Variável: ' + i)
    plt.xlabel(i)    
    plt.show()
columns = [col for col in ds_train_Selected.columns if col not in ['churn']]

for i in columns:
    #ds_train_Selected[i].plot(kind = 'box')
    sns.boxplot(data=ds_train_Selected[i])
    plt.title('BoxPlots Variável: ' + i)  
    plt.show()
def standardizationData(dataset):
    # Transforma o Dataframe do Pandas em um Array Numpy (Array é um formato esperado para utilização do StandardScale)
    ds_train_array = dataset.values

    # Separando os dados em inputs(features) e outputs (target)
    features_x = ds_train_array[:,:-1] # selecionando todas as colunas exceto a ultima (que é a TARGET)
    target_y = ds_train_array[:,5:] # selecionando somente a ultima coluna Target

    # Gerando os novos dados Padronizados
    StScaler = StandardScaler(with_mean=False, with_std=False).fit(features_x)
    ds_train_return = StScaler.transform(features_x) # Transformamos somente as variáveis preditoras.

    # Nomeia as colunas
    df_features = pd.DataFrame(ds_train_return, columns= ['total_intl_calls', 'total_intl_charge','number_customer_service_calls', 'international_plan_num',
                                                           'voice_mail_plan_num'])
    df_target = pd.DataFrame(target_y, columns=['churn'])

    # Junta os 2 dataframes por coluna
    ds_train_return = pd.concat([df_features,df_target],axis=1) 

    del (ds_train_array)
    del (features_x)
    del (target_y)
    del (StScaler)
    del (df_features)
    del (df_target)

    return ds_train_return
ds_train_standard = standardizationData(ds_train_Selected)
ds_train_standard.head(5)
ds_train_standard.describe()
columns = [col for col in ds_train_Selected.columns if col not in ['churn']]
for i in columns:
    #ds_train_Selected[i].plot(kind = 'hist')
    sns.distplot(ds_train_Selected[i], rug=True)
    plt.title('Histograma Variável (After Standard): ' + i)
    plt.xlabel(i)    
    plt.show()
columns = [col for col in ds_train_standard.columns if col not in ['churn']]

for i in columns:
    #ds_train_standard[i].plot(kind = 'box')
    sns.boxplot(data=ds_train_standard[i])
    plt.title('BoxPlots Variável: ' + i)  
    plt.show()
def BalancingData(dataset, var_target):
    
    # Split da variavel target e variaveis preditoras
    x_train = dataset.drop([var_target], axis=1)
    y_train = dataset[var_target]
    
    smt = SMOTE()
    
    # Separando os dados em features e target
    features_train, target_train = smt.fit_sample(x_train, y_train)
    
    target_train_DF = pd.DataFrame(target_train)
    # atribuindo o nome da coluna
    target_train_DF.columns = ['churn']
    
    features_train_DF = pd.DataFrame(features_train)
    features_train_DF.columns = x_train.columns
    
    return pd.concat([features_train_DF, target_train_DF], axis=1)
ds_train_final = BalancingData(ds_train_standard, 'churn')

Count_No = len(ds_train_final[ds_train_final['churn']==0])
Count_Yes = len(ds_train_final[ds_train_final['churn']==1])

print ("Percentual de Clientes que Cancelou Depois do Balanceamento: ", round((Count_Yes / (Count_Yes+Count_No))*100,2))
print ("Percentual de Clientes que Não Cancelou Depois do Balanceamento: ", round((Count_No / (Count_Yes+Count_No))*100,2))
ds_train_array = ds_train_final.values

# Separando os dados em inputs(features) e outputs (target)
features_x = ds_train_array[:,:-1] # selecionando todas as colunas exceto a ultima (que é a TARGET)
target_y = ds_train_array[:,5:] # selecionando somente a ultima coluna Target

# Definindo o tamanho da amonstra
data_size_test = 0.30 # 30%

# Criando os conjuntos de dados de treino e de teste
X_treino, X_teste, Y_treino, Y_teste = train_test_split(features_x, target_y, test_size = data_size_test)

del (features_x)
del (target_y)
del (data_size_test)
del (ds_train_array)
# Criação do modelo
modelo = LogisticRegression()

# Treinamento do modelo com os dados de treino
modelo.fit(X_treino, Y_treino)

# Acurácia do modelo nos dados de teste
# Utilizado 
modelo.score(X_teste, Y_teste)
def CrossValidation(model, x_test, y_test, metric_scoring, kfold):
    
    cv_result = cross_val_score(model, X= x_test, y= y_test, cv=kfold, scoring= metric_scoring)

    return print("Cross-Validation mean:",cv_result.mean())
kfold = KFold(n_splits= 10, shuffle=True)

CrossValidation(modelo, X_teste, Y_teste, 'accuracy', kfold)
# Prevendo o resultado do modelo informando as variáveis preditoras de teste, para depois realizar a classificação e taxa
# de acerto obtido pelo modelo.
target_predicted = modelo.predict(X_teste)
target_proba     = modelo.predict_proba(X_teste)

print("Accuracy (TP/Total):",metrics.accuracy_score(Y_teste, target_predicted))
print("Precision (TP/TP+FP):",metrics.precision_score(Y_teste, target_predicted))
print("Recall (TP/TP+FN):",metrics.recall_score(Y_teste, target_predicted))
print("Classification Report:")
print(metrics.classification_report(Y_teste, target_predicted))
print(target_proba)
cnf_matrix = metrics.confusion_matrix(Y_teste, target_predicted)

class_names = [1,0]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="viridis" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
valores_grid = {'C': [0.001,0.01,0.1,1,10,100,1000,1000000]}

rsearch = GridSearchCV(estimator = modelo, param_grid = valores_grid)

rsearch.fit(X_treino, Y_treino)

rsearch.best_score_

# Carregando o dataset de treino utilizando Pandas
ds_test = pd.read_csv("/kaggle/input/projeto4_telecom_teste.csv", sep = ",")
ds_test.head(5)
# Remove a coluna 
ds_test = ds_test.drop('Unnamed: 0', axis=1)
# Verificando se existe valores missing
ds_test.isnull().values.any()
ds_test = transformColumns(ds_test)
ds_test.head(5)
# Aplica a seleção de Variáveis 
ds_test_Selected = SelectVariablesRFECV(ds_test)
# Aplica a padronização dos dados e remover a coluna Churn
ds_test_standard = standardizationData(ds_test_Selected)

ds_test_standard = ds_test_standard.drop('churn', axis=1)

ds_test_standard.head(5)
# Transforma as variáveis target em array
features_x = ds_test_standard.values
predicted_value = modelo.predict(features_x)
predicted_proba = modelo.predict_proba(features_x)

predict_value_df = pd.DataFrame(predicted_value, columns=['Ação Prevista'])
predict_proba_df = pd.DataFrame(predicted_proba, columns=['Prob. Não Cancelamento','Prob. Cancelamento'])
result_final = pd.concat([ds_test_standard, predict_value_df, predict_proba_df],axis=1)
result_final.head(10)
result_final.groupby('Ação Prevista').size()