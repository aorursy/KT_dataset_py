!wget -O train.parquet https://www.dropbox.com/s/j3jupvnmi4xelwz/train.parquet?dl=1
!wget -O test.parquet https://www.dropbox.com/s/95jwpl5bs7o8i7g/test.parquet?dl=1
!ls /kaggle/working
!pip install xgboost
# imports necessarios

import pandas as pd

import numpy as np

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")



from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier
#work_dir = "/content"



#leitura dos dados de entrada

#df_train_bruto = pd.read_parquet(work_dir +"/train.parquet", engine="pyarrow")

#df_test_bruto = pd.read_parquet(work_dir +"/test.parquet", engine="pyarrow")



df_train_bruto = pd.read_parquet('./train.parquet', engine="pyarrow")

df_test_bruto = pd.read_parquet('./test.parquet', engine="pyarrow")



identificador = df_test_bruto['ID_CANDIDATO']
## Junta os datasets de treino e teste para obter o mesmo numero de features durante a conversão de categorias

train_len = len(df_train_bruto) # guarda o tamanho do train para divisão do dataset depois

dataset =  pd.concat(objs=[df_train_bruto, df_test_bruto], axis=0).reset_index(drop=True)
# Análise dos datasets

len(df_train_bruto)
len(df_test_bruto)
len(dataset)
dataset.isnull().sum()
dataset.info()
dataset.head()
from collections import Counter

Counter(dataset['NR_IDADE_DATA_POSSE'])
# Explorar NR_IDADE_DATA_POSSE vs ELEITO

g = sns.barplot(x="NR_IDADE_DATA_POSSE",y="ELEITO",data=dataset)

g = g.set_ylabel("ELEITO")
# Categorizando a feature NR_IDADE_DATA_POSSE

age_med = dataset["NR_IDADE_DATA_POSSE"].median()

dataset["NR_IDADE_DATA_POSSE"] = dataset["NR_IDADE_DATA_POSSE"].fillna(age_med)

dataset["NR_IDADE_DATA_POSSE"] = dataset["NR_IDADE_DATA_POSSE"].astype(int)

dataset["NR_IDADE_DATA_POSSE"] = dataset["NR_IDADE_DATA_POSSE"].map(lambda s: 0 if 18 <= s <= 30 else s)

dataset["NR_IDADE_DATA_POSSE"] = dataset["NR_IDADE_DATA_POSSE"].map(lambda s: 1 if 31 <= s <= 50 else s)

dataset["NR_IDADE_DATA_POSSE"] = dataset["NR_IDADE_DATA_POSSE"].map(lambda s: 2 if 51 <= s <= 70 else s)

dataset["NR_IDADE_DATA_POSSE"] = dataset["NR_IDADE_DATA_POSSE"].map(lambda s: 3 if 71 <= s else s)

Counter(dataset['NR_IDADE_DATA_POSSE'])
Counter(dataset['CD_GENERO'])
Counter(dataset['CD_GRAU_INSTRUCAO'])
# Tratando os dados nao divulgaveis

dataset['CD_GENERO'] = dataset['CD_GENERO'].map(lambda s: 2 if s == -4 else s)

dataset['CD_GRAU_INSTRUCAO'] = dataset['CD_GRAU_INSTRUCAO'].map(lambda s: 8 if s == -4 else s)
import math

index_NaN_bens = []

for i in dataset["TOTAL_BENS"].index:

  if math.isnan(dataset["TOTAL_BENS"].loc[i]):

    index_NaN_bens.append(i)



len(index_NaN_bens)
df_out = dataset.drop(index_NaN_bens, axis = 0).reset_index(drop=True)

mean = df_out["TOTAL_BENS"].mean()

df_out = df_out[df_out["TOTAL_BENS"]<mean]

df_out.head()
df_out["TOTAL_BENS"].mean()
# Tratando os valores nulos da coluna TOTAL_BENS



for i in index_NaN_bens :

    #media dos valores dos bens sem os outliers

    bens_media = 234690

    bens_pred = dataset["TOTAL_BENS"][((dataset['CD_COR_RACA'] == dataset.iloc[i]["CD_COR_RACA"]) & (dataset['CD_GRAU_INSTRUCAO'] == dataset.iloc[i]["CD_GRAU_INSTRUCAO"]) & (dataset['CD_CARGO'] == dataset.iloc[i]["CD_CARGO"]))].median()

    #verifica se TOTAL_BENS é NaN, ou seja, caso tenha encontrado outros índices e feito a média, atribui esse valor no dataset. Caso contrário atribui a média do dataset original

    if not np.isnan(bens_pred) :

        dataset["TOTAL_BENS"].iloc[i] = bens_pred

    else :

        dataset["TOTAL_BENS"].iloc[i] = bens_media
print(dataset.isnull().sum())
# remoção das colunas 'DS_COMPOSICAO_COLIGACAO' e 'NM_MUNICIPIO_NASCIMENTO'

dataset = dataset.drop(['DS_COMPOSICAO_COLIGACAO', 'NM_MUNICIPIO_NASCIMENTO'], axis=1)
# transformar colunas categóricas em numéricas

colunas_categoricas = ['SG_UF','TP_AGREMIACAO','SG_UF_NASCIMENTO','ST_REELEICAO','ST_DECLARAR_BENS']

dataset = pd.get_dummies(dataset, columns=colunas_categoricas)

dataset.head()
print(dataset.shape)
from imblearn.over_sampling import RandomOverSampler 



treino = dataset[:train_len]

teste = dataset[train_len:]

teste.drop(labels=["ELEITO"],axis = 1,inplace=True)



skf = StratifiedKFold(n_splits=5, random_state=420, shuffle=True)

X = treino.drop('ELEITO', axis=1).values

y = treino['ELEITO'].values

media_treino =[]

media_teste = []

media_validacao =[]



ros = RandomOverSampler(random_state=420)
contador = 1

for indice_treino, indice_validacao in skf.split(X, y):    

    X_treino = X[indice_treino]

    y_treino = y[indice_treino]    

    X_validacao = X[indice_validacao]

    y_validacao = y[indice_validacao]



    X_treino, y_treino = ros.fit_resample(X_treino, y_treino)

    print('Resampled dataset shape %s' % Counter(y_treino))



    modelo = XGBClassifier(random_state=420, n_estimators=2000, n_jobs=4)

    modelo.fit(X_treino, y_treino, eval_set=[(X_validacao, y_validacao)], eval_metric="auc", early_stopping_rounds=200, verbose=True)    

    

    y_pred = modelo.predict_proba(X_treino)    

    y_pred = y_pred[:,1]

    score_treino = roc_auc_score(y_treino, y_pred)  

    print("Treino número {} : {}".format(contador, score_treino))



    y_validacao_pred = modelo.predict_proba(X_validacao)

    y_validacao_pred = y_validacao_pred[:,1]

    score_validacao = roc_auc_score(y_validacao, y_validacao_pred)

    print("Validacao número {} : {} \n".format(contador, score_validacao))



    contador += 1



    X_teste = teste.values

    y_pred_teste = modelo.predict_proba(X_teste)

    y_pred_teste = y_pred_teste[:, 1]  



    media_treino.append(score_treino)  

    media_validacao.append(score_validacao)

    media_teste.append(y_pred_teste)



print("Media de todos treinos {}:".format(np.mean(media_treino)))

print("Media de todas validações {}:".format(np.mean(media_validacao)))
mediafinal_pred = np.mean(media_teste, axis=0)
output = pd.concat([identificador, pd.DataFrame(mediafinal_pred, columns=['ELEITO'])], axis=1)

output
#output.to_csv(work_dir +"/ouput_sklearn.csv", index=False)

output.to_csv('ouput_sklearn.csv', index=False)
#from google.colab import files

#files.download('ouput_sklearn.csv')