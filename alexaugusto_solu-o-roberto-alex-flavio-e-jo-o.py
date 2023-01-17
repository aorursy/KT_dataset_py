!wget -O train.parquet https://www.dropbox.com/s/j3jupvnmi4xelwz/train.parquet?dl=1
!wget -O test.parquet https://www.dropbox.com/s/95jwpl5bs7o8i7g/test.parquet?dl=1
!ls /home/jovyan/pos/estudo-de-caso-eleicoes
import pandas as pd

work_dir = "//home/jovyan/pos/estudo-de-caso-eleicoes"

#leitura dos dados de entrada
df_train_bruto = pd.read_parquet(work_dir +"/train.parquet", engine="pyarrow")
df_test_bruto = pd.read_parquet(work_dir +"/test.parquet", engine="pyarrow")

print(df_train_bruto.shape)
print(df_test_bruto.shape)
import numpy as np
df_train_bruto.info()
df_test_bruto.info()
df_train_bruto['TOTAL_BENS'] = df_train_bruto['TOTAL_BENS'].fillna(value=0)
df_test_bruto['TOTAL_BENS'] = df_test_bruto['TOTAL_BENS'].fillna(value=0)
bin_ranges = [
    -1,
    0,
    25000,
    50000,
    100000, # 5
    150000,
    200000,
    250000,
    300000,
    350000,
    400000,
    500000,
    1000000,
    2000000] # 14
bin_names = [1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13]
# df_train_bruto['TOTAL_BENS_BINS'] = pd.cut(
#                                            np.array(
#                                               df_train_bruto['TOTAL_BENS']), 
#                                               bins=bin_ranges)

df_train_bruto['TOTAL_BENS_BINS_LABELS'] = pd.cut(
                                           np.array(
                                              df_train_bruto['TOTAL_BENS']), 
                                              bins=bin_ranges,            
                                              labels=bin_names)

df_test_bruto['TOTAL_BENS_BINS_LABELS'] = pd.cut(
                                           np.array(
                                              df_test_bruto['TOTAL_BENS']), 
                                              bins=bin_ranges,            
                                              labels=bin_names)
# view the binned features 
print(df_train_bruto[['TOTAL_BENS', 'TOTAL_BENS_BINS_LABELS']].head())
print(df_test_bruto[['TOTAL_BENS', 'TOTAL_BENS_BINS_LABELS']].head())
def formatitem(item):
    l = item.split('/')
    for x,i in enumerate(l):
        l[x] = i.strip()

    l = set(l)
    return len(l)

df_train_bruto['DS_COMPOSICAO_COLIGACAO_QTD'] = df_train_bruto['DS_COMPOSICAO_COLIGACAO'].map(formatitem)
df_test_bruto['DS_COMPOSICAO_COLIGACAO_QTD'] = df_test_bruto['DS_COMPOSICAO_COLIGACAO'].map(formatitem)
np.unique(df_train_bruto['DS_COMPOSICAO_COLIGACAO_QTD'])
bin_ranges = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90]
bin_names = [2, 25, 3, 35, 4, 45, 5, 55, 6, 65, 7, 8]

df_train_bruto['NR_IDADE_DATA_POSSE_BIN'] = pd.cut(
                                           np.array(
                                              df_train_bruto['NR_IDADE_DATA_POSSE']), 
                                              bins=bin_ranges,            
                                              labels=bin_names)

df_test_bruto['NR_IDADE_DATA_POSSE_BIN'] = pd.cut(
                                           np.array(
                                              df_test_bruto['NR_IDADE_DATA_POSSE']), 
                                              bins=bin_ranges,            
                                              labels=bin_names)

print(df_train_bruto[['NR_IDADE_DATA_POSSE', 'NR_IDADE_DATA_POSSE_BIN']].head())
print(df_test_bruto[['NR_IDADE_DATA_POSSE', 'NR_IDADE_DATA_POSSE_BIN']].head())
# iremos remover as features NR_IDADE_DATA_POSSE e TOTAL_BENS, 
# mas fica como exercício você aproveitá-las no conjunto de features
df_train_preparado = df_train_bruto.drop(['ID_CANDIDATO', 'NR_IDADE_DATA_POSSE', 'TOTAL_BENS'], axis=1)
df_test_preparado = df_test_bruto.drop(['ID_CANDIDATO', 'NR_IDADE_DATA_POSSE', 'TOTAL_BENS'], axis=1)
# Substitui valores nulos por 0 nas colunas numéricas
colunas_numericas = [
    'ANO_ELEICAO',
    'CD_TIPO_ELEICAO',
    'NR_TURNO',
    'CD_ELEICAO',
    'CD_CARGO',
    'CD_SITUACAO_CANDIDATURA',
    'CD_DETALHE_SITUACAO_CAND',
    'NR_PARTIDO',
    'CD_NACIONALIDADE',
    'CD_GENERO',
    'CD_GRAU_INSTRUCAO',
    'CD_ESTADO_CIVIL',
    'CD_COR_RACA',
    'CD_OCUPACAO'
]

df_train_preparado[colunas_numericas] = df_train_preparado[colunas_numericas].fillna(value=0)
df_test_preparado[colunas_numericas] = df_test_preparado[colunas_numericas].fillna(value=0)
# remoção das 
df_train_preparado = df_train_preparado.drop(['DS_COMPOSICAO_COLIGACAO', 'NM_MUNICIPIO_NASCIMENTO'], axis=1)
df_test_preparado = df_test_preparado.drop(['DS_COMPOSICAO_COLIGACAO', 'NM_MUNICIPIO_NASCIMENTO'], axis=1)
# transformar colunas categóricas em numéricas
colunas_categoricas = ['SG_UF','TP_AGREMIACAO','SG_UF_NASCIMENTO','ST_REELEICAO','ST_DECLARAR_BENS',
                      'NR_IDADE_DATA_POSSE_BIN',
                      'TOTAL_BENS_BINS_LABELS',
                      'DS_COMPOSICAO_COLIGACAO_QTD'
                      ]
df_train_preparado = pd.get_dummies(df_train_preparado, columns=colunas_categoricas)
df_test_preparado = pd.get_dummies(df_test_preparado, columns=colunas_categoricas)
df_train_preparado.head()
print(df_test_preparado.shape)
print(df_train_preparado.shape)
df_train_preparado['ELEITO'] = df_train_preparado['ELEITO'].astype(bool)
# !pip install h2o
import h2o
from h2o.automl import H2OAutoML

h2o.init(max_mem_size=8)
train_df = h2o.H2OFrame(df_train_preparado)

train, valid = train_df.split_frame(ratios = [.8], seed = 1234)

x = train.columns
y = "ELEITO"
x.remove(y)
aml = H2OAutoML(max_models=3, seed=10, include_algos = ["XGBoost", "GBM"], nfolds=0, sort_metric="AUC")
# aml = H2OAutoML(max_models=20, seed=10, nfolds=0, sort_metric="AUC")
aml.train(x=x, y=y, training_frame=train, validation_frame=valid)

#aml = H2OAutoML(max_models=5, seed=1, include_algos = ["XGBoost"])
#aml.train(x=x, y=y, training_frame=train_df)
lb = aml.leaderboard
lb.head(rows=lb.nrows)
model = aml.leader
model
model.varimp(use_pandas=True)
model.varimp_plot(20)
df_test_preparado.info()
test = h2o.H2OFrame(df_test_preparado)
preds = aml.predict(test)

#predict
#y_test =  preds.as_data_frame()['predict'].astype(int)

#predict_proba
y_test =  preds.as_data_frame()['True']
preds
output = df_test_bruto.assign(ELEITO=y_test)
output = output.loc[:, ['ID_CANDIDATO','ELEITO']]
output.head()
output.to_csv(work_dir +"/ouput_h2o_v4_2f.csv", index=False)
# from google.colab import files
# files.download('ouput_h2o.csv') 