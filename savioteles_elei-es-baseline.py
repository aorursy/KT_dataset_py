!wget -O train.parquet https://www.dropbox.com/s/j3jupvnmi4xelwz/train.parquet?dl=1
!wget -O test.parquet https://www.dropbox.com/s/95jwpl5bs7o8i7g/test.parquet?dl=1
!ls /content
import pandas as pd

work_dir = "/content"

#leitura dos dados de entrada
df_train_bruto = pd.read_parquet(work_dir +"/train.parquet", engine="pyarrow")
df_test_bruto = pd.read_parquet(work_dir +"/test.parquet", engine="pyarrow")

print(df_train_bruto.shape)
print(df_test_bruto.shape)
df_train_bruto.info()
df_test_bruto.info()
# iremos remover as features NR_IDADE_DATA_POSSE e TOTAL_BENS, 
# mas fica como exercício você aproveitá-las no conjunto de features
df_train_preparado = df_train_bruto.drop(['ID_CANDIDATO', 'NR_IDADE_DATA_POSSE', 'TOTAL_BENS'], axis=1)
df_test_preparado = df_test_bruto.drop(['ID_CANDIDATO', 'NR_IDADE_DATA_POSSE', 'TOTAL_BENS'], axis=1)
# Substitui valores nulos por 0 nas colunas numéricas
colunas_numericas = ['ANO_ELEICAO','CD_TIPO_ELEICAO','NR_TURNO','CD_ELEICAO','CD_CARGO','CD_SITUACAO_CANDIDATURA','CD_DETALHE_SITUACAO_CAND','NR_PARTIDO','CD_NACIONALIDADE','CD_GENERO','CD_GRAU_INSTRUCAO','CD_ESTADO_CIVIL','CD_COR_RACA','CD_OCUPACAO']

df_train_preparado[colunas_numericas] = df_train_preparado[colunas_numericas].fillna(value=0)
df_test_preparado[colunas_numericas] = df_test_preparado[colunas_numericas].fillna(value=0)
# remoção das 
df_train_preparado = df_train_preparado.drop(['DS_COMPOSICAO_COLIGACAO', 'NM_MUNICIPIO_NASCIMENTO'], axis=1)
df_test_preparado = df_test_preparado.drop(['DS_COMPOSICAO_COLIGACAO', 'NM_MUNICIPIO_NASCIMENTO'], axis=1)
# transformar colunas categóricas em numéricas
colunas_categoricas = ['SG_UF','TP_AGREMIACAO','SG_UF_NASCIMENTO','ST_REELEICAO','ST_DECLARAR_BENS']
df_train_preparado = pd.get_dummies(df_train_preparado, columns=colunas_categoricas)
df_test_preparado = pd.get_dummies(df_test_preparado, columns=colunas_categoricas)
df_train_preparado.head()
print(df_test_preparado.shape)
print(df_train_preparado.shape)
df_train_preparado['ELEITO'] = df_train_preparado['ELEITO'].astype(bool)
!pip install h2o
import h2o
from h2o.automl import H2OAutoML

h2o.init(max_mem_size=8)
train_df = h2o.H2OFrame(df_train_preparado)

train, valid = train_df.split_frame(ratios = [.8], seed = 1234)

x = train.columns
y = "ELEITO"
x.remove(y)
aml = H2OAutoML(max_models=2, seed=10, include_algos = ["XGBoost"], nfolds=0, sort_metric="AUC")
aml.train(x=x, y=y, training_frame=train, validation_frame=valid)

#aml = H2OAutoML(max_models=5, seed=1, include_algos = ["XGBoost"])
#aml.train(x=x, y=y, training_frame=train_df)
lb = aml.leaderboard
lb.head(rows=lb.nrows)
model = aml.leader
model
model.varimp(use_pandas=True)
model.varimp_plot()
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
output.to_csv(work_dir +"/ouput_h2o.csv", index=False)
from google.colab import files
files.download('ouput_h2o.csv') 