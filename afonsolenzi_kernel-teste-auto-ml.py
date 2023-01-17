# Instalando o pacote h2o

!pip install h2o
# Importando as bibliotecas

import pandas as pd

import h2o

from h2o.automl import H2OAutoML



# Inicializando o cluster

h2o.init()

h2o.cluster().show_status()
train = h2o.import_file("../input/titanic-train-dataset/train.csv")
test  = h2o.import_file("../input/testtitanic/titanic_data.csv")
# Convertendo a coluna target em fator

train['Survived'] = train['Survived'].asfactor()
train.columns
# Indicando as colunas preditoras e a target

x = train.columns

y = 'Pclass'



# Converter o target em fator

train[y] = train[y]



# Executa o AutoML para 20 modelos (limitado a 1h de duracao máxima)

aml = H2OAutoML(max_models=10, seed=0)

aml.train(x=x, y=y, training_frame=train)
# AutoML Leaderboard

lb = aml.leaderboard



# Exibir as colunas do resultado dos modelos

lb.head(rows=lb.nrows)
# Detalhes do melhor modelo

aml.leader
# Gerar as previsoes no dataset de teste

preds = aml.leader.predict(test)
# Salvar o resultado no dataset de teste

result = preds[:,0]

test['PredictedProb'] = result
preds