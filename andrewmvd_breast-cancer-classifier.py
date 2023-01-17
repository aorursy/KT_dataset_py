import pandas as pd

import numpy  as np
# Ler arquivo e mostrar primeiras 5 linhas

full_data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

full_data.head()
# Verificar Nomes de Colunas

full_data.columns
# Eliminar ultima coluna e coluna de id

full_data.drop('Unnamed: 32', axis=1, inplace=True)

full_data.drop('id', axis=1, inplace=True)
# Verificar tamanho do dataset - linhas e colunas

full_data.shape
full_data['diagnosis'].unique()
full_data['diagnosis'].value_counts()
357/(212 + 357)
full_data['diagnosis'] = np.where(full_data['diagnosis'] == 'M',1,0)

full_data.head()
from sklearn.model_selection import train_test_split
# Entender como funciona função para separar em set de treino e teste

help(train_test_split)
# Fazer split entre treino e teste

train, test = train_test_split(full_data, test_size = 0.3, random_state = 42, stratify=full_data['diagnosis'])
# Separar variável dependente das independentes

train_y = train.pop('diagnosis')

test_y  = test.pop('diagnosis')
train.head()
from sklearn.linear_model import LogisticRegression
help(LogisticRegression)
modelo = LogisticRegression(random_state = 42, max_iter = 1000)

modelo.fit(train, train_y)
from sklearn.metrics import accuracy_score, roc_auc_score
def evaluate_predictions(preds, eval_series):

    '''

    Evaluate Predictions Function

    Returns accuracy and auc of the model

    '''

    auroc = roc_auc_score(eval_series.astype('uint8'), preds)

    accur = accuracy_score(eval_series.astype('uint8'), preds >= 0.5)

    print('Accuracy: ' + str(auroc))

    print('AUC: ' + str(accur))
# Gerar predições para o set de teste

logit_preds = modelo.predict_proba(test)

logit_preds
# Avaliar Acurácia e AUC

evaluate_predictions(logit_preds[:,1], eval_series = test_y)