import pandas as pd

import csv

import numpy as np

import sklearn

import matplotlib
# Carregar os dados TREINO

arq = '../input/abalone-train.csv'

dados = pd.read_csv(arq)

dados.head()
def preproc(dados):



    # LE(sex), CONV(sex), DROP(length), NORM(X)

    from sklearn.model_selection import cross_val_score

    from sklearn import metrics



    # Mudar coluna String para Inteiro

    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()



    campos = ['sex']

    for campo in campos:

        dados[campo] = label_encoder.fit_transform(dados[campo])



    # Remover length

    #dados = dados.drop(['length'], axis='columns')



    # Normalização

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))

    #lista_nome_colunas = list(dados.columns.values)

    lista_nome_colunas = ['length','diameter','height','whole_weight','shucked_weight','viscera_weight','shell_weight']

    dados[lista_nome_colunas] = scaler.fit_transform(dados[lista_nome_colunas])

    dados[lista_nome_colunas] = pd.DataFrame(dados[lista_nome_colunas], columns=lista_nome_colunas)

    

        # Converter para categorico

    #dados[['sex']] = dados['sex'].astype('category')



    return dados

dados = preproc(dados)

dados.info()
# Dados entrada(X) e saida(Y)

Y = dados['rings']

X = dados.drop(['rings'], axis='columns')

X = preproc(X)

X.head()

import xgboost as xgb

model = xgb.XGBRegressor(n_estimators = 300)

model.fit(X, Y)
#==================================================================================================
# Carregar os dados TESTE

arq_teste = '../input/abalone-test.csv'

X_teste = pd.read_csv(arq_teste)

X_teste.head()
X_teste_norm = X_teste.copy()

X_teste_norm = preproc(X_teste_norm)
# executar previsão usando o modelo

y_pred = model.predict(X_teste_norm)
y_pred = np.round(y_pred)

y_pred = y_pred.astype('int')

y_pred


# gerar dados de envio (submissão)

submission = pd.DataFrame({

  'id': X_teste.id,

  'rings': y_pred

})

submission.set_index('id', inplace=True)



# gerar arquivo CSV para o envio

filename = 'abalone-submission.csv'

# submission.to_csv(filename)