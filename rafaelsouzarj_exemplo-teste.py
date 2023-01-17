# https://pypi.org/project/pandas/

import pandas as pd



# https://numpy.org/

import numpy as np



# https://matplotlib.org/

import matplotlib.pyplot as plt



# https://seaborn.pydata.org/

import seaborn as sns



# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

from sklearn.model_selection import train_test_split



# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

from sklearn.linear_model import LinearRegression



# https://scikit-learn.org/stable/modules/model_evaluation.html

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



# https://docs.python.org/3/library/math.html

from math import sqrt
# Arquivos de entrada de dados estão disponíveis no diretório somente-leitura "../input/"

# Por exemplo, executar esta célula (ao clicar no botão ">" à esquerda da célula ou pressionando Shift+Enter) irá listar todos os arquivos dentro do diretório de entrada



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Você pode escrever até 5GB no diretório atual (/kaggle/working/) que serão mantidos como output quando você criar uma versão usando "Save & Run All"        

# Você também pode escrever arquivos temporário em /kaggle/temp, mas eles não serão salvos fora da sessão atual
# Realiza um pré-processamento básico no dataframe.

def preparar_dataset(df_pre):



    # Remove os atributos com baixa correlação com o alvo (price).

    # Fique à vontade para incluir ou retirar atributos que julgar relevantes aos resultados.

    df_pre.drop(['id'],axis=1,inplace=True)

    df_pre.drop(['host_name','amenities'],axis=1,inplace=True)

    df_pre.drop(['zipcode','latitude','longitude'],axis=1,inplace=True)



    # Converte os atributos categóricos em quantitativos discretos.

    for column in df_pre.columns:

        if str(df_pre[column].dtype) not in ['float64', 'int64']:

            df_pre[column] = df_pre[column].map \

            (dict(zip(df_pre[column].unique().tolist(),range(len(df_pre[column].unique().tolist())))))



    return df_pre
# Calcula métricas de desempenho do classificador. 

# Fique à vontade para incluir outras métricas que julgar úteis na avalição dos resultados.

def get_metrics(y_test, y_pred):  



    rmse = round(sqrt(mean_squared_error(y_test, y_pred)),4)

    r2 = round(r2_score(y_test, y_pred),4)



    return rmse, r2
# Semente aleatória a ser usada ao longo desse notebook.

# Procure manter sempre a mesma semente aleatória. Desse modo, poderá comparar a evolução entre diferentes técnicas.

random_state=660601



# Nome do arquivo fornecido pelo desafio com os dados rotulados para treino.

nome_arquivo_com_rotulos_para_treino = '../input/teste-para-fabio-temp/' + 'treino.csv'



# Nome do arquivo fornecido pelo desafio com os dados não rotulados, que deverão ser analisados pelo modelo construído aqui.

nome_arquivo_sem_rotulos = '../input/teste-para-fabio-temp/' + 'teste.csv'



# Nome do arquivo que será criado com os rótulos gerados pelo modelo

# Esse é o arquivo se será submetido à página do desafio.

nome_arquivo_rotulado_regressor = '../working/' + 'submissao.csv'
df = pd.read_csv(nome_arquivo_com_rotulos_para_treino, index_col=None, engine='python', sep =';', encoding="utf-8")

print('Total de registros carregados:',len(df))



# Exibe uma amostra dos dados.

df.head()
df = preparar_dataset(df)

df.head()
corr = df.corr()

corr.style.background_gradient(cmap='coolwarm')

plt.figure(figsize=(20,8))

sns.heatmap(corr,annot=True)
corr.sort_values(by = ['price'], ascending = False, inplace = True)

corr['price']
regr = LinearRegression(fit_intercept=True, normalize=False)
# Vamos dividir da seguinte forma: 80% para treino, 20% para teste. 

# Os registros deverão ser embaralhados.

df_train, df_test = train_test_split(df, test_size=0.20, shuffle=True, random_state=random_state)

print('Tamanho do Treino:',len(df_train),'- Tamanho do Teste:',len(df_test))
# Treina o modelo com a massa de treino.

X_train = pd.DataFrame(df_train.drop(['price'],axis=1))

y_train = pd.DataFrame(df_train['price'])

regr.fit(X_train, y_train)
# Avalia a performance do modelo treinado, usando a massa reservada para testes.

X_test = pd.DataFrame(df_test.drop(['price'],axis=1))

y_test = pd.DataFrame(df_test['price'])

y_pred = regr.predict(X_test)
# Obtém as métricas de desempenho - o quanto nosso modelo acertou?

rmse, r2 = get_metrics(y_test, y_pred)

print('RMSE (Raiz do erro médio quadrático - Quanto mais próximo a 0.0000, melhor):',rmse)

print('R² (Coeficiente de determinação - Quanto mais  próximo a 1.0000, melhor):',r2)
# Exemplo de visualização de dados

results = pd.DataFrame(np.array(y_test).flatten(),columns=['Realizado'])

results['Previsto'] = np.array(y_pred)

results.head(100).plot(kind='line')
# Carrega os dados da base não rotulada.

df_test = pd.read_csv(nome_arquivo_sem_rotulos, index_col=None, engine='python', sep =';', encoding="utf-8")

print('Total de registros carregados:',len(df_test))
df_test.head()
# Salvando a coluna 'id', para montar o arquivo de envio ao final da execução.

id_col = df_test['id'] 
# Prepara os dados para classificação.

X_test = preparar_dataset(df_test)

X_test.head()
# Executa a predição dos registros não rotulados

y_pred = regr.predict(X_test)

df_test['id'] = id_col

df_test['price'] = y_pred

df_test['price'] = df_test['price'] * 1.01 #Multiplicação para diferenciar o resultado da submissão de baseline.

df_test['price'] = df_test['price'].round(decimals=2)



# Exibe uma amostra dos resultados

df_test.head(10)
# Salva os registros classificados

df_test.to_csv(nome_arquivo_rotulado_regressor, index=False, sep=",", encoding="utf-8", columns=['id','price'])