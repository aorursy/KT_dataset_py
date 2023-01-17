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
# Carregando os dados
df = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')


df.shape
# Quais colunas do dataframe são do tipo object
df.select_dtypes('object').head()
# Transforma data em datetime
df['date'] = pd.to_datetime(df['date'])
df['date'].head()
df.columns
# Olhando as colu nas
df.dtypes
#Verificando se há dados nulos
df.info()
# Importando as bibliotecas de graficos
import seaborn as sns
import matplotlib.pyplot as plt
# Correlação das variáveis numéricas
plt.figure(figsize= (15, 15))

sns.heatmap(df.corr(), square=True, annot=True, linewidth=0.5)
# Selecionando as colunas que serão usadas para treino

# Não vamos usar as notas intermediárias para treinamento
# e também não usaremos a variável target 'G3'
remove = ['price','date']

# Lista com as colunas a serem usadas
feats = [col for col in df.columns if col not in remove]
feats
# Importando o train_test_split do scikit learn
from sklearn.model_selection import train_test_split

# Separando o dataframe em treino e teste e usando uma semente aleatória
# para reproduzir os resultados
# Por padrão o test_size é 0.25
train, test = train_test_split(df, test_size=0.2, random_state=42)
# Instanciando o random forest classifier
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=-1, n_estimators=200, random_state=42)
# Treinando o modelo
rf.fit(train[feats], train['price'])
# Prever o Target de teste usando o modelo treinado
test['price'] = rf.predict(test[feats]).astype(int)
# Vamos verificar as previsões
test['price'].value_counts(normalize=True)
# Criando o arquivo para submissão
test[['id', 'price']].to_csv('submission.csv', index=False)
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(15, 20))

# Avaliando a importancia de cada coluna (cada variável de entrada)
pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()