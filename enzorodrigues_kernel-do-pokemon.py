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
df = pd.read_csv("/kaggle/input/pokemon/pokemon_alopez247.csv")
df.shape
# Olhando os dados
df.head().T
# Tipos dos dados
df.info()
# Variável target 
df['Type_1'].value_counts()
# Verificar os valores nulos no dataframe
df.isnull().sum()
# Verificando os valores na variável Type_2
df['Type_2'].value_counts()
# Olhar quantos atributos tem por coluna
df.nunique()
# Verificando a coluna Pr_Male
df['Pr_Male'].value_counts()
# Preenchendo os valores nulos nas variáveis de tipo object
df['Type_2'].fillna('NoType',inplace=True)
df['Egg_Group_2'].fillna('Desconhecido',inplace=True)
df.head().T
# Posso usar dummies nas variaveis 'isLegendary', 'hasGender', 'hasMegaEvolution' pq  elas são booleanas, possuem dois atributos.
#pd.get_dummies(df['isLegendary'])
#pd.get_dummies(df['hasGender'])
#pd.get_dummies(df['hasMegaEvolution'])
df.info()
df.drop('Pr_Male', inplace=True, axis=1)
from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder() 
df['Type_1']= le.fit_transform(df['Type_1']) 
df['Type_2']= le.fit_transform(df['Type_2'])
df['Egg_Group_1']= le.fit_transform(df['Egg_Group_1'])
df['Egg_Group_2']= le.fit_transform(df['Egg_Group_2'])
df['Body_Style']= le.fit_transform(df['Body_Style'])
df['Name']= le.fit_transform(df['Name'])
df['Color']= le.fit_transform(df['Color'])
# Informação sobre o Dataframe
df.info()
# Dividindo o Dataframe em treino e teste
from sklearn.model_selection import train_test_split
train, test = train_test_split(df,test_size=0.2, random_state=42)

train.shape, test.shape
# Definindo as feats
feats = [c for c in df.columns if c not in ['Type_1']]
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_leaf=3,oob_score=True)
rf.fit(train[feats],train['Type_1'])
preds_train = rf.predict(train[feats])
# Vamos olhar as variáveis que foram as mais importantes usadas pelo modelo
pd.Series(rf.feature_importances_ , index=feats).sort_values().plot.barh()
# Importar a acurácia 
from sklearn.metrics import accuracy_score
accuracy_score(train['Type_1'], preds_train)
# Esta é a precisão ao avaliar nossas instâncias no conjunto de
# treinamento usando apenas as árvores para as quais elas foram omitidas
print(rf.oob_score_)
print(rf.score(X_test, y_test))