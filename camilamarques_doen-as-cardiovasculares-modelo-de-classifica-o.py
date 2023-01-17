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
# Leitura do dataset:



df = pd.read_csv('/kaggle/input/cardiovascular-disease-dataset/cardio_train.csv', sep=';')
# Verificando o tamanho do dataset: 



df.shape
# Algumas informações: 



df.info()
# Verificando as cinco primeiras linhas:



df.head()
# Verificando a correlação entre as variáveis:



df.corr()
# Verificando estatísticas descritivas:



df.describe()
# Selecionando as colunas para serem treinadas:



cols = [c for c in df.columns if c not in ['id','cardio']]
# Separando o dataset em treino, teste e validação:





from sklearn.model_selection import train_test_split



train, test = train_test_split(df, random_state = 42, test_size = 0.1)



train, valid = train_test_split(train, random_state = 42, test_size = 0.1)



print('Train Shape:', train.shape) 



print('Valid Shape:', valid.shape)



print('Test Shape:', test.shape)
# Treinando o modelo de Random Forest Classifier:



from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators = 200, n_jobs = -1, random_state = 42)



rf.fit(train[cols], train['cardio'])
# Fazendo previsões com o modelo treinado na base de validação:



pred_val = rf.predict(valid[cols])



pred_val
# Verificando o desempenho do modelo com a métrica na base de validação:



from sklearn.metrics import accuracy_score



accuracy_score(valid['cardio'], pred_val)
# Verificando a distribuição da target da base de validação:



valid['cardio'].value_counts(normalize=True)
from sklearn import svm

from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=rf,X=train[cols],y=train['cardio'],cv=10)



print(scores)
print (scores.mean())