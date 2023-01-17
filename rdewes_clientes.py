# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')

df.head(6).T
# Fazendo uma cópia do dataframe

df_raw = df.copy()
df.info()
brancos = df[df['TotalCharges']==' '].index

df.iloc[brancos]
# Corrigindo a coluna TotalCharges

df['TotalCharges'] = df['TotalCharges'].str.replace(' ', '-1').astype(float)
df.info()
# Transformando dados categóricos em numéricos

for col in df.columns:

    if df[col].dtype == 'object':

        df[col] = df[col].astype('category').cat.codes
df.info()
df.head()
# Divisão do dataframe

# Importando as funções

from sklearn.model_selection import train_test_split



# Separando os dados em treino e teste

train, test = train_test_split(df, test_size=0.2, random_state=42)
# Separando em treino e validação

train, valid = train_test_split(train, test_size=0.2, random_state=42)
train.shape, valid.shape, test.shape
feats = [c for c in df.columns if c not in ['customerID', 'Churn']]
# Treinando o modelo

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=200, min_samples_split=5, max_depth=4, random_state=42, n_jobs=-1)
rf.fit(train[feats], train['Churn'])
preds = rf.predict(valid[feats])
# avaliando o desempenho do modelo

from sklearn.metrics import accuracy_score

accuracy_score(valid['Churn'], preds)
valid['Churn'].value_counts()
# Verificando o Dataframe completo

df['Churn'].value_counts(normalize=True)
(valid['Churn']==0).mean()
# Rodando o modelo para prever o churn nos dados de teste

accuracy_score(test['Churn'], rf.predict(test[feats]))
# Analisando a importância das caracteristicas 

pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()
from sklearn.metrics import confusion_matrix

pd.DataFrame(confusion_matrix(valid['Churn'], preds))
from yellowbrick.classifier import ConfusionMatrix

cm = ConfusionMatrix(rf)

cm.fit(train[feats], train['Churn'])

cm.score(test[feats], test['Churn'])

cm.poof()