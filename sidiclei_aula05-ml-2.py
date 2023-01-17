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



df.head().T

# Fazendo um cópia do dataframe

df_raw = df.copy()
# Verificar o dataframe

df.info()
# Corrigindo a coluna TotalCharges e convertendo em FLOAT

df['TotalCharges'] = df['TotalCharges'].str.replace(' ','-1').astype(float)
df.info()
# Transformando dados categorias em números

for col in df.columns:

    if df[col].dtype == 'object':

        df[col] = df[col].astype('category').cat.codes
df.info()
df.head()


# Importandoa função

from sklearn.model_selection import train_test_split
# Separação do dataframe em treino e teste

train, test = train_test_split(df, test_size=0.2,random_state=42)
# Separando o dataframe de treino em treino e validação

train, valid = train_test_split(train,test_size=0.2,random_state=42)
train.shape, valid.shape, test.shape
# Selecionar as colunas para treinamento

feats = [c for c in df.columns if c not in['customerID','Churn']]
# Importar o RandomForest

from sklearn.ensemble import RandomForestClassifier
# Instanciar o RandomForest

rf = RandomForestClassifier(n_estimators=200, min_samples_split=5,max_depth=4,random_state=42)
# Treinamento do modelo

rf.fit(train[feats],train['Churn'])
preds = rf.predict(valid[feats])
# Avaliando o desempenho do modelo

from sklearn.metrics import accuracy_score



accuracy_score(valid['Churn'], preds)
df['Churn'].value_counts(normalize=True)
(valid['Churn']== 0).mean()
test['Churn'].value_counts(normalize=True)
# Rodar o modelo para prever o Churn dos dados de teste

accuracy_score(test['Churn'], rf.predict(test[feats]))
# Analisando a importância das características

pd.Series(rf.feature_importances_,index=feats).sort_values().plot.barh()
# Matriz de confusao

from sklearn.metrics import confusion_matrix

confusion_matrix(test['Churn'],rf.predict(test[feats]))
pd.DataFrame(confusion_matrix(test['Churn'],rf.predict(test[feats])))
from yellowbrick.classifier import ConfusionMatrix

viz = ConfusionMatrix(RandomForestClassifier())

viz.fit(train[feats],train['Churn'])

viz.score(test[feats],test['Churn'])

viz.poof()