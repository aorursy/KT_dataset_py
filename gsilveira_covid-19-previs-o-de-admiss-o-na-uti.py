# Tudo isso baseado no trabalho de: Rodrigo Cabrera Castaldoni, Felipe/Antonildes, equipe do sirio libanes
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
import pandas as pd
dados = pd.read_excel("/kaggle/input/covid19/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx")

dados.head()
dados.query("PATIENT_VISIT_IDENTIFIER==0")
dados.query("PATIENT_VISIT_IDENTIFIER==0").describe()
dados["AGE_PERCENTIL"].astype("category").cat.codes
dados["AGE_PERCENTIL"] = dados["AGE_PERCENTIL"].astype("category").cat.codes

dados.head()
dados = dados.fillna(dados.mean())

dados.head()
from sklearn.linear_model import LogisticRegression



modelo = LogisticRegression(max_iter = 1000)
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, roc_auc_score

import numpy as np



def roda_modelo(modelo, dados):

    np.random.seed(687423)



    X = dados.drop(["WINDOW", "ICU"], axis=1).values

    y = dados["ICU"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)

    modelo.fit(X_train, y_train)

    

    y_train_hat = modelo.predict(X_train)

    y_test_hat = modelo.predict(X_test)

    

    print(f"Train {roc_auc_score(y_train, y_train_hat)}")

    print(f"Teste {roc_auc_score(y_test, y_test_hat)}")

    

    print(classification_report(y_test, y_test_hat))

    
roda_modelo(modelo, dados)
features_continuas = dados.iloc[:,13:-2].columns

dados[features_continuas].head()
dados.groupby("PATIENT_VISIT_IDENTIFIER", as_index=False)[features_continuas].mean().head()
# PETROBRAS 15, 16, 17, 17, 16

# PETROBRAS 15, 15, 16, 17, 17
dados_fake = pd.DataFrame([[1, None, 1], [1, 2.5, 2], [2, 3, 1], [2, None, 2], [3,None,1],[3,1,2],[3,None,3],[4,None,1],[4,1,2],[4,None,3],[4,6,4],[4,None,5]], columns=["usuario", "medida", "janela"])

dados_fake
dados_fake.groupby("usuario", as_index=False)["medida"].apply(lambda d: d.fillna(method='bfill'))
dados_fake.groupby("usuario", as_index=False)["medida"].apply(lambda d: d.fillna(method='ffill'))
dados_fake.groupby("usuario", as_index=False)["medida"].apply(lambda d: d.fillna(method='bfill').fillna(method='ffill'))
dados = pd.read_excel("/kaggle/input/covid19/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx")

dados.head()
dados.groupby("PATIENT_VISIT_IDENTIFIER", as_index=False)[features_continuas].apply(lambda d: d.fillna(method='bfill').fillna(method='ffill')).head()
def preenche_tabela(dados):

    saida = dados.iloc[:,-2:]

    features_categoricas = dados.iloc[:,:13]

    features_continuas_colunas = dados.iloc[:,13:-2].columns

    

    features_continuas_preenchidas = dados.groupby("PATIENT_VISIT_IDENTIFIER", as_index=False)[features_continuas_colunas].apply(lambda d: d.fillna(method='bfill').fillna(method='ffill'))

    dados_finais = pd.concat([features_categoricas, features_continuas_preenchidas, saida], axis=1, ignore_index=True)

    dados_finais.columns = dados.columns

    return dados_finais



# talvez?

# dados[features_continuas_colunas] = features_continuas_preenchidas
dados_limpos = preenche_tabela(dados)

dados_limpos.describe()
a_remover = dados_limpos.query("WINDOW=='0-2' and ICU==1")["PATIENT_VISIT_IDENTIFIER"].values

a_remover = ",".join(a_remover.astype(str))

dados_limpos = dados_limpos.query(f"PATIENT_VISIT_IDENTIFIER not in [{a_remover}]")

dados_limpos.head()
def prepare_window(x):

    if(np.any(x["ICU"])):

        x.loc[x["WINDOW"]=="0-2", "ICU"] = 1

    return x[x["WINDOW"]=="0-2"]



dados_limpos = dados_limpos.groupby("PATIENT_VISIT_IDENTIFIER").apply(prepare_window)

dados_limpos.head()
dados_limpos = dados_limpos.set_index("PATIENT_VISIT_IDENTIFIER")

dados_limpos = dados_limpos.dropna(how="any")

dados_limpos.head()
dados_limpos["AGE_PERCENTIL"] = dados_limpos["AGE_PERCENTIL"].astype("category").cat.codes
modelo = LogisticRegression(max_iter = 1000)

roda_modelo(modelo, dados_limpos)
from sklearn.dummy import DummyClassifier



roda_modelo(DummyClassifier(strategy="stratified"), dados_limpos)