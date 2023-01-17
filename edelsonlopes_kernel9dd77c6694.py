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
df=pd.read_csv('../input/train.csv')

df2=pd.read_csv('../input/test.csv')
df.head()
df.head().T
df2.head()
df.nunique()
df.info()
df['nota_mat']=np.log(df['nota_mat'])
df['codigo_mun'] = df['codigo_mun'].apply(lambda x: x.replace('ID_ID_', ''))

df['codigo_mun'] = df['codigo_mun'].values.astype('int64')

df2['codigo_mun'] = df2['codigo_mun'].apply(lambda x: x.replace('ID_ID_', ''))

df2['codigo_mun'] = df2['codigo_mun'].values.astype('int64')
df.dtypes
df2.head()
for c in df.columns:

    if df[c].dtype == 'object':

        df[c] = df[c].astype('category').cat.codes
for c in df2.columns:

    if df2[c].dtype == 'object':

        df2[c] = df2[c].astype('category').cat.codes
df.head()

df2.head()
df.dtypes
df.min().min()
df.fillna(-1.0, inplace=True)
df2.fillna(-1.0, inplace=True)
df.info()
df2.head()




df.loc[:, (df.nunique() > 2 ) & (df.nunique() < 8)].head()



for col in df.loc[:, (df.nunique() > 2 ) & (df.nunique() < 8)].columns:

    df = pd.concat([df, pd.get_dummies(df[col], prefix=col).iloc[:, :-1]], 

                   axis =1)

    del df[col]    
for col in df2.loc[:, (df2.nunique() > 2 ) & (df2.nunique() < 8)].columns:

    df2 = pd.concat([df2, pd.get_dummies(df2[col], prefix=col).iloc[:, :-1]], 

                   axis =1)

    del df2[col]    
df.head()
df2.head()
from sklearn.model_selection import train_test_split
treino, validacao = train_test_split(df, random_state=42)
treino.shape, validacao.shape
colunas_removidas=['nota_mat', 'municipio','codigo_mun']
feats=[c for c in df.columns if c not in colunas_removidas]
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR
models = {'RandomForest': RandomForestRegressor(random_state=42),

         'ExtraTrees': ExtraTreesRegressor(random_state=42),

         'GBM': GradientBoostingRegressor(random_state=42),

         'DecisionTree': DecisionTreeRegressor(random_state=42),

         'AdaBoost': AdaBoostRegressor(random_state=42),

         'KNN 1': KNeighborsRegressor(n_neighbors=1),

         'KNN 3': KNeighborsRegressor(n_neighbors=3),

         'KNN 11': KNeighborsRegressor(n_neighbors=11),

         'SVR': SVR(),

         'Linear Regression': LinearRegression()}

from sklearn.metrics import mean_squared_error
def run_model(model, train, valid, feats, y_name):

    model.fit(train[feats], train[y_name])

    preds = model.predict(valid[feats])

    return mean_squared_error(valid[y_name], preds)**(1/2)
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
scores = []

for name, model in models.items():

    score = run_model(model, treino, validacao, feats, 'nota_mat')

    scores.append(score)

    print(name+':', score)

pd.Series(scores, index=models.keys()).sort_values(ascending=False).plot.barh()
GBM=GradientBoostingRegressor(random_state=42, n_estimators=100)
GBM.fit(treino[feats], treino['nota_mat'])
valida_preds = GBM.predict(validacao[feats])
mean_squared_error(validacao['nota_mat'], valida_preds)**(1/2)
from sklearn.metrics import accuracy_score


mean_squared_error(validacao['nota_mat'], valida_preds)**(1/2)
df2['nota_mat'] = np.exp(GBM.predict(df2[feats]))
df2[['codigo_mun', 'nota_mat']]
df2[['codigo_mun', 'nota_mat']].to_csv('Edelson.csv', index=False)