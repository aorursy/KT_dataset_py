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
import numpy as np

import pandas as pd 
test = pd.read_csv('../input/test.csv')

train = df = pd.read_csv('../input/train.csv')
#Juntando as bases
df = pd.concat([train, test])
#retirando o ID_ID_ do campo codigo_mun
df['codigo_mun'] = df['codigo_mun'].apply(lambda x: x.replace('ID_ID_', ''))

df['codigo_mun'] = df['codigo_mun'].values.astype('int64')
texto = df.select_dtypes(include='object').columns
for c in texto:

    df[c]=df[c].astype('category').cat.codes
#selecioanando os campos -nota_mat e codigo_municipio
feats = [c for c in df.columns if c not in ['nota_mat', 'cod_mun', 'Unnamed: 0']]
for c in feats:

    df[c].fillna(df2[c+'_mean'], inplace=True)
df.sample(10)
df['nota_mat'] = np.log(df['nota_mat'])
test = df[df['nota_mat'].isnull()]

train = df[df['nota_mat'].notnull()]
from sklearn.model_selection import train_test_split
train, valid = train_test_split(train, random_state = 10)
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression
models = {'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=500),

         'ExtraTrees': ExtraTreesRegressor(random_state=42, n_estimators=500),

              'Linear Regression': LinearRegression()}
from sklearn.metrics import mean_squared_error
def run_model(model, train, valid, feats, y_name):

    model.fit(train[feats], train[y_name])

    preds = model.predict(valid[feats])

    return mean_squared_error(valid[y_name], preds)**(1/2)
scores = []

names = []

for name, model in models.items():

    score = run_model(model, train, valid, feats, 'nota_mat')

    names.append(name)

    scores.append(score)

scores = pd.DataFrame([names, scores])

scores = scores.T

scores.columns = ['model_name', 'score']

scores = scores.sort_values(['score'])

print(scores)
#Usar o modelo Extratrees por ter menor valor
model = models['ExtraTrees']

model.fit(train[feats], train['nota_mat'])

preds = model.predict(valid[feats])
#predição dos valores
test['nota_mat'] = model.predict(test[feats])
test['nota_mat'] = np.exp(test['nota_mat'])
test[['codigo_mun','nota_mat']].to_csv('rf10.csv', index=False)