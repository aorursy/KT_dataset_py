f = open('../input/train_ver2.csv','r')

g = open('last_month.csv','w')



for line in f:

    date = line[:10]

    if date == '2016-05-28':

        g.write(line)
with open('../input/train_ver2.csv', 'r') as f:

    cols = f.readline().split(',')
cols = [s.replace('"', '') for s in cols]
import numpy as np

import pandas as pd
df = pd.read_csv('last_month.csv',dtype={'indrel_1mes': str, 'conyuemp':str},names=cols)
df_features = df.iloc[:,:24]
df_features.head()
del df_features['fecha_dato']

del df_features['ncodpers']
df_features['tipodom'].value_counts()
del df_features['tipodom']
(df_features.isnull().sum()/len(df_features)).sort_values()
del df_features['ult_fec_cli_1t']

del df_features['conyuemp']
from sklearn import preprocessing



for f in df_features.columns:

    if df_features[f].dtype == 'object':

        print(f)

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(df_features[f].values))

        df_features[f] = lbl.transform(list(df_features[f].values))
df_features.corr()['renta'].sort_values()
df_features['segmento'].value_counts()
df_features['ind_actividad_cliente'].value_counts()
median_renta = np.zeros((2,4))

for i in range(2):

    for j in range (4):

        median_renta[i][j] = df_features[(df_features['ind_actividad_cliente'] == i) & \

                                         (df_features['segmento'] == j)]['renta'].dropna().median()
median_renta
for i in range(0, 2):

    for j in range(0, 4):

        df_features.loc[(df_features['renta'].isnull()) & \

                        (df_features['ind_actividad_cliente'] == i) & \

                        (df_features['segmento'] == j), 'renta'] = median_renta[i][j]
(df_features.isnull().sum()/len(df_features)).sort_values()
df_features['cod_prov'] = df_features['cod_prov'].fillna(df_features['cod_prov'].median())
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_regression

import operator
X = df_features.values

test = SelectKBest(score_func=f_regression)

prod_cols = list(df.columns[24:48])
d = {}

for p in prod_cols:

    y = np.array(df[p])

    fit = test.fit(X, y)

    l = zip(df_features.columns, np.around(fit.scores_))

    d[p] = sorted(l, key=lambda x: x[1], reverse=True)
df_ranking = pd.DataFrame(index=df_features.columns,columns=prod_cols)
for p in prod_cols:

    i = 0

    for r in d[p]:

        df_ranking[p][r[0]] = i

        i += 1
df_ranking['total'] = df_ranking.sum(axis=1)
df_ranking.sort_values('total')
df_ranking.sort_values('total')['total'].plot(kind='bar')