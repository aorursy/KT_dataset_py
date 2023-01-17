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
#imprime a área que está sendo utilizada

print(os.getcwd())

#importando a base de dados - amostra de base de clientes pf de banco descaracterizada 

train = pd.read_csv('../input/train.csv', sep=',', decimal='.')

test = pd.read_csv('../input/test.csv', sep=',', decimal='.')

sample_submission = pd.read_csv('../input/sample_submission.csv', sep=',', decimal='.')





# Descrição das variáveis

# Estatísticas básicas



# Verificar mínimos e máximos para garantir se estão dentro dos limites esperados

# Verificar intervalo de variação da medida

# Verificar possíveis outliers

train.describe()



#descrição do tipo de cada uma das variáveis do banco

train.dtypes
# Análise capital e nota

train.groupby(['nota_mat']).agg({'capital' : 'sum'})
# Análise capital e nota

train.groupby(['porte']).agg({'nota_mat' : 'sum'})

# Análise capital e nota

train.groupby(['regiao']).agg({'nota_mat' : 'sum'})
#CORRELATION MATRICES

import seaborn as sns

corr = data_final[['capital','populacao','area','densidade_dem','pib','pib_pc','participacao_transf_receita',

                   'servidores','comissionados','comissionados_por_servidor','perc_pop_econ_ativa',

                   'taxa_empreendedorismo','anos_estudo_empreendedor','jornada_trabalho','gasto_pc_saude',

                   'hab_p_medico','exp_vida','gasto_pc_educacao','exp_anos_estudo','nota_mat','porte_Grande porte',

                   'porte_Médio porte','porte_Pequeno porte 1','porte_Pequeno porte 2','regiao_CENTRO-OESTE','regiao_NORDESTE','regiao_NORTE','regiao_SUDESTE','regiao_SUL']]

#mask = np.zeros_like(corr, dtype=np.bool)

#mask[np.triu_indices_from(mask)]=True

#f, ax = plt.subplots(figsize=(11,9))

#cmap = sns.diverging_palette(220, 10, as_cmap=True)

#sns.heatmap(corr, mask=mask, cmap =cmap, vmax=1, vmin=-1, center=0, square=True, linewidth=.5,cbar_kws={"shrink":.5})

corr = data_final.corr()

ax = sns.heatmap(corr)

#sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220,10,as_cmap=True), square=True, ax=ax)
train.dtypes

#descrição do tipo de cada uma das variáveis do banco

#train.dtypes

train.head()
train.dtypes
train['populacao'] = train['populacao'].str.replace(',','').str.split('(').str[0].astype(float)

train['area'] = train['area'].str.replace(',','').str.split('(').str[0].astype(float)

train['densidade_dem'] = train['densidade_dem'].str.replace(',','').str.split('(').str[0].astype(float)
train['comissionados_por_servidor']= train['comissionados_por_servidor'].str.replace('#DIV/0!','0')

train['comissionados_por_servidor']= (train['comissionados_por_servidor'].str.replace('%','').astype(float))/100





train.isna().sum()







test = train[train['nota_mat'].isnull()]

train = train[~train['nota_mat'].isnull()]



train.shape, test.shape
train = train.fillna(0)

test = train.fillna(0)
train.shape, test.shape
train.isna().sum()
cat_vars= ['porte','regiao']

for var in cat_vars:

    cat_list='var'+'_'+var

    cat_list = pd.get_dummies(train[var], prefix=var)

    data1=train.join(cat_list)

    train=data1



cat_vars= ['porte','regiao']

data_vars=train.columns.values.tolist()

to_keep=[i for i in data_vars if i not in cat_vars]

from sklearn.model_selection import train_test_split
train, valid = train_test_split(train, random_state=42)
train.shape, valid.shape, test.shape
test = test.drop('nota_mat',axis=1)
removed_cols = ['nota_mat','capital','estado','porte','comissionados_por_servidor','municipio','regiao']
feats = [c for c in train.columns if c not in removed_cols]
feats
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, min_samples_split=3, max_depth=8, random_state=42, oob_score=True, n_jobs=-1)
rf.fit(train[feats],train['nota_mat'])
preds = rf.predict(valid[feats])
from sklearn.metrics import accuracy_score
accuracy_score(valid['nota_mat'],preds)
pd.Series(rf.feature_importances_,index=feats)
preds_test
test['nota_mat'] = preds_test
test[['codigo_mun', 'nota_mat']].to_csv('rf3.csv', index=False)