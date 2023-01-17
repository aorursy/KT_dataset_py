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
res_aluno = pd.read_csv('../input/TS_RESULTADO_ALUNO.csv', sep=';', decimal=',')
quest_aluno = pd.read_csv('../input/TS_QUEST_ALUNO.csv', sep=';', decimal=',')
res_aluno.shape, quest_aluno.shape
res_aluno.info()
res_aluno.head(10)
quest_aluno.sample(10)
import seaborn as sns



# sns.heatmap(res_aluno, dropna=True)

# sns.pairplot(res_aluno, dropna=True)  # demora muito
# Filtrando dados úteis

res_aluno = res_aluno[res_aluno['IN_PROFICIENCIA']==1]  # Somente as que contém o mínimo de respostas



# Removendo peso

remover = 2000000

drop_indices = np.random.choice(res_aluno.index, remover, replace=False)

res_aluno = res_aluno.drop(drop_indices)
# Limpeza ALUNO



colunas = ['PESO', 'PROFICIENCIA_LP', 'DESVIO_PADRAO_LP', 'PROFICIENCIA_LP_SAEB', 'DESVIO_PADRAO_LP_SAEB',

           'PROFICIENCIA_MT', 'DESVIO_PADRAO_MT', 'PROFICIENCIA_MT_SAEB', 'DESVIO_PADRAO_MT_SAEB', 'ID_TURNO']

res_aluno[colunas] = res_aluno[colunas].applymap(lambda x: str(x).replace(',', '.'))  # applymap funcionou outros map, apply nops

res_aluno.replace(r'^\s*$', np.nan, regex=True, inplace=True)  # limpa campos que só tem espaço em branco

res_aluno.replace('nan', np.nan, regex=True, inplace=True)  # limpa campos que tem escrito 'nan'

res_aluno[colunas] = res_aluno[colunas].apply(pd.to_numeric)

res_aluno.info()
res_aluno.isna().sum()
# A função abaixo foi obtida em https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65

def reduce_mem_usage(props):

    """Função para reduzir a memória dos Dataframes"""

    start_mem_usg = props.memory_usage().sum() / 1024**2 

    print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in props.columns:

        if props[col].dtype != object:  # Exclude strings

            

            # Print current column type

            print("******************************")

            print("Column: ",col)

            print("dtype before: ",props[col].dtype)

            

            # make variables for Int, max and min

            IsInt = False

            mx = props[col].max()

            mn = props[col].min()

            

            # Integer does not support NA, therefore, NA needs to be filled

            if not np.isfinite(props[col]).all(): 

                NAlist.append(col)

                props[col].fillna(mn-1,inplace=True)  

                   

            # test if column can be converted to an integer

            asint = props[col].fillna(0).astype(np.int64)

            result = (props[col] - asint)

            result = result.sum()

            if result > -0.01 and result < 0.01:

                IsInt = True



            

            # Make Integer/unsigned Integer datatypes

            if IsInt:

                if mn >= 0:

                    if mx < 255:

                        props[col] = props[col].astype(np.uint8)

                    elif mx < 65535:

                        props[col] = props[col].astype(np.uint16)

                    elif mx < 4294967295:

                        props[col] = props[col].astype(np.uint32)

                    else:

                        props[col] = props[col].astype(np.uint64)

                else:

                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                        props[col] = props[col].astype(np.int8)

                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                        props[col] = props[col].astype(np.int16)

                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                        props[col] = props[col].astype(np.int32)

                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                        props[col] = props[col].astype(np.int64)    

            

            # Make float datatypes 32 bit

            else:

                props[col] = props[col].astype(np.float32)

            

            # Print new column type

            print("dtype after: ",props[col].dtype)

            print("******************************")

    

    # Print final result

    print("___MEMORY USAGE AFTER COMPLETION:___")

    mem_usg = props.memory_usage().sum() / 1024**2 

    print("Memory usage is: ",mem_usg," MB")

    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")

    return props, NAlist
res_aluno, _ = reduce_mem_usage(res_aluno)

quest_aluno, _ = reduce_mem_usage(quest_aluno)
# Criando a coluna alvo

mediana = res_aluno['PROFICIENCIA_LP'].median()

print(mediana)



res_aluno['mediana_lp'] = res_aluno['PROFICIENCIA_LP'] >= mediana
# Criandos os dummies do res_aluno

dummies_res = ['ID_UF', 'ID_DEPENDENCIA_ADM', 'ID_LOCALIZACAO', 'ID_SERIE', 'ID_TURNO']

res_aluno = pd.get_dummies(data=res_aluno, columns=dummies_res)
# Juntando os dummies ao dataframe

# res_aluno = res_aluno[['ID_ALUNO', 'IN_SITUACAO_CENSO', 'IN_PROFICIENCIA', 'PROFICIENCIA_MT', 'PROFICIENCIA_LP']]

# res_aluno = pd.concat([res_aluno, dummies_res], axis=1)

# res_aluno.info()
# Criando os dummies do quest_aluno

colunas_quest = ['ID_ALUNO', 'TX_RESP_Q001', 'TX_RESP_Q002', 'TX_RESP_Q004', 'TX_RESP_Q005', 'TX_RESP_Q012', 'TX_RESP_Q013', 'TX_RESP_Q017',

                 'TX_RESP_Q018', 'TX_RESP_Q020', 'TX_RESP_Q022', 'TX_RESP_Q024', 'TX_RESP_Q027', 'TX_RESP_Q033', 'TX_RESP_Q038',

                 'TX_RESP_Q039', 'TX_RESP_Q044', 'TX_RESP_Q045', 'TX_RESP_Q046', 'TX_RESP_Q047', 'TX_RESP_Q049', 'TX_RESP_Q050',

                 'TX_RESP_Q051', 'TX_RESP_Q052']

quest_aluno = quest_aluno[colunas_quest]
# Juntando os dataframes

colunas_quest.remove('ID_ALUNO')

quest_aluno = pd.get_dummies(data=quest_aluno, columns=colunas_quest)
alunos = res_aluno.merge(quest_aluno, on='ID_ALUNO', how='left')
alunos.info()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



train, test = train_test_split(alunos, random_state=42, test_size=0.3)



cols = [c for c in train.columns if c not in ['mediana_lp', 'PROFICIENCIA_LP', 'PROFICIENCIA_LP_SAEB', 

                                              'DESVIO_PADRAO_LP_SAEB', 'ID_MUNICIPIO', 'ID_PROVA_BRASIL', 

                                              'ID_TURMA','ID_ALUNO', 'IN_SITUACAO_CENSO', 'IN_PREENCHIMENTO', 

                                              'PESO', 'IN_PROFICIENCIA', 'DESVIO_PADRAO_MT', 'DESVIO_PADRAO_LP',

                                             'PROFICIENCIA_MT_SAEB', 'DESVIO_PADRAO_MT_SAEB', 'ID_ESCOLA']]
print(cols)
from sklearn.linear_model import LogisticRegression



lg = LogisticRegression(n_jobs=-1)

lg.fit(train[cols], train['mediana_lp'])
preds2 = lg.predict(test[cols])

accuracy_score(test['mediana_lp'], preds2)
rf = RandomForestClassifier(n_jobs=-1, n_estimators=25)

rf.fit(train[cols], train['mediana_lp'])
preds = rf.predict(test[cols])

accuracy_score(test['mediana_lp'], preds)
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))

pd.Series(rf.feature_importances_[:10], index=cols[:10]).sort_values().plot.barh()
# fig = plt.figure(figsize=(10, 10))

# pd.Series(lg.feature_importances_, index=cols).sort_values().plot.barh()