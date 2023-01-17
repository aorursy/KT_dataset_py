import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

cand_df = pd.read_csv("../input/consulta_cand_2018_BRASIL.csv",
                      error_bad_lines=False, sep=',',encoding='utf-8')
cand_df = cand_df.set_index('SQ_CANDIDATO')
cand_df.head()
bens_df = pd.read_csv("../input/bem_candidato_2018_BRASIL.csv",
                      error_bad_lines=False, sep=',',encoding='utf-8')
bens_df.head()
bens_df['VR_BEM_CANDIDATO'] = bens_df['VR_BEM_CANDIDATO'].apply(lambda x: float(str(x).split(',')[0]))

# Remove falhas na construção da planilha no valor dos bens com falha de digitação
bens_df = bens_df[bens_df['VR_BEM_CANDIDATO'] < 100000000]
bens_por_candidato = bens_df.groupby(['SQ_CANDIDATO'])['VR_BEM_CANDIDATO'].sum()

cand_df['BENS_AGG_SUM'] = bens_por_candidato

cand_df[['NM_URNA_CANDIDATO', 'BENS_AGG_SUM']].head()

import locale
top_candidatos = cand_df.set_index('NM_CANDIDATO').sort_values(by='BENS_AGG_SUM', ascending=False)['BENS_AGG_SUM'][:50][::-1]
fig, ax = plt.subplots()
x, y = list(top_candidatos.values), list(top_candidatos.index)
ax.bar(y, x)
ax.set_xticklabels(y, rotation='vertical')
fig.set_size_inches(18.5, 10.5)

def format_func(value, tick_number):
    return 'R$ {}'.format(float(value))

ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
bens_por_partido = cand_df.groupby(['SG_PARTIDO'])['BENS_AGG_SUM'].mean()
bens_por_partido = bens_por_partido.sort_values()
fig, ax = plt.subplots()
x, y = list(bens_por_partido.values), list(bens_por_partido.index)
fig.set_size_inches(18.5, 10.5)

ax.bar(y, x)
ax.set_xticklabels(y, rotation='vertical')
fig.suptitle('Média de Bens declarados por Deputados por Partido')
fig.savefig('top_partidos.png', bbox_inches='tight')