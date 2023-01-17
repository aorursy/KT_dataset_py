import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as matplot

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
microdados = pd.read_csv('/kaggle/input/enem-por-escola-2005-a-2015/microdados_enem_por_escola/DADOS/MICRODADOS_ENEM_ESCOLA.csv',

                         sep=';',

                         na_filter=False,

                         engine='python')
# Dados Completos

microdados

# Dados do DF

dado = microdados.where(microdados['SG_UF_ESCOLA'] == 'DF')

microdadosDF = dado[pd.notnull(dado['SG_UF_ESCOLA'])]

microdados = microdadosDF # Retirar linha para usar dados globais

#

microdados
# microdados.dtypes # Tipo dos dados

# microdados.info() # Informações de tipo

# Mudando tipos 

microdados = microdados.astype({

    "NU_ANO":'int',

    "CO_UF_ESCOLA":'int',

    "CO_MUNICIPIO_ESCOLA":'int',

    "CO_ESCOLA_EDUCACENSO":'int',

    "TP_DEPENDENCIA_ADM_ESCOLA":'int',

    "TP_LOCALIZACAO_ESCOLA":'int',

    "NU_MATRICULAS":'int',

}) 

microdados.info()
microdados
microdados.groupby('SG_UF_ESCOLA').count() # Quantidade de dados por Estado

# microdados.groupby('SG_UF_ESCOLA').count()['NU_ANO'][0]
anosAnalizados = microdados.groupby('NU_ANO')

arrayAnosAnalizados = anosAnalizados.size().keys()

print('Lista de anos analizados: ')

print(arrayAnosAnalizados)
print(sorted(arrayAnosAnalizados, reverse=True)[0])

grupo_2015 = anosAnalizados.get_group(sorted(arrayAnosAnalizados, reverse=True)[0])

print(grupo_2015['PORTE_ESCOLA'].count()) # Quantidade total de escolas no ano de 2015
dict_porte_por_escola= dict(grupo_2015.groupby('PORTE_ESCOLA').size())

dict_porte_por_escola # Quantidade de escolas por porte
def customLabelInterna(pct):

    if pct:

        total = grupo_2015['PORTE_ESCOLA'].count()

        return '{} escolas \n({:.2f}%) '.format(int(round(pct*total/100.0)), pct)



fig, ax = matplot.subplots(figsize=(7,10))

wedges, texts, autotexts  = ax.pie(dict_porte_por_escola.values(),

            labels=dict_porte_por_escola.keys(),

            counterclock=False,

            textprops={'fontsize': 14},

            autopct=lambda pct: customLabelInterna(pct),

            shadow=True)

ax.legend(loc='upper right', title="Porte") # Quadro de legendas

ax.set_title("Quantidade de Escolas por Porte em 2015 ({} escolas)".format(grupo_2015['PORTE_ESCOLA'].count())) # Titulo do grafico

matplot.setp(autotexts, size=10, weight="bold") # tamanho de label inside

matplot.show()
import copy

dict_inse_por_escola= dict(grupo_2015.groupby('INSE').size())

ordenado = copy.deepcopy(dict_inse_por_escola)

ordenado['Grupo 2'] = 0

ordenado = dict(sorted(ordenado.items()))

matplot.bar([1,2,3,4,5,6],

            ordenado.values(),

            color=['#8e44ad','#16a085', '#27ae60', '#2980b9', '#2c3e50', '#95a5a6'])

# matplot.xticks([1,2,3,4,5,6], ordenado.keys(),fontsize=10, rotation=30,horizontalalignment='right')

matplot.xticks([1,2,3,4,5,6], ordenado.keys(),fontsize=10)

matplot.ylabel('Escolas', fontsize=12)

matplot.xlabel('INSE', fontsize=12)

matplot.title("Quantidade de Escolas\n por \nCategoria do INSE em 2015 ({} escolas)".format(grupo_2015['INSE'].count()),fontweight='bold')

matplot.show()
grupo_2015.TP_LOCALIZACAO_ESCOLA = grupo_2015.TP_LOCALIZACAO_ESCOLA.replace(to_replace=[1], value='Urbana') 

grupo_2015.TP_LOCALIZACAO_ESCOLA = grupo_2015.TP_LOCALIZACAO_ESCOLA.replace(to_replace=[2], value='Rural')

dictLocalizacaoEscola = dict(grupo_2015.groupby('TP_LOCALIZACAO_ESCOLA').size())
def customLabelInterna(pct):

    if pct:

        total = grupo_2015['TP_LOCALIZACAO_ESCOLA'].count()

        return '{} escolas \n({:.2f}%) '.format(int(round(pct*total/100.0)), pct)



fig, ax = matplot.subplots(figsize=(6,6))

wedges, texts, autotexts  = ax.pie(dictLocalizacaoEscola.values(),

            labels=dictLocalizacaoEscola.keys(),

            counterclock=False,

            colors=['#27ae60', '#2980b9'],

            textprops={'fontsize': 14},

            wedgeprops=dict(width=0.7),

            autopct=lambda pct: customLabelInterna(pct))

ax.legend(loc='lower left', title="Porte", bbox_to_anchor=(-0.15, 0.1)) # Quadro de legendas

ax.set_title("Localização de Escolas em 2015 ({} escolas)".format(grupo_2015['TP_LOCALIZACAO_ESCOLA'].count()), fontweight='bold') # Titulo do grafico

matplot.setp(autotexts, size=10, weight="bold", color='w') # tamanho de label inside

matplot.show()
grupo_2015.TP_DEPENDENCIA_ADM_ESCOLA = grupo_2015.TP_DEPENDENCIA_ADM_ESCOLA.replace(to_replace=[1], value='Federal') 

grupo_2015.TP_DEPENDENCIA_ADM_ESCOLA = grupo_2015.TP_DEPENDENCIA_ADM_ESCOLA.replace(to_replace=[2], value='Estadual')

grupo_2015.TP_DEPENDENCIA_ADM_ESCOLA = grupo_2015.TP_DEPENDENCIA_ADM_ESCOLA.replace(to_replace=[3], value='Municipal')

grupo_2015.TP_DEPENDENCIA_ADM_ESCOLA = grupo_2015.TP_DEPENDENCIA_ADM_ESCOLA.replace(to_replace=[4], value='Privada')

dictTipoEscola = dict(grupo_2015.groupby('TP_DEPENDENCIA_ADM_ESCOLA').size())

dictTipoEscola
import copy

dict_tipo_escola = dict(grupo_2015.groupby('TP_DEPENDENCIA_ADM_ESCOLA').size())

ordenado = copy.deepcopy(dict_tipo_escola)

ordenado['Municipal'] = 0

ordenado = dict(sorted(ordenado.items(), key=lambda x: x[1]))

fig, ax = matplot.subplots(figsize=(6, 6))

rects = ax.bar([1,2,3,4],

            ordenado.values(),

            color=['#8e44ad','#16a085', '#27ae60', '#2980b9', '#2c3e50', '#95a5a6'])

ax.set_xticks([1,2,3,4])

ax.set_xticklabels(ordenado.keys(),fontdict = {'fontsize':15})

ax.set_ylabel('Quantidade', fontsize=12)

ax.set_xlabel('Escola', fontsize=12)

for rect in rects:

    height = rect.get_height()

    ax.annotate('{}'.format(height),

                fontsize=12,

                weight="bold",

                xy=(rect.get_x() + rect.get_width() / 2, height),

                xytext=(0, 3),  # 3 points vertical offset

                textcoords="offset points",

                ha='center')

ax.set_title("Tipos de Escola em 2015 ({} escolas)".format(grupo_2015['TP_DEPENDENCIA_ADM_ESCOLA'].count()),fontweight='bold')

matplot.show()
# NU_MEDIA_CN --- Média das notas de Ciências da Natureza do Ensino Médio Regular.

a = grupo_2015.groupby('TP_DEPENDENCIA_ADM_ESCOLA')

# filtrar por privadas

grupo_2015_privada = grupo_2015[grupo_2015['TP_DEPENDENCIA_ADM_ESCOLA'] == 'Privada']

grupo_2015_privada
grupo_2015_privada = grupo_2015_privada.astype({

    "NU_MEDIA_CN": "float",

    "NU_MEDIA_CH": "float",

    "NU_MEDIA_LP": "float",

    "NU_MEDIA_MT": "float",

    "NU_MEDIA_RED": "float"

    })
valores = [

    round(grupo_2015_privada.NU_MEDIA_CN.mean(),3),

    round(grupo_2015_privada.NU_MEDIA_CH.mean(),3),

    round(grupo_2015_privada.NU_MEDIA_LP.mean(),3),

    round(grupo_2015_privada.NU_MEDIA_MT.mean(),3),

    round(grupo_2015_privada.NU_MEDIA_RED.mean(),3)

]

labels = [

    'Ciências da Natureza',

    'Ciências Humanas',

    'Linguagens e Códigos',

    'Matemática',

    'Redação'

]

y_pos = np.arange(len(labels))

fig, ax = matplot.subplots(figsize=(12,5))

rects = ax.barh( y_pos,

            valores,

            color=['#8e44ad','#16a085', '#27ae60', '#2980b9', '#2c3e50', '#95a5a6'],

            align='center')

ax.set_yticklabels(labels,fontdict = {'fontsize':15})

ax.set_xlabel('Notas', fontsize=12)

ax.set_yticks(y_pos)

ax.invert_yaxis()

for i, v in enumerate(valores):

    ax.text(v * 0.9, i + .1, str(v), color='w', fontweight='bold')

ax.set_title("Media de notas das escolas privadas em 2015 ({} escolas)".format(grupo_2015['TP_DEPENDENCIA_ADM_ESCOLA'].count()),fontweight='bold')

matplot.show()
grupo_2015_nao_privada = grupo_2015[grupo_2015['TP_DEPENDENCIA_ADM_ESCOLA'] != 'Privada']

grupo_2015_nao_privada

grupo_2015_nao_privada = grupo_2015_nao_privada.astype({

    "NU_MEDIA_CN": "float",

    "NU_MEDIA_CH": "float",

    "NU_MEDIA_LP": "float",

    "NU_MEDIA_MT": "float",

    "NU_MEDIA_RED": "float"

    })
valores = [

    round(grupo_2015_nao_privada.NU_MEDIA_CN.mean(),3),

    round(grupo_2015_nao_privada.NU_MEDIA_CH.mean(),3),

    round(grupo_2015_nao_privada.NU_MEDIA_LP.mean(),3),

    round(grupo_2015_nao_privada.NU_MEDIA_MT.mean(),3),

    round(grupo_2015_nao_privada.NU_MEDIA_RED.mean(),3)

]

labels = [

    'Ciências da Natureza',

    'Ciências Humanas',

    'Linguagens e Códigos',

    'Matemática',

    'Redação'

]

y_pos = np.arange(len(labels))

fig, ax = matplot.subplots(figsize=(12,5))

rects = ax.barh( y_pos,

            valores,

            color=['#8e44ad','#16a085', '#27ae60', '#2980b9', '#2c3e50', '#95a5a6'],

            align='center')

ax.set_yticklabels(labels,fontdict = {'fontsize':15})

ax.set_xlabel('Notas', fontsize=12)

ax.set_yticks(y_pos)

ax.invert_yaxis()

for i, v in enumerate(valores):

    ax.text(v * 0.9, i + .1, str(v), color='w', fontweight='bold')

ax.set_title("Media de notas das escolas publicas em 2015 ({} escolas)".format(grupo_2015_nao_privada['NU_MEDIA_CN'].count()),fontweight='bold')

matplot.show()
dado = grupo_2015.where(grupo_2015['TP_LOCALIZACAO_ESCOLA'] == 'Rural')

dadosRural = dado[pd.notnull(dado['TP_LOCALIZACAO_ESCOLA'])]

dadosRural = dadosRural.astype({

    "NU_TAXA_ABANDONO":'float'

})

mediaAbandonoRural = dadosRural['NU_TAXA_ABANDONO'].mean()



dado = grupo_2015.where(grupo_2015['TP_LOCALIZACAO_ESCOLA'] == 'Urbana')

dadosUrbana = dado[pd.notnull(dado['TP_LOCALIZACAO_ESCOLA'])]

dadosUrbana = dadosUrbana.astype({

    "NU_TAXA_ABANDONO":'float'

})

mediaAbandonoUrbana = dadosUrbana['NU_TAXA_ABANDONO'].mean()

mediaAbandonoRural = round(mediaAbandonoRural,2)

mediaAbandonoUrbana = round(mediaAbandonoUrbana,2)

medias = {"Urbana":mediaAbandonoUrbana,

          "Rural":mediaAbandonoRural}

mediasGrafico = dict(medias)



ordenado = dict(sorted(medias.items()))

matplot.bar([1,2],

            ordenado.values(),

            color=['#8e44ad','#16a085', '#27ae60', '#2980b9', '#2c3e50', '#95a5a6'])

matplot.xticks([1,2], ordenado.keys(),fontsize=10)

matplot.ylabel('Media', fontsize=12)

matplot.xlabel('Tipo', fontsize=12)

matplot.title("Media de abandono por tipo de escola ({} escolas)".format(grupo_2015['INSE'].count()),fontweight='bold')

matplot.show()

IDEB_ESCOLA_ENDERECOS = pd.read_csv("../input/ideb-escolas/IDEB_ESCOLA_ENDERECOS.csv", usecols=['Código','Nome','Endereço'])

IDEB_ESCOLA_ENDERECOS = IDEB_ESCOLA_ENDERECOS.rename(columns={'Código': 'CO_ESCOLA_EDUCACENSO'})

IDEB_ESCOLA_ENDERECOS

grupo_2015
result = pd.merge(IDEB_ESCOLA_ENDERECOS,grupo_2015,how='inner', on='CO_ESCOLA_EDUCACENSO')

result