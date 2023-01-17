import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline

print(os.listdir("../input"))
# Variavel contendo o nome do arquivo
arquivo = "../input/arq_municipios_fronteiricos.csv"

# Complete o código abaixo para realizar a leitura do arquivo
# o arquivo utiliza tabulação como separador
# dentro de strings a tabulação é representada como \t
df_muni_front = pd.read_csv(arquivo, dtype=str)


# Visualize as 5 primeiras linhas do DataFrame
df_muni_front.head(10)

# função que irá realizar o tratamento para um município
def tratar_nome_municipio(nome_municipio):
    #realize a limpeza
    nome_municipio_tratado = re.sub(r'([\d]* [–-] )(.*)', r'\2', nome_municipio)    
    return nome_municipio_tratado

# faça um teste de sua função para verificar se a transformação está correta
tratar_nome_municipio('1 – Aceguá')
# aplique a função utilizando o método apply
municipios_tratados = df_muni_front['Município'].apply(tratar_nome_municipio)
# exiba todos os dados e verifique se o resultado está correto
municipios_tratados.head()
# problemas com caracteres são comuns
print(hex(ord('-'))) # código UTF8 do caracter (hexadecimal)
print(hex(ord('–'))) # código UTF8 do caracter (hexadecimal)
'-' == '–'
# com todos os municípios devidamente tratados
# sobrescreva a coluna Município com os novos valores
df_muni_front.loc[:,'Município'] = municipios_tratados


# exiba as informações
df_muni_front.head(100)
# verifique os tipos das colunas do DataFrame. Utilize o método info(): df_muni_front
df_muni_front.info()
# faça a conversão do campo 'Área territorial'
def converter_para_float(texto):
    # faça as operações necessárias e 
    # devolva um objeto do tipo float
    # ou np.NaN (tipo Not a Number do numpy)
    t = texto.replace(' ','').replace('.','').replace(',','.');
    return float(t)

# aplique a função de conversão utilizando o método apply na coluna 'Área territorial'
area_territorial = df_muni_front['Área territorial'].apply(converter_para_float)
pib = df_muni_front['PIB (IBGE/2005'].apply(converter_para_float)

# exiba alguns valores da com valores convertidos
area_territorial.head()
pib.head
# substitua a coluna pelos valores convertidos
df_muni_front.loc[:, 'Área territorial'] = area_territorial

# faça o mesmo para a coluna PIB
df_muni_front.loc[:, 'PIB (IBGE/2005'] = pib
# imprima novamente as informações das colunas e verifique os tipos
df_muni_front.info()

# imprima novamente as primeiras linhas do DataFrame
df_muni_front.head(60)
# crie o set e verifique o seu conteúdo
nomes_estados = set(df_muni_front['Estado'])


# exiba o set gerado
nomes_estados
# crie o dicionário
dic_nomes_siglas = {
     'Acre':'AC',
     'Amapá':'AP',
     'Amazonas':'AM',
     'Mato Grosso':'MT',
     'Mato Grosso do Sul':'MS',
     'Paraná':'PR',
     'Pará':'PA',
     'Rio Grande do Sul':'RS',
     'Rondônia':'RO',
     'Roraima':'RR',
     'Santa Cataria':'SC',
     'Santa Catarina': 'SC'
}

# faça o mapeamento dos valores e atribua a : coluna_siglas_uf
coluna_siglas_uf = df_muni_front['Estado'].map(dic_nomes_siglas)


# verifique os 10 primeiros itens criados
coluna_siglas_uf.head(10)

# crie a coluna sigla
df_muni_front['Sigla'] = coluna_siglas_uf

# verifique as informações do dataframe
df_muni_front.head()

# verifique quantos registros possuem o nome do estado de Santa Catarina escrito errado "Santa Cataria"
len(df_muni_front.loc[df_muni_front['Estado'] == 'Santa Cataria'])
# faça a correção dos registros que possuem o nome do estado de Santa Catarina escrito errado
df_muni_front.loc[df_muni_front['Estado'] == 'Santa Cataria', 'Estado'] = 'Santa Catarina'
# verifique, novamente, quantos registros possuem o nome do estado de Santa Catarina escrito errado
len(df_muni_front.loc[df_muni_front['Estado'] == 'Santa Cataria'])
# normalize a coluna PIB em quantidade de desvios padrão
pib_desvios = df_muni_front['PIB (IBGE/2005'].apply(lambda x: (x - df_muni_front['PIB (IBGE/2005'].mean())/df_muni_front['PIB (IBGE/2005'].std())

# exiba um histograma para as informações de PIB normalizado
plt.figure(figsize=(15,6))
sns.distplot(pib_desvios.dropna(), bins=50)
plt.xlabel('PIB (Quantidade de Desvios)')
plt.ylabel('Frequência')
plt.show()

# quais cidades possuem mais de 2 desvios 
df_muni_front['pib_desvios'] = pib_desvios
df_muni_front.loc[df_muni_front['pib_desvios'] > 2, ['Município','pib_desvios']]
# quantos registros possuem NaN na coluna IDH/2000?
len(df_muni_front.loc[df_muni_front['IDH/2000'] == 'ni'])
# quantas cidades por estado?
df_muni_front.Sigla.value_counts()
# faça a ordenação do DataFrame pelo nome do município
df_muni_front.sort_values(by=['Município'])

# a ordenação está correta?
#Não. A cidade de Óbidos, acentuada, ficou incorretame