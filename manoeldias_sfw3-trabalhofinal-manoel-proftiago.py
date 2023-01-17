import sqlite3 # Declara que vamos utilizar o SQlite para ler dados

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
#Conexão no kaggle

conexao = sqlite3.connect('../input/amostra10pc/amostra10pc.sqlite')

#conexao = sqlite3.connect('dados/amostra10pc.sqlite')

#conexao = sqlite3.connect('/content/drive/My Drive/Pós Ciência de Dados/Modulo 3/SOFW3/dadosTrabalho/amostra5pc.sqlite')

# Carrega os dados do CNPJ

#df_CNPJ10 = pd.read_sql_query("SELECT tipo_de_registro, razao_social, nome_fantasia, situacao_cadastral, data_situacao_cadastral, motivo_situacao_cadastral, codigo_natureza_juridica, data_inicio_atividade, cnae_fiscal, uf, codigo_municipio, municipio, capital_social_empresa, porte_empresa, fim_registro FROM cnpj_dados_cadastrais_pj where situacao_cadastral in (02,08) and (data_situacao_cadastral like ('%2008%') or data_situacao_cadastral like ('%2020%'))", conexao)

df_CNPJ = pd.read_sql_query("SELECT * FROM cnpj_dados_cadastrais_pj", conexao)
# Informações sobre o dataset

print ("O dataset possui ", df_CNPJ.shape[0], "linhas e ", df_CNPJ.shape[1], "colunas.")
# Verificados os primeiros registros do dataset

df_CNPJ.head()
# Carrega os dados dos cnaes

df_desc_cnae = pd.read_sql_query("SELECT cod_secao, nm_secao, cod_cnae, nm_cnae FROM tab_cnae", conexao)

df_desc_cnae.head()
# Carrega os dados das UFs

#df_uf_regiao = pd.read_csv('/content/drive/My Drive/analisePosGraduacao/dados/uf_regiao.csv')



# Carregando no kaggle

df_uf_regiao = pd.read_csv('../input/uf-regiao/uf_regiao.csv')

df_uf_regiao.head()
def tabela_categoricas(df,col,name_col):

    df_1 = pd.DataFrame(df[col].value_counts()).reset_index()

    df_1.columns = [name_col,'Qtd empresas']

    df_1['%Total'] = (df_1['Qtd empresas']/len(df))*100

    return df_1
def distribuicaoNumericas(df,col,nameCol):

    med = pd.DataFrame(columns=[nameCol], index=['Media', 'Mediana', 'Minimo','Maximo'])

    med.loc['Media'][nameCol] = float(df[col].mean())

    med.loc['Mediana'][nameCol] = float(df[col].median())

    med.loc['Minimo'][nameCol] = float(df[col].min())

    med.loc['Maximo'][nameCol] = float(df[col].max())

    return med
# Verificando a situação cadstral das empresas - quais são os valores disponíveis

df_CNPJ['situacao_cadastral'].value_counts()
df_cnpjs = df_CNPJ.copy()



df_cnpjs = df_cnpjs[(df_cnpjs["situacao_cadastral"].isin(["08","02"]))]

print(df_cnpjs.shape)

df_cnpjs.head(1)
# Descrição do Porte da Empresa

df_cnpjs['porte_empresa_descr'] = df_cnpjs['porte_empresa'].replace(['00','01','03','05'],['NAO INFORMADO','MICRO EMPRESA','EMPRESA DE PEQUENO PORTE','DEMAIS'])

df_cnpjs['situacao_atividade']= df_cnpjs['situacao_cadastral'].replace(['08','02'],['INATIVA','ATIVA'])

df_cnpjs.head(1)
# Join para incluir descrição do CNAE e o a região

df_cnpjs = pd.merge(df_cnpjs, df_desc_cnae,how='left', left_on=['cnae_fiscal'], right_on=['cod_cnae'], )

df_cnpjs = pd.merge(df_cnpjs, df_uf_regiao,how='left', on=['uf'])

df_cnpjs.head(1)
# Verificar os tipos de cada coluna

df_cnpjs.info()
# Transformando as colunas em datetime

df_cnpjs['data_inicio_atividade'] = pd.to_datetime(df_cnpjs['data_inicio_atividade'], errors = 'coerce')

df_cnpjs['data_situacao_cadastral'] = pd.to_datetime(df_cnpjs['data_situacao_cadastral'], errors = 'coerce')
df_cnpjs.info()
# Verifica a distribuição por categorias de atividade na base total

tabela_categoricas(df_cnpjs,'nm_secao','SEÇÃO')
# Verifica a distribuição por status das empresas na base total

tabela_categoricas(df_cnpjs,'situacao_atividade', 'Atividade')
# Verifica a distribuição do tamanho das empresas na base total

tabela_categoricas(df_cnpjs,'porte_empresa_descr','Porte')
# Verificação da última data disponível no dataset

max(df_cnpjs['data_situacao_cadastral'])
# Novo dataset somente da empresas de 2008 no período da crise econômica 

df_cnpjs_filtrado_2008 = df_cnpjs[(df_cnpjs['data_situacao_cadastral'] >= "2008-09-15") & 

                            (df_cnpjs['data_situacao_cadastral'] <= "2008-12-31")]

df_cnpjs_filtrado_2008.shape
# Criando algumas variáveis que podem ser úteis no decorrer do estudo

menor_data_situacao_2008 = min(df_cnpjs_filtrado_2008['data_situacao_cadastral'])

maior_data_situacao_2008 = max(df_cnpjs_filtrado_2008['data_situacao_cadastral'])

menor_data_inicio_2008 = min(df_cnpjs_filtrado_2008['data_inicio_atividade'])

maior_data_inicio_2008 = max(df_cnpjs_filtrado_2008['data_inicio_atividade'])

print (menor_data_situacao_2008, maior_data_situacao_2008, menor_data_inicio_2008, maior_data_inicio_2008)
# Novo dataset somente da empresas de 2020 no período da crise enquanto estamos na pandemia 

df_cnpjs_filtrado_2020 = df_cnpjs[(df_cnpjs['data_situacao_cadastral'] >= "2020-02-26") & 

                            (df_cnpjs['data_situacao_cadastral'] <= "2020-07-03")]

df_cnpjs_filtrado_2020.shape
menor_data_situacao_2020 = min(df_cnpjs_filtrado_2020['data_situacao_cadastral'])

maior_data_situacao_2020 = max(df_cnpjs_filtrado_2020['data_situacao_cadastral'])

menor_data_inicio_2020 = min(df_cnpjs_filtrado_2020['data_inicio_atividade'])

maior_data_inicio_2020 = max(df_cnpjs_filtrado_2020['data_inicio_atividade'])

print (menor_data_situacao_2020, maior_data_situacao_2020, menor_data_inicio_2020, maior_data_inicio_2020)
# Quantidade de registros de empresas Ativos e Inativas

tabela_categoricas(df_cnpjs_filtrado_2008,'situacao_atividade', 'Situação Empresa')
df_cnpjs_filtrado_2008['situacao_atividade'].hist(grid = False, bins = 'auto',figsize=(7,7));
# Tipo de empresas durante o periodo em 2008

tabela_categoricas(df_cnpjs_filtrado_2008,'nm_secao','SEÇÃO')
# Tipo de empresas durante o periodo em 2008

tabela_categoricas(df_cnpjs_filtrado_2008,'cod_cnae','NOME_CNAE')
# Tipo de empresas durante o periodo em 2008

tabela_categoricas(df_cnpjs_filtrado_2008,'nm_cnae','NOME_CNAE')
# Separação do dataset com CNPJs Ativos e Inativos em 2008

cnpjAtivo2008 = df_cnpjs_filtrado_2008[df_cnpjs_filtrado_2008['situacao_cadastral'] == "02"].copy()

cnpjInativo2008 = df_cnpjs_filtrado_2008[df_cnpjs_filtrado_2008['situacao_cadastral'] == "08"].copy()
# Mostra o dado tabular das áreas de atuação das empresas que fecharam em 2008

tabela_categoricas(cnpjAtivo2008,'nm_secao','SEÇÃO')
# Mostra o dado tabular das áreas de atuação das empresas que fecharam em 2008

tabela_categoricas(cnpjAtivo2008,'nm_cnae','SEÇÃO')
# Mostra o dado tabular das áreas de atuação das empresas que fecharam em 2008

tabela_categoricas(cnpjInativo2008,'nm_secao','SEÇÃO')
# Mostra o dado tabular das áreas de atuação das empresas que fecharam em 2008

tabela_categoricas(cnpjInativo2008,'nm_cnae','SEÇÃO')
# Mostra o dado tabular das áreas de atuação das empresas que fecharam em 2008

tabela_categoricas(cnpjInativo2008,'cod_cnae','SEÇÃO')
# Criação das colunas relacionadas a data de abertura no dataset de empresas ativas.

cnpjAtivo2008['diaAbertura'] = cnpjAtivo2008['data_inicio_atividade'].dt.day

cnpjAtivo2008['mesAbertura'] = cnpjAtivo2008['data_inicio_atividade'].dt.month

cnpjAtivo2008['anoAbertura'] = cnpjAtivo2008['data_inicio_atividade'].dt.year
# Criação das colunas relacionadas a data de abertura no dataset de empresas inativas.

cnpjInativo2008['diaAbertura'] = cnpjInativo2008['data_inicio_atividade'].dt.day

cnpjInativo2008['mesAbertura'] = cnpjInativo2008['data_inicio_atividade'].dt.month

cnpjInativo2008['anoAbertura'] = cnpjInativo2008['data_inicio_atividade'].dt.year

cnpjInativo2008['diaFechamento'] = cnpjInativo2008['data_situacao_cadastral'].dt.day

cnpjInativo2008['mesFechamento'] = cnpjInativo2008['data_situacao_cadastral'].dt.month

cnpjInativo2008['anoFechamento'] = cnpjInativo2008['data_situacao_cadastral'].dt.year
# Criação das colunas relacionadas a data de abertura no dataset geral de 2008.

df_cnpjs_filtrado_2008['diaAbertura'] = df_cnpjs_filtrado_2008['data_inicio_atividade'].dt.day

df_cnpjs_filtrado_2008['mesAbertura'] = df_cnpjs_filtrado_2008['data_inicio_atividade'].dt.month

df_cnpjs_filtrado_2008['anoAbertura'] = df_cnpjs_filtrado_2008['data_inicio_atividade'].dt.year
# Confirmando as colunas para ver se foram criadas conforme esperado

cnpjAtivo2008.head(1)
# Confirmando as colunas para ver se foram criadas conforme esperado

cnpjInativo2008.head(1)
df_cnpjs_filtrado_2008.head(1)
print (menor_data_situacao_2008, maior_data_situacao_2008, menor_data_inicio_2008, maior_data_inicio_2008)
# Criação coluna idade para as empresas ativas

cnpjAtivo2008['idade'] = pd.to_datetime("2008-12-31") - cnpjAtivo2008['data_inicio_atividade']

print(cnpjAtivo2008.head())
# Transformação da coluna idade

cnpjAtivo2008['idade'] = pd.to_numeric(cnpjAtivo2008['idade'].dt.days, downcast = 'integer') / 365

print(cnpjAtivo2008.head())
distribuicaoNumericas(cnpjAtivo2008, 'idade' , 'Idade das Empresas Ativas')
# Criação coluna idade para as empresas ativas

cnpjInativo2008['idade'] = cnpjInativo2008['data_situacao_cadastral'] - cnpjInativo2008['data_inicio_atividade']

print(cnpjInativo2008.head())
# Transformação da coluna idade

cnpjInativo2008['idade'] = pd.to_numeric(cnpjInativo2008['idade'].dt.days, downcast = 'integer') / 365

print(cnpjInativo2008.head())
distribuicaoNumericas(cnpjInativo2008, 'idade' , 'Idade das Empresas Inativas')
# Criando um novo dataset com as empresas que abriram e fecharam no período da crise de 2008

cnpjInativo2008_AbriuFechou = cnpjInativo2008.query('data_inicio_atividade >= "2008-09-15" & data_situacao_cadastral <= "2008-12-31"')

cnpjInativo2008_AbriuFechou.head(1)
print ("O dataset possui ", cnpjInativo2008_AbriuFechou.shape[0], "linhas e ", cnpjInativo2008_AbriuFechou.shape[1], "colunas.")
# Criando um novo dataset com as empresas que abriram no período da crise de 2008

cnpjAtivo2008_Abriu = cnpjAtivo2008.query('data_inicio_atividade >= "2008-09-15" & data_situacao_cadastral <= "2008-12-31"')

cnpjAtivo2008_Abriu.head(1)
print ("O dataset possui ", cnpjAtivo2008_Abriu.shape[0], "linhas e ", cnpjAtivo2008_Abriu.shape[1], "colunas.")
# Criando um novo dataset com as empresas que fecharam no período da crise de 2008

cnpjInativo2008_Fechou = cnpjInativo2008.query('data_situacao_cadastral <= "2008-12-31"')

cnpjInativo2008_Fechou.head(1)
print ("O dataset possui ", cnpjInativo2008_Fechou.shape[0], "linhas e ", cnpjInativo2008_Fechou.shape[1], "colunas.")
mes_abertura08 = pd.DataFrame(cnpjAtivo2008_Abriu['mesAbertura'].value_counts()).reset_index()

mes_abertura08.columns = ['Mes Abertura' , 'Qtd. Empresas 2008']

mes_abertura08 = mes_abertura08.sort_values(by='Mes Abertura')
cnpjAtivo2008_Abriu['mesAbertura'].value_counts()
ax = mes_abertura08.plot(kind = "bar", x = 'Mes Abertura',

                     title = " Quantidade de Empresas Abertas",figsize = (10,4));

ax.set_xlabel("Mes Abertura");
mes_fechamento08 = pd.DataFrame(cnpjInativo2008_Fechou['mesFechamento'].value_counts()).reset_index()

mes_fechamento08.columns = ['Mes Fechamento' , 'Qtd. Empresas 2008']

mes_fechamento08 = mes_fechamento08.sort_values(by='Mes Fechamento')
cnpjInativo2008_Fechou['mesFechamento'].value_counts()
ax = mes_fechamento08.plot(kind = "bar", x = 'Mes Fechamento',

                     title = " Quantidade de Empresas Fechadas",figsize = (10,4));

ax.set_xlabel("Mes Fechamento");
mes_AbrFec08 = pd.DataFrame(cnpjInativo2008_AbriuFechou['mesFechamento'].value_counts()).reset_index()

mes_AbrFec08.columns = ['Mes Fechamento' , 'Qtd. Empresas']

mes_AbrFec08 = mes_AbrFec08.sort_values(by='Mes Fechamento')
cnpjInativo2008_AbriuFechou['mesFechamento'].value_counts()
ax = mes_AbrFec08.plot(kind = "bar", x = 'Mes Fechamento',

                     title = " Quantidade de Empresas Fechadas",figsize = (10,4));

ax.set_xlabel("Mes Fechamento");
# Mostra o porte das empresas que fecharam em 2008

tabela_categoricas(cnpjInativo2008,'porte_empresa_descr','Porte')
# Quantidade de registros de empresas Ativos e Inativas

tabela_categoricas(df_cnpjs_filtrado_2020,'situacao_atividade', 'Situação Empresa')
df_cnpjs_filtrado_2020['situacao_atividade'].hist(grid = False, bins = 'auto',figsize=(7,7));
# Tipo de empresas durante o periodo em 2020

tabela_categoricas(df_cnpjs_filtrado_2020,'nm_secao','SEÇÃO')
# Tipo de empresas durante o periodo em 2020

tabela_categoricas(df_cnpjs_filtrado_2020,'cod_cnae','SEÇÃO')
# Tipo de empresas durante o periodo em 2020

tabela_categoricas(df_cnpjs_filtrado_2020,'nm_cnae','NOME_CNAE')
# Separação do dataset com CNPJs Ativos e Inativos em 2020

cnpjAtivo2020 = df_cnpjs_filtrado_2020[df_cnpjs_filtrado_2020['situacao_cadastral'] == "02"].copy()

cnpjInativo2020 = df_cnpjs_filtrado_2020[df_cnpjs_filtrado_2020['situacao_cadastral'] == "08"].copy()
# Mostra o dado tabular das áreas de atuação das empresas que permaneceram ativas em 2020

tabela_categoricas(cnpjAtivo2020,'nm_secao','SEÇÃO')
# Mostra o dado tabular das áreas de atuação das empresas que permaneceram ativas em 2020

tabela_categoricas(cnpjAtivo2020,'nm_cnae','SEÇÃO')
# Mostra o dado tabular das áreas de atuação das empresas que fecharam em 2020

tabela_categoricas(cnpjInativo2020,'nm_secao','SEÇÃO')
# Mostra o dado tabular das áreas de atuação das empresas que fecharam em 2020

tabela_categoricas(cnpjInativo2020,'nm_cnae','SEÇÃO')
# Criação das colunas relacionadas a data de abertura no dataset de empresas ativas.

cnpjAtivo2020['diaAbertura'] = cnpjAtivo2020['data_inicio_atividade'].dt.day

cnpjAtivo2020['mesAbertura'] = cnpjAtivo2020['data_inicio_atividade'].dt.month

cnpjAtivo2020['anoAbertura'] = cnpjAtivo2020['data_inicio_atividade'].dt.year
# Criação das colunas relacionadas a data de abertura no dataset de empresas inativas.

cnpjInativo2020['diaAbertura'] = cnpjInativo2020['data_inicio_atividade'].dt.day

cnpjInativo2020['mesAbertura'] = cnpjInativo2020['data_inicio_atividade'].dt.month

cnpjInativo2020['anoAbertura'] = cnpjInativo2020['data_inicio_atividade'].dt.year

cnpjInativo2020['diaFechamento'] = cnpjInativo2020['data_situacao_cadastral'].dt.day

cnpjInativo2020['mesFechamento'] = cnpjInativo2020['data_situacao_cadastral'].dt.month

cnpjInativo2020['anoFechamento'] = cnpjInativo2020['data_situacao_cadastral'].dt.year
# Criação das colunas relacionadas a data de abertura no dataset geral de 2008.

df_cnpjs_filtrado_2020['diaAbertura'] = df_cnpjs_filtrado_2020['data_inicio_atividade'].dt.day

df_cnpjs_filtrado_2020['mesAbertura'] = df_cnpjs_filtrado_2020['data_inicio_atividade'].dt.month

df_cnpjs_filtrado_2020['anoAbertura'] = df_cnpjs_filtrado_2020['data_inicio_atividade'].dt.year
# Confirmando as colunas para ver se foram criadas conforme esperado

cnpjAtivo2020.head(1)
# Confirmando as colunas para ver se foram criadas conforme esperado

cnpjInativo2020.head(1)
df_cnpjs_filtrado_2020.head(1)
print (menor_data_situacao_2008, maior_data_situacao_2008, menor_data_inicio_2008, maior_data_inicio_2008)
# Criação coluna idade para as empresas ativas

cnpjAtivo2020['idade'] = pd.to_datetime("2020-07-03") - cnpjAtivo2020['data_inicio_atividade']

print(cnpjAtivo2020.head())
# Transformação da coluna idade

cnpjAtivo2020['idade'] = pd.to_numeric(cnpjAtivo2020['idade'].dt.days, downcast = 'integer') / 365

print(cnpjAtivo2020.head())
distribuicaoNumericas(cnpjAtivo2020, 'idade' , 'Idade das Empresas Ativas')
# Criação coluna idade para as empresas ativas

cnpjInativo2020['idade'] = cnpjInativo2020['data_situacao_cadastral'] - cnpjInativo2020['data_inicio_atividade']

print(cnpjInativo2020.head())
# Transformação da coluna idade

cnpjInativo2020['idade'] = pd.to_numeric(cnpjInativo2020['idade'].dt.days, downcast = 'integer') / 365

print(cnpjInativo2020.head())
distribuicaoNumericas(cnpjInativo2020, 'idade' , 'Idade das Empresas Inativas')
# Criando um novo dataset com as empresas que abriram e fecharam no período da crise de 2020

cnpjInativo2020_AbriuFechou = cnpjInativo2020.query('data_inicio_atividade >= "2020-01-01" & data_situacao_cadastral <= "2020-07-03"')

cnpjInativo2020_AbriuFechou.head(1)
print ("O dataset possui ", cnpjInativo2020_AbriuFechou.shape[0], "linhas e ", cnpjInativo2020_AbriuFechou.shape[1], "colunas.")
# Criando um novo dataset com as empresas que abriram no período da crise de 2020

cnpjAtivo2020_Abriu = cnpjAtivo2020.query('data_inicio_atividade >= "2020-01-01" & data_situacao_cadastral <= "2020-07-03"')

cnpjAtivo2020_Abriu.head(1)
print ("O dataset possui ", cnpjAtivo2020_Abriu.shape[0], "linhas e ", cnpjAtivo2020_Abriu.shape[1], "colunas.")
# Criando um novo dataset com as empresas que fecharam no período da crise de 2020

cnpjInativo2020_Fechou = cnpjInativo2020.query('data_situacao_cadastral <= "2020-07-03"')

cnpjInativo2020_Fechou.head(1)
print ("O dataset possui ", cnpjInativo2020_Fechou.shape[0], "linhas e ", cnpjInativo2020_Fechou.shape[1], "colunas.")
mes_abertura20 = pd.DataFrame(cnpjAtivo2020_Abriu['mesAbertura'].value_counts()).reset_index()

mes_abertura20.columns = ['Mes Abertura' , 'Qtd. Empresas 2020']

mes_abertura20 = mes_abertura20.sort_values(by='Mes Abertura')
cnpjAtivo2020_Abriu['mesAbertura'].value_counts()
ax = mes_abertura20.plot(kind = "bar", x = 'Mes Abertura',

                     title = " Quantidade de Empresas Abertas",figsize = (10,4));

ax.set_xlabel("Mes Abertura");
mes_fechamento20 = pd.DataFrame(cnpjInativo2020_Fechou['mesFechamento'].value_counts()).reset_index()

mes_fechamento20.columns = ['Mes Fechamento' , 'Qtd. Empresas 2020']

mes_fechamento20 = mes_fechamento20.sort_values(by='Mes Fechamento')
cnpjInativo2020_Fechou['mesFechamento'].value_counts()
ax = mes_fechamento20.plot(kind = "bar", x = 'Mes Fechamento',

                     title = " Quantidade de Empresas Fechadas",figsize = (10,4));

ax.set_xlabel("Mes Fechamento");
mes_AbrFec20 = pd.DataFrame(cnpjInativo2020_AbriuFechou['mesFechamento'].value_counts()).reset_index()

mes_AbrFec20.columns = ['Mes Fechamento' , 'Qtd. Empresas 2020']

mes_AbrFec20 = mes_AbrFec20.sort_values(by='Mes Fechamento')
cnpjInativo2020_AbriuFechou['mesFechamento'].value_counts()
ax = mes_AbrFec20.plot(kind = "bar", x = 'Mes Fechamento',

                     title = " Quantidade de Empresas Fechadas",figsize = (10,4));

ax.set_xlabel("Mes Fechamento");
# Mostra o porte das empresas que fecharam em 2008

tabela_categoricas(cnpjInativo2020,'porte_empresa_descr','Porte')
cnpjAtivo2008.query('anoAbertura == 2008').groupby(['nm_cnae']).agg({'nm_cnae':'count'}).nlargest(n=5,columns='nm_cnae')
cnpjInativo2008.query('anoFechamento == 2008').groupby(['nm_cnae']).agg({'nm_cnae':'count'}).nlargest(n=5,columns='nm_cnae')
cnpjInativo2008_AbriuFechou.query('anoFechamento == 2008').groupby(['nm_cnae']).agg({'nm_cnae':'count'}).nlargest(n=5,columns='nm_cnae')
ax = mes_abertura08.plot(kind = "bar", x = 'Mes Abertura',

                     title = " Quantidade de Empresas Abertas",figsize = (10,4));

ax = mes_fechamento08.plot(kind = "bar", x = 'Mes Fechamento',

                     title = " Quantidade de Empresas Fechadas",figsize = (10,4));

ax.set_xlabel("Gráfico quantidade - Abertura x Fechamento - 2008");



# utilização do ;  é para não aparecer o título gerado automaticamente pelos gráficos.
cnpjInativo2008['idade'].hist(grid = False, bins = 10);
cnpjAtivo2020.query('anoAbertura == 2020').groupby(['nm_cnae']).agg({'nm_cnae':'count'}).nlargest(n=5,columns='nm_cnae')
cnpjInativo2020.query('anoFechamento == 2020').groupby(['nm_cnae']).agg({'nm_cnae':'count'}).nlargest(n=5,columns='nm_cnae')
cnpjInativo2020_AbriuFechou.query('anoFechamento == 2020').groupby(['nm_cnae']).agg({'nm_cnae':'count'}).nlargest(n=5,columns='nm_cnae')
ax = mes_abertura20.plot(kind = "bar", x = 'Mes Abertura',

                     title = " Quantidade de Empresas Abertas",figsize = (10,4));

ax = mes_fechamento20.plot(kind = "bar", x = 'Mes Fechamento',

                     title = " Quantidade de Empresas Fechadas",figsize = (10,4));

ax.set_xlabel("Gráfico quantidade - Abertura x Fechamento - 2020");



# utilização do ;  é para não aparecer o título gerado automaticamente pelos gráficos.
cnpjInativo2020['idade'].hist(grid = False, bins = 10);
plt.figure(figsize=(12,8))

plt.hist(cnpjAtivo2008['idade'],alpha=0.6,label='Ativas 2008')

plt.hist(cnpjAtivo2020['idade'],alpha=0.6,label='Ativas 2020')

plt.legend()

plt.show()
cnpjAtivo2008.query('anoAbertura == 2008').groupby(['nm_cnae']).agg({'nm_cnae':'count'}).nlargest(n=5,columns='nm_cnae')
cnpjAtivo2020.query('anoAbertura == 2020').groupby(['nm_cnae']).agg({'nm_cnae':'count'}).nlargest(n=5,columns='nm_cnae')
plt.figure(figsize=(12,8))

plt.hist(cnpjInativo2008['idade'],alpha=0.6,label='Inativas 2008')

plt.hist(cnpjInativo2020['idade'],alpha=0.6,label='Inativas 2020')

plt.legend()

plt.show()
cnpjInativo2008.query('anoFechamento == 2008').groupby(['nm_cnae']).agg({'nm_cnae':'count'}).nlargest(n=5,columns='nm_cnae')
cnpjInativo2020.query('anoFechamento == 2020').groupby(['nm_cnae']).agg({'nm_cnae':'count'}).nlargest(n=5,columns='nm_cnae')
plt.figure(figsize=(12,12))

plt.hist(cnpjInativo2008_AbriuFechou['idade'],alpha=0.6,label='Inativas 2008')

plt.hist(cnpjInativo2020_AbriuFechou['idade'],alpha=0.6,label='Inativas 2020')

plt.legend()

plt.show()
# Empresas que abriram e fecharam no período da crise de 2008

cnpjInativo2008_AbriuFechou.query('anoFechamento == 2008').groupby(['nm_cnae']).agg({'nm_cnae':'count'}).nlargest(n=5,columns='nm_cnae')
# Empresas que abriram e fecharam no período da crise de 2020

cnpjInativo2020_AbriuFechou.query('anoFechamento == 2020').groupby(['nm_cnae']).agg({'nm_cnae':'count'}).nlargest(n=5,columns='nm_cnae')
cnpjInativo2008_semPolitica = cnpjInativo2008.query('data_situacao_cadastral <= "2008-12-31" & anoFechamento == 2008 & cod_cnae != "9492800"')

cnpjInativo2008_semPolitica
cnpjInativo2008_semPolitica = cnpjInativo2008.query('anoFechamento == 2008 & cod_cnae != "9492800"')

cnpjInativo2008_semPoliticaCnae = cnpjInativo2008_semPolitica.groupby(['nm_cnae']).agg({'nm_cnae':'count'}).nlargest(n=5,columns='nm_cnae')

cnpjInativo2008_semPoliticaCnae
cnpjInativo2008_semPolitica
cnpjInativo2020.query('anoFechamento == 2020').groupby(['nm_cnae']).agg({'nm_cnae':'count'}).nlargest(n=5,columns='nm_cnae')
mes_fechamento08_semPolitica = pd.DataFrame(cnpjInativo2008_semPolitica['mesFechamento'].value_counts()).reset_index()

mes_fechamento08_semPolitica.columns = ['Mes Fechamento' , 'Qtd. Empresas 2008']

mes_fechamento08_semPolitica = mes_fechamento08_semPolitica.sort_values(by='Mes Fechamento')
mes_fechamento08_semPolitica
ax = mes_fechamento08_semPolitica.plot(kind = "bar", x = 'Mes Fechamento',

                     title = " Quantidade de Empresas Fechadas - Sem empresas políticas",figsize = (10,4));

ax = mes_fechamento20.plot(kind = "bar", x = 'Mes Fechamento',

                     title = " Quantidade de Empresas Fechadas",figsize = (10,4));

ax.set_xlabel("Gráfico quantidade - Fechamento - 2008 x 2020");



# utilização do ;  é para não aparecer o título gerado automaticamente pelos gráficos.