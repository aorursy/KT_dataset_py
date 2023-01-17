# Caso o Anaconda não esteja instalado, faz a instalação
# !conda install --yes pandas
# Desenvolvido por Tiago Baroni 
# Modificado por Marcilon Cunha

import sqlite3 # Declara que vamos utilizar o SQlite para ler dados
import pandas as pd
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Desenvolvido por Tiago Baroni 
# Modificado por Marcilon Cunha
# Filtrando somente os registrados dos últimos 10 anos, possíbilitando útilizar mais dados da base 100%

# Devido a impossibilidade de carragamento da base de dados 100%, que acarreta em estouro de memória, 
  # foi necessário efetuar a restrição dos dados tendo sido utilizado a base de dados de 5 % contendo 
  # pouco mais de 2.2 milhoes de registros para desenvolvimento ágio das análises, e, ao final, validados 
  # utilizando a base de dados de 100%, porem aplicando o filtro maximo de 12 milhoes de registros. 
  # Ao tentar 13 milhoes de registros ou mais já causa o estouro de memória no kaggle e também no colab.
  # A máquina local utilizada tambem não consegue processar a base de dados de 100%.
    
#conexao = sqlite3.connect('/kaggle/input/amostra5pc/amostra5pc.sqlite')
conexao = sqlite3.connect('/kaggle/input/amostracnpj/amostra100pc.sqlite')

chunksize=12000000 # definindo a quantidade máxima de registros suportado para mémoria disponivel no kaggle.

# Carrega os dados do CNPJ
df_CNPJ = pd.read_sql_query('''SELECT razao_social
                                     ,situacao_cadastral
                                     ,cnpj
                                     ,uf
                                     ,capital_social_empresa
                                     ,porte_empresa
                                     ,cnae_fiscal
                                     ,data_inicio_atividade
                                     ,data_situacao_cadastral 
                                FROM cnpj_dados_cadastrais_pj 
                               WHERE data_inicio_atividade 
                                     BETWEEN '2010-01-01' 
                                         AND '2020-09-31' ''', conexao, chunksize=chunksize).__next__()


# Desenvolvido por Tiago Baroni 
print(df_CNPJ.shape)
# Desenvolvido por Tiago Baroni 

df_desc_cnae = pd.read_sql_query("SELECT cod_secao, nm_secao, cod_cnae FROM tab_cnae", conexao)
df_desc_cnae.head()
# Desenvolvido por Tiago Baroni 
# Modificado por Marcilon Cunha

df_uf_regiao = pd.read_csv('/kaggle/input/uf-regiao/uf_regiao.csv')
df_uf_regiao.head()
# Desenvolvido por Tiago Baroni 

def tabela_categoricas(df,col,name_col):
    df_1 = pd.DataFrame(df[col].value_counts()).reset_index()
    df_1.columns = [name_col,'Qtd empresas']
    df_1['%Total'] = (df_1['Qtd empresas']/len(df))*100
    return df_1
# Desenvolvido por Tiago Baroni

def distribuicaoNumericas(df,col,nameCol):
    med = pd.DataFrame(columns=[nameCol], index=['Media', 'Mediana', 'Minimo','Maximo'])
    med.loc['Media'][nameCol] = float(df[col].mean())
    med.loc['Mediana'][nameCol] = float(df[col].median())
    med.loc['Minimo'][nameCol] = float(df[col].min())
    med.loc['Maximo'][nameCol] = float(df[col].max())
    return med
# Desenvolvido por Tiago Baroni 

df_cnpjs = df_CNPJ.copy()

df_cnpjs = df_cnpjs[(df_cnpjs["situacao_cadastral"].isin(["08","02"]))]
print(df_cnpjs.shape)
df_cnpjs.head()
# Desenvolvido por Tiago Baroni 

df_cnpjs['porte_empresa_descr'] = df_cnpjs['porte_empresa'].replace(['00','01','03','05'],['NAO INFORMADO','MICRO EMPRESA','EMPRESA DE PEQUENO PORTE','DEMAIS'])
df_cnpjs['situacao_atividade'] = df_cnpjs['situacao_cadastral'].replace(['08','02'],['INATIVA','ATIVA'])
df_cnpjs.head()
# Desenvolvido por Tiago Baroni 
df_cnpjs['cnae_fiscal']

# Desenvolvido por Tiago Baroni 

df_cnpjs = pd.merge(df_cnpjs, df_desc_cnae,how='left', left_on=['cnae_fiscal'], right_on=['cod_cnae'])
df_cnpjs = pd.merge(df_cnpjs, df_uf_regiao,how='left', on=['uf'])
df_cnpjs.head()
# Desenvolvido por Tiago Baroni 
tabela_categoricas(df_cnpjs,'situacao_atividade', 'Atividade')
# Desenvolvido por Tiago Baroni 
df_cnpjs['data_inicio_atividade'] = pd.to_datetime(df_cnpjs['data_inicio_atividade'])
df_cnpjs['data_situacao_cadastral'] = pd.to_datetime(df_cnpjs['data_situacao_cadastral'])

#print(df_cnpjs.dtypes)
# Desenvolvido por Tiago Baroni 

df_cnpjs_filtrado = df_cnpjs[(df_cnpjs['data_situacao_cadastral'] >= "2010-01-01") & 
                            (df_cnpjs['situacao_cadastral'] < "08")]
df_cnpjs_filtrado.shape
# Desenvolvido por Tiago Baroni 

tabela_categoricas(df_cnpjs,'nm_secao','SEÇÃO')
# Desenvolvido por Tiago Baroni 

tabela_categoricas(df_cnpjs,'porte_empresa_descr','Porte')
# Desenvolvido por Tiago Baroni 

tabela_categoricas(df_cnpjs_filtrado,'porte_empresa_descr','Porte')
# Desenvolvido por Tiago Baroni 

tabela_categoricas(df_cnpjs_filtrado,'nm_secao','SEÇÃO')
# Desenvolvido por Tiago Baroni 
df_cnpjs['capital_social_empresa'].hist(grid = False)
# Desenvolvido por Tiago Baroni 
distribuicaoNumericas(df_cnpjs,'capital_social_empresa','Capital Social - Empresas')
# Desenvolvido por Tiago Baroni 

df_cnpjs[df_cnpjs['capital_social_empresa']==0].shape[0]
# Desenvolvido por Tiago Baroni 
df_cnpjs[df_cnpjs['capital_social_empresa'] > 10000000000]
# Desenvolvido por Tiago Baroni 

cnpjAtivos = df_cnpjs[df_cnpjs['situacao_cadastral'] == "02"].copy()
cnpjInativo = df_cnpjs[df_cnpjs['situacao_cadastral'] == "08"].copy()
# Desenvolvido por Tiago Baroni 
cnpjAtivos['idade'] = pd.to_datetime("2010-01-01") - cnpjAtivos['data_inicio_atividade']
print(cnpjAtivos.head())
# Desenvolvido por Tiago Baroni 
cnpjAtivos['idade'] = pd.to_numeric(cnpjAtivos['idade'].dt.days, downcast = 'integer') / 365
print(cnpjAtivos.head())
# Desenvolvido por Tiago Baroni 
distribuicaoNumericas(cnpjAtivos, 'idade' , 'Idade das Empresas Ativas')
# Desenvolvido por Tiago Baroni 
cnpjInativo['idade'] = cnpjInativo['data_situacao_cadastral'] - cnpjInativo['data_inicio_atividade']
cnpjInativo['idade'] = pd.to_numeric(cnpjInativo['idade'].dt.days, downcast = 'integer') / 365
distribuicaoNumericas(cnpjInativo, 'idade' , 'Idade das Empresas Inativas')
# Desenvolvido por Tiago Baroni 
cnpjAtivos['idade'].hist(grid = False, bins = 10)
# Desenvolvido por Tiago Baroni 
cnpjInativo['idade'].hist(grid = False, bins = 10)
# Desenvolvido por Tiago Baroni 
cnpjAtivos['anoAbertura'] = cnpjAtivos['data_inicio_atividade'].dt.year
cnpjInativo['anoAbertura'] = cnpjInativo['data_inicio_atividade'].dt.year
cnpjInativo['anoFechamento'] = cnpjInativo['data_situacao_cadastral'].dt.year
df_cnpjs['anoAbertura'] = df_cnpjs['data_inicio_atividade'].dt.year
# Desenvolvido por Tiago Baroni 
# Modificado por Marcilon
abert_anos = pd.DataFrame(df_cnpjs['anoAbertura'].value_counts()).reset_index()
abert_anos = abert_anos[abert_anos['index'] > 2010]
abert_anos.columns = ['Ano Abertura' , 'Qtd. Empresas']
abert_anos['Ano Abertura'] = abert_anos['Ano Abertura'].apply(str)
abert_anos = abert_anos.sort_values(by='Ano Abertura')
# Desenvolvido por Tiago Baroni 
ax = abert_anos.plot(kind = "bar", x = 'Ano Abertura',
                     title = " Quantidade de Empresas Abertas",figsize = (10,4))
ax.set_xlabel("Ano Abertura")
# Desenvolvido por Tiago Baroni 
# Modificado por Marcilon
df_cnpjs_ult_ano_df = df_cnpjs[(df_cnpjs['anoAbertura'] > 2010) & (df_cnpjs['uf'] == 'DF')]
df_cnpjs_ult_ano_df = tabela_categoricas(df_cnpjs_ult_ano_df, 'nm_secao', 'SEÇÃO')

df_cnpjs_ult_ano_df.head(20)
df_cnpjs_ult_ano_df_porte01e03 = df_cnpjs[(df_cnpjs['anoAbertura'] > 2010) & (df_cnpjs['uf'] == 'DF') & df_cnpjs["porte_empresa"].isin(["01","03"])]
df_cnpjs_ult_ano_df_porte01e03 = tabela_categoricas(df_cnpjs_ult_ano_df_porte01e03, 'nm_secao', 'SEÇÃO')

df_cnpjs_ult_ano_df_porte01e03.head(20)
# Desenvolvido por por Marcilon
df_cnpjs_ult_ano = df_cnpjs[(df_cnpjs['anoAbertura'] > 2010)]

df_abertas_10anos = tabela_categoricas(df_cnpjs_ult_ano, 'nm_secao', 'SEÇÃO')
df_abertas_10anos.head(20)

# Desenvolvido por Tiago Baroni 
# Modificado por Marcilon
fech_anos = pd.DataFrame(cnpjInativo['anoFechamento'].value_counts()).reset_index()
fech_anos = fech_anos[fech_anos['index'] > 2010]
fech_anos.columns = ['Ano Fechamento','Qtd Empresas']
fech_anos['Ano Fechamento'] = fech_anos['Ano Fechamento'].apply(str)
fech_anos = fech_anos.sort_values(by='Ano Fechamento')

# Desenvolvido por Tiago Baroni 
ax = fech_anos.plot(kind = "bar", x = 'Ano Fechamento', title = "Quantidade de Empresas Fechadas", figsize = (10,4))
ax.set_xlabel("Ano Fechamento")
# Desenvolvido por Tiago Baroni 
# Modificado por Marcilon
df_cnpjs_ult_ano = cnpjInativo[(cnpjInativo['anoFechamento'] > 2010)]
tabela_categoricas(df_cnpjs_ult_ano, 'nm_secao', 'SEÇÃO')
# Desenvolvido por Tiago Baroni 
uf_fechamento = pd.DataFrame(cnpjInativo['uf'].value_counts()).reset_index()
uf_fechamento.columns = ['Estados', 'Quantidade de Empresas Fechadas']
uf_fechamento = uf_fechamento.sort_values(by='Estados')
# Desenvolvido por Tiago Baroni 
ax = uf_fechamento.plot(kind = "bar", x = 'Estados',
                    title = "Quantidade de Empresas Fechadas por UF",figsize = (10,4))
ax.set_xlabel("UF")
# Desenvolvido por Marcilon
abert_10_ult_anos = pd.DataFrame(cnpjAtivos['anoAbertura'].value_counts()).reset_index()
abert_10_ult_anos = abert_10_ult_anos[abert_10_ult_anos['index'] > 2010]
abert_10_ult_anos.columns = ['Ano Abertura' , 'Qtd. Empresas']
abert_10_ult_anos['Ano Abertura'] = abert_10_ult_anos['Ano Abertura'].apply(str)
abert_10_ult_anos = abert_10_ult_anos.sort_values(by='Ano Abertura')
ax = abert_10_ult_anos.plot(kind = "bar", x = 'Ano Abertura',
                     title = " Quantidade de Empresas ATIVAS e Abertas nos últimos 10 anos",figsize = (10,4))
ax.set_xlabel("Ano Abertura")

Ativa_10_ult_anos_DF = cnpjAtivos[(cnpjAtivos['anoAbertura'] > 2010) & (cnpjAtivos['uf'] == 'DF')]
Ativa_10_ult_anos_DF = tabela_categoricas(Ativa_10_ult_anos_DF, 'nm_secao', 'SEÇÃO')

Ativa_10_ult_anos_DF.head(20)
df_cnpjs_ult_ativa_df_porte01e03 = cnpjAtivos[(cnpjAtivos['anoAbertura'] > 2010) & (cnpjAtivos['uf'] == 'DF') & cnpjAtivos["porte_empresa"].isin(["01","03"])]
df_cnpjs_ult_ativa_df_porte01e03 = tabela_categoricas(df_cnpjs_ult_ativa_df_porte01e03, 'nm_secao', 'SEÇÃO')

df_cnpjs_ult_ativa_df_porte01e03.head(20)
df_Ativas_10anos = cnpjAtivos[(cnpjAtivos['anoAbertura'] > 2010) ]
df_Ativas_10anos = tabela_categoricas(df_Ativas_10anos, 'nm_secao', 'SEÇÃO')

df_Ativas_10anos.head() 


# Desenvolvido por Marcilon
fechada_10_ult_anos = pd.DataFrame(cnpjInativo['anoAbertura'].value_counts()).reset_index()
fechada_10_ult_anos = fechada_10_ult_anos[fechada_10_ult_anos['index'] > 2010]
fechada_10_ult_anos.columns = ['Ano Abertura' , 'Qtd. Empresas']
fechada_10_ult_anos['Ano Abertura'] = fechada_10_ult_anos['Ano Abertura'].apply(str)
fechada_10_ult_anos = fechada_10_ult_anos.sort_values(by='Ano Abertura')

ax = fechada_10_ult_anos.plot(kind = "bar", x = 'Ano Abertura',
                     title = " Quantidade de Empresas Fechadas nos ultima 10 anos",figsize = (10,4))
ax.set_xlabel("Ano Abertura")

fechada_10_ult_anos_DF = cnpjInativo[(cnpjInativo['anoAbertura'] > 2010) & (cnpjInativo['uf'] == 'DF')]
fechada_10_ult_anos_DF = tabela_categoricas(fechada_10_ult_anos_DF, 'nm_secao', 'SEÇÃO')

fechada_10_ult_anos_DF.head(20)
df_cnpjs_ult_inativa_df_porte01e03 = cnpjInativo[(cnpjInativo['anoAbertura'] > 2010) & (cnpjInativo['uf'] == 'DF') & cnpjInativo["porte_empresa"].isin(["01","03"])]
df_cnpjs_ult_inativa_df_porte01e03 = tabela_categoricas(df_cnpjs_ult_inativa_df_porte01e03, 'nm_secao', 'SEÇÃO')

df_cnpjs_ult_inativa_df_porte01e03.head(20)
df_Inativas_10anos = cnpjInativo[(cnpjInativo['anoAbertura'] > 2010) ]
df_Inativas_10anos = tabela_categoricas(df_Inativas_10anos, 'nm_secao', 'SEÇÃO')

df_Inativas_10anos.head(20) 
df_result = pd.merge(df_abertas_10anos, df_Ativas_10anos,how='left', left_on=['SEÇÃO'], right_on=['SEÇÃO'])
df_result = pd.merge(df_result, df_Inativas_10anos,how='left', left_on=['SEÇÃO'], right_on=['SEÇÃO'])

df_result_df = pd.merge(df_cnpjs_ult_ano_df, Ativa_10_ult_anos_DF,how='left', left_on=['SEÇÃO'], right_on=['SEÇÃO'])
df_result_df = pd.merge(df_result_df, fechada_10_ult_anos_DF, how='left', left_on=['SEÇÃO'], right_on=['SEÇÃO'])

df_result_df_porte1e3 = pd.merge(df_cnpjs_ult_ano_df_porte01e03, df_cnpjs_ult_ativa_df_porte01e03,how='left', left_on=['SEÇÃO'], right_on=['SEÇÃO'])
df_result_df_porte1e3 = pd.merge(df_result_df_porte1e3, df_cnpjs_ult_inativa_df_porte01e03, how='left', left_on=['SEÇÃO'], right_on=['SEÇÃO'])

df_result.rename(columns={"Qtd empresas_x": "Qtd_Empresas_Abertas", "Qtd empresas_y": "Qtd_Empresas_Ativas", "Qtd empresas": "Qtd_Empresas_Inativas"}, 
                 inplace=True)
df_result.drop(['%Total_x', '%Total_y', '%Total'], axis=1, inplace=True)

df_result_df.rename(columns={"Qtd empresas_x": "Qtd_Empresas_Abertas", "Qtd empresas_y": "Qtd_Empresas_Ativas", "Qtd empresas": "Qtd_Empresas_Inativas"}, 
                 inplace=True)
df_result_df.drop(['%Total_x', '%Total_y', '%Total'], axis=1, inplace=True)

df_result_df_porte1e3.rename(columns={"Qtd empresas_x": "Qtd_Empresas_Abertas", "Qtd empresas_y": "Qtd_Empresas_Ativas", "Qtd empresas": "Qtd_Empresas_Inativas"}, 
                 inplace=True)
df_result_df_porte1e3.drop(['%Total_x', '%Total_y', '%Total'], axis=1, inplace=True)
df_result['%_Empresas_Ativas'] = df_result.Qtd_Empresas_Ativas / df_result.Qtd_Empresas_Abertas
df_result['%_Empresas_Inativas'] = df_result.Qtd_Empresas_Inativas / df_result.Qtd_Empresas_Abertas

df_result_df['%_Empresas_Ativas'] = df_result_df.Qtd_Empresas_Ativas / df_result_df.Qtd_Empresas_Abertas
df_result_df['%_Empresas_Inativas'] = df_result_df.Qtd_Empresas_Inativas / df_result_df.Qtd_Empresas_Abertas

df_result_df_porte1e3['%_Empresas_Ativas'] = df_result_df_porte1e3.Qtd_Empresas_Ativas / df_result_df_porte1e3.Qtd_Empresas_Abertas
df_result_df_porte1e3['%_Empresas_Inativas'] = df_result_df_porte1e3.Qtd_Empresas_Inativas / df_result_df_porte1e3.Qtd_Empresas_Abertas
df_result.fillna(0, inplace=True)
df_result_df.fillna(0, inplace=True)
df_result_df_porte1e3.fillna(0, inplace=True)
df_result.sort_values('%_Empresas_Ativas', ascending=False)
df_result_df.sort_values('%_Empresas_Ativas', ascending=False)
df_result_df_porte1e3.sort_values('%_Empresas_Ativas', ascending=False)
result = df_result.sort_values('%_Empresas_Ativas', ascending=False)
result.set_index('SEÇÃO', inplace = True)
result['%_Empresas_Ativas'].plot.bar(color = 'blue', figsize = (15, 8))
plt.title('Gráfico1: Resultado para empresas ATIVAS - 10 últimos anos - BRASIL');
plt.ylabel('PERCENTUAL');
plt.show()
result = df_result_df.sort_values('%_Empresas_Ativas', ascending=False)
result.set_index('SEÇÃO', inplace = True)
result['%_Empresas_Ativas'].plot.bar(color = 'blue', figsize = (15, 8))
plt.title('Gráfico2: Resultado para empresas ATIVAS - 10 últimos anos - Distrito Federal');
plt.ylabel('PERCENTUAL');
plt.show()
result = df_result_df_porte1e3.sort_values('%_Empresas_Ativas', ascending=False)
result.set_index('SEÇÃO', inplace = True)
result['%_Empresas_Ativas'].plot.bar(color = 'blue', figsize = (15, 8))
plt.title('Gráfico2: Resultado para empresas ATIVAS - 10 últimos anos - Distrito Federal - Porte 1 e 3');
plt.ylabel('PERCENTUAL');
plt.show()
df_result.sort_values('%_Empresas_Inativas', ascending=False)
df_result_df.sort_values('%_Empresas_Inativas', ascending=False)
df_result_df_porte1e3.sort_values('%_Empresas_Inativas', ascending=False)
result = df_result.sort_values('%_Empresas_Inativas', ascending=False)
result.set_index('SEÇÃO', inplace = True)
result['%_Empresas_Inativas'].plot.bar(color = 'blue', figsize = (15, 8))
plt.title('Gráfico 3: Resultado para empresas INATIVAS - 10 últimos anos - BRASIL');
plt.ylabel('PERCENTUAL');
plt.show()
result = df_result_df.sort_values('%_Empresas_Inativas', ascending=False)
result.set_index('SEÇÃO', inplace = True)
result['%_Empresas_Inativas'].plot.bar(color = 'blue', figsize = (15, 8))
plt.title('Gráfico 4: Resultado para empresas INATIVAS - 10 últimos anos - Distrito Federal');
plt.ylabel('PERCENTUAL');
plt.show()
result = df_result_df_porte1e3.sort_values('%_Empresas_Inativas', ascending=False)
result.set_index('SEÇÃO', inplace = True)
result['%_Empresas_Inativas'].plot.bar(color = 'blue', figsize = (15, 8))
plt.title('Gráfico 4: Resultado para empresas INATIVAS - 10 últimos anos - Distrito Federal - Porte 1 e 3');
plt.ylabel('PERCENTUAL');
plt.show()