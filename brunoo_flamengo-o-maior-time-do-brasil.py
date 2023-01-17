import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as rc
df = pd.read_csv("../input/campeonato-brasileiro-de-futebol/campeonato-brasileiro-full.csv")

df.head()
print(df['Clube 1'].unique())
print(df['Clube 2'].unique())
df = df.replace(['INTERNACIONAL','REMO','MALUTROM','PALMEIRAS','BAHIA','CRUZEIRO','PONTE PRETA',

                 'FLUMINENSE','SPORT','VASCO','SANTOS','CORINTHIANS','JUVENTUDE','FLAMENGO','GUARANI','PAYSANDU','Botafogo-RJ'],

                ['Internacional','Remo','Malutrom','Palmeiras','Bahia','Cruzeiro','Ponte Preta',

                 'Fluminense','Sport','Vasco','Santos','Corinthians','Juventude','Flamengo','Guarani','Paysandu','Botafogo-rj'])
# Criar coluna 'Ano'

df['Data'] = pd.to_datetime(df['Data'])

df['Ano'] = df['Data'].dt.year



# Eliminar as partidas antes de 2003

df = df.drop(df[df.Ano < 2003].index)



# Eliminar colunas que não serão usadas

df = df.drop(['Horário', 'Dia','Rodada','Arena','Clube 1 Estado','Clube 2 Estado','Estado Clube Vencedor'], axis = 1)



# Renomear colunas

df = df.rename(columns={"Clube 1":"Mandante","Clube 2":"Visitante","Clube 1 Gols":"Mandante Gols",

                        "Clube 2 Gols":"Visitante Gols"})



# Trocar "-" por "Empate" na coluna 'Vencedor'

df = df.replace('-','Empate')



# Criar outro dataframe

dados = df.copy()



# Eliminar partidas que o Flamengo não jogou

df = df.drop(df[(df['Mandante'] != 'Flamengo') & (df['Visitante'] != 'Flamengo')].index)



# Vamos dar uma olhada no dataframe

df.head()
jogos = df.groupby('Ano')['Ano'].count()

jogos = pd.DataFrame(jogos)

jogos.columns = ['Jogos']

jogos.reset_index(level=0, inplace=True)

print(jogos)
## Partidas como mandante ##

# Criar um dataframe para partidas como mandante

mandante = df.copy()

mandante = mandante.drop(mandante[mandante['Mandante'] != 'Flamengo'].index)



# Criar um dataframe com nº de jogos por ano

jogos_mandante = mandante.groupby('Ano')['Ano'].count()

jogos_mandante = pd.DataFrame(jogos_mandante)

jogos_mandante.columns = ['Partidas mandante']

jogos_mandante.reset_index(level=0, inplace=True)



# Dataframe com o nº de gols por edição

gols_mandante = mandante.groupby('Ano')['Mandante Gols'].sum()

gols_mandante = pd.DataFrame(gols_mandante)

gols_mandante.columns = ['Mandante Gols']

gols_mandante.reset_index(level=0, inplace=True)



# Dataframe com partidas e gols por edição

gp_mandante = jogos_mandante.merge(gols_mandante, how='left', on='Ano')



# Criar coluna com a média de gols por partida

gp_mandante['GP Mandante'] = round(gp_mandante['Mandante Gols']/gp_mandante['Partidas mandante'],2)



## Partidas como visitante ##

# Criar um dataframe para partidas como visitante

visitante = df.copy()

visitante = visitante.drop(visitante[visitante['Visitante'] != 'Flamengo'].index)



# Criar um dataframe com nº de jogos por ano

jogos_visitante = visitante.groupby('Ano')['Ano'].count()

jogos_visitante = pd.DataFrame(jogos_visitante)

jogos_visitante.columns = ['Partidas visitante']

jogos_visitante.reset_index(level=0, inplace=True)



# Dataframe com o nº de gols por edição

gols_visitante = visitante.groupby('Ano')['Visitante Gols'].sum()

gols_visitante = pd.DataFrame(gols_visitante)

gols_visitante.columns = ['Visitante Gols']

gols_visitante.reset_index(level=0, inplace=True)



# Dataframe com partidas e gols por edição

gp_visitante = jogos_visitante.merge(gols_visitante, how='left', on='Ano')



# Criar coluna com a média de gols por partida

gp_visitante['GP Visitante'] = round(gp_visitante['Visitante Gols']/gp_visitante['Partidas visitante'],2)



## Todas as Partidas ##

gp_geral = gp_mandante.merge(gp_visitante, how='left', on='Ano')

gp_geral['Partidas'] = gp_geral['Partidas mandante'] + gp_geral['Partidas visitante']

gp_geral['Gols'] = gp_geral['Mandante Gols'] + gp_geral['Visitante Gols']

gp_geral['GP Geral'] = round(gp_geral['Gols']/gp_geral['Partidas'],2)
r = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]



nomes = ('2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016',

        '2017','2018','2019')



coluna1 = gp_geral['Gols']



# gráfico

barWidth = 0.9

# Criando a barra de gols marcados

plt.bar(r, coluna1, color='black', width=barWidth)



# Eixo X

plt.xticks(r, nomes)

plt.xlabel("Anos")

plt.ylabel("Gols")

plt.title("Gols por Ano")



# Tamanho do gráfico

plt.rcParams["figure.figsize"] = [12,6]

 

# Visualizar o gráfico

plt.show()
coluna1 = gp_geral['GP Geral']



# gráfico

barWidth = 0.9

# Criando a barra de gols marcados por partida

plt.bar(r, coluna1, color='red', width=barWidth)



# Eixo X

plt.xticks(r, nomes)

plt.xlabel("Anos")

plt.ylabel("Gols")

plt.title("Gols por Partida")



# Tamanho do gráfico

plt.rcParams["figure.figsize"] = [12,6]

 

# Visualizar o gráfico

plt.show()
coluna1 = gp_geral['Mandante Gols']/gp_geral['Gols']

coluna2 = gp_geral['Visitante Gols']/gp_geral['Gols']



# gráfico

barWidth = 0.9

# Criando a barra de gols em casa

plt.bar(r, coluna1, color='black', edgecolor='white', width=barWidth,label='Gols em Casa')

# Criando a barra de gols fora de casa

plt.bar(r, coluna2, bottom=coluna1, color='red', edgecolor='white', width=barWidth,label='Gols Fora')



# Eixo X

plt.xticks(r, nomes)

plt.xlabel("Anos")

plt.ylabel("Proporção")

plt.title("Dist. de gols em casa e fora")

 

plt.rcParams["figure.figsize"] = [12,6]

    

# Visualizar o gráfico

plt.legend()

plt.show()
# Gols por jogo

gols = gp_geral['Gols'].sum()

partidas =  gp_geral['Partidas'].sum()

gols_partida = gols/partidas

print("O Flamengo tem uma média de "+str(round(gols_partida,1))+ " gols por partida")



# Gols por jogo em casa

gols_casa = gp_geral['Mandante Gols'].sum()

partidas_casa =  gp_geral['Partidas mandante'].sum()

gols_partida_casa = gols_casa/partidas_casa

print("Em casa, o Flamengo tem uma média de "+str(round(gols_partida_casa,1))+ " gols por partida")



# Gols por jogo fora de casa

gols_fora = gp_geral['Visitante Gols'].sum()

partidas_fora =  gp_geral['Partidas visitante'].sum()

gols_partida_fora = gols_fora/partidas_fora

print("Fora de casa, o Flamengo tem uma média de "+str(round(gols_partida_fora,1))+ " gols por partida")
# Contar o número de vitórias

vitorias_casa=mandante.groupby('Ano')['Vencedor'].apply(lambda x: (x=='Flamengo').sum()).reset_index(name='Vitórias')



# Contar o número de empates

empates_casa=mandante.groupby('Ano')['Vencedor'].apply(lambda x: (x=='Empate').sum()).reset_index(name='Empates')



# Juntar os dataframes

resultado_casa = jogos_mandante.merge(vitorias_casa, how='left', on='Ano')

resultado_casa = resultado_casa.merge(empates_casa, how='left', on='Ano')



# Ajeitar coluna

resultado_casa = resultado_casa.rename(columns={"Partidas mandante":"Jogos"}) 



# Criar coluna de derrotas

resultado_casa['Derrotas']= resultado_casa['Jogos'] - resultado_casa['Vitórias']-resultado_casa['Empates']
coluna1 = resultado_casa['Vitórias']/resultado_casa['Jogos']

coluna2 = resultado_casa['Empates']/resultado_casa['Jogos']

coluna3 = resultado_casa['Derrotas']/resultado_casa['Jogos']



# gráfico

barWidth = 0.9

# Criando a barra de vitórias

plt.bar(r, coluna1, color='black', edgecolor='white', width=barWidth,label='Vitórias')

# Criando a barra de empates

plt.bar(r, coluna2, bottom=coluna1, color='red', edgecolor='white', width=barWidth,label='Empates')

# Criando a barra de derrotas

plt.bar(r, coluna3, bottom=[i+j for i,j in zip(coluna1, coluna2)], color='lightgray', edgecolor='white',

        width=barWidth,label='Derrotas')



# Eixo X

plt.xticks(r, nomes)

plt.xlabel("Anos")

plt.ylabel("Proporção")

plt.title("Dist. de resultados em casa")

 

# Visualizar o gráfico

plt.legend()

plt.show()
resultado_casa['% Aprov.']= round((3*resultado_casa['Vitórias']+resultado_casa['Empates'])/(3*resultado_casa['Jogos']),2)

print(resultado_casa)
coluna1 = resultado_casa['% Aprov.']



# gráfico

barWidth = 0.9

# Criando a barra de aproveitamento

plt.bar(r, coluna1, color='black', width=barWidth)



# Eixo X

plt.xticks(r, nomes)

plt.xlabel("Anos")

plt.ylabel("Aproveitamento")

plt.title("Aproveitamento em casa")



plt.rcParams["figure.figsize"] = [12,6]

 

# Visualizar o gráfico

plt.show()
# Contar o número de vitórias

vitorias_fora=visitante.groupby('Ano')['Vencedor'].apply(lambda x: (x=='Flamengo').sum()).reset_index(name='Vitórias')



# Contar o número de empates

empates_fora=visitante.groupby('Ano')['Vencedor'].apply(lambda x: (x=='Empate').sum()).reset_index(name='Empates')



# Juntar os dataframes

resultado_fora = jogos_visitante.merge(vitorias_fora, how='left', on='Ano')

resultado_fora = resultado_fora.merge(empates_fora, how='left', on='Ano') 



# Ajeitar coluna

resultado_fora = resultado_fora.rename(columns={"Partidas visitante":"Jogos"})



# Criar coluna de derrotas

resultado_fora['Derrotas']= resultado_fora['Jogos'] - resultado_fora['Vitórias']-resultado_fora['Empates']
coluna1 = resultado_fora['Vitórias']/resultado_fora['Jogos']

coluna2 = resultado_fora['Empates']/resultado_fora['Jogos']

coluna3 = resultado_fora['Derrotas']/resultado_fora['Jogos']



# gráfico

barWidth = 0.9

# Criando a barra de vitórias

plt.bar(r, coluna1, color='black', edgecolor='white', width=barWidth,label='Vitórias')

# Criando a barra de empates

plt.bar(r, coluna2, bottom=coluna1, color='red', edgecolor='white', width=barWidth,label='Empates')

# Criando a barra de derrotas

plt.bar(r, coluna3, bottom=[i+j for i,j in zip(coluna1, coluna2)], color='lightgray', edgecolor='white',

        width=barWidth,label='Derrotas')



# Eixo X

plt.xticks(r, nomes)

plt.xlabel("Anos")

plt.ylabel("Proporção")

plt.title("Dist. de resultados fora de casa")

 

# Visualizar o gráfico

plt.legend()

plt.show()
resultado_fora['% Aprov.']= round((3*resultado_fora['Vitórias']+resultado_fora['Empates'])/(3*resultado_fora['Jogos']),2)



coluna1 = resultado_fora['% Aprov.']



# gráfico

barWidth = 0.9

# Criando a barra de aproveitamento

plt.bar(r, coluna1, color='black', width=barWidth)



# Eixo X

plt.xticks(r, nomes)

plt.xlabel("Anos")

plt.ylabel("Aproveitamento")

plt.title("Aproveitamento fora de casa")



plt.rcParams["figure.figsize"] = [12,6]

 

# Visualizar o gráfico

plt.show()
# Deletar colunas de aproveitamento

resultado_casa = resultado_casa.drop(['% Aprov.'], axis = 1)

resultado_fora = resultado_fora.drop(['% Aprov.'], axis = 1)



# Deletar coluna repetida

resultado_casa = resultado_casa.drop(['Jogos'], axis = 1)

resultado_fora = resultado_fora.drop(['Jogos'], axis = 1)



# Renomear colunas

resultado_casa = resultado_casa.rename(columns={"Vitórias":"Vit casa","Empates":"Emp casa","Derrotas":"Der casa"})

resultado_fora = resultado_fora.rename(columns={"Vitórias":"Vit fora","Empates":"Emp fora","Derrotas":"Der fora"})



# Juntar dataframes

resultado = jogos.merge(resultado_casa, how='left', on='Ano')

resultado = resultado.merge(resultado_fora, how='left', on='Ano')



# Coluna de Pontos

resultado['Pts'] = 3*(resultado['Vit casa']+resultado['Vit fora'])+resultado['Emp casa']+resultado['Emp fora']



# Coluna com a proporção de pontos conquistados em casa

resultado['% Pts casa']= round((3*resultado['Vit casa']+resultado['Emp casa'])/resultado['Pts'],2)



# Coluna com a proporção de pontos conquistados fora de casa

resultado['% Pts fora']= round((3*resultado['Vit fora']+resultado['Emp fora'])/resultado['Pts'],2)



resultado['Vit'] = resultado['Vit casa'] + resultado['Vit fora']

resultado['Der'] = resultado['Der casa'] + resultado['Der fora']

resultado['Emp'] = resultado['Emp casa'] + resultado['Emp fora']
coluna1 = resultado['Vit']/resultado['Jogos']

coluna2 = resultado['Emp']/resultado['Jogos']

coluna3 = resultado['Der']/resultado['Jogos']



# gráfico

barWidth = 0.9

# Criando a barra de vitórias

plt.bar(r, coluna1, color='black', edgecolor='white', width=barWidth,label='Vitórias')

# Criando a barra de empates

plt.bar(r, coluna2, bottom=coluna1, color='red', edgecolor='white', width=barWidth,label='Empates')

# Criando a barra de derrotas

plt.bar(r, coluna3, bottom=[i+j for i,j in zip(coluna1, coluna2)], color='lightgray', edgecolor='white',

        width=barWidth,label='Derrotas')



# Eixo X

plt.xticks(r, nomes)

plt.xlabel("Anos")

plt.ylabel("Proporção")

plt.title("Dist. de resultados")

 

# Visualizar o gráfico

plt.legend()

plt.show()
resultado['% Aprov.']= round((3*resultado['Vit']+resultado['Emp'])/(3*resultado['Jogos']),2)



coluna1 = resultado['% Aprov.']



# gráfico

barWidth = 0.9

# Criando a barra de aproveitamento

plt.bar(r, coluna1, color='black', width=barWidth)



# Eixo X

plt.xticks(r, nomes)

plt.xlabel("Anos")

plt.ylabel("Aproveitamento")

plt.title("Aproveitamento de pontos")



plt.rcParams["figure.figsize"] = [12,6]

 

# Visualizar o gráfico

plt.show()
coluna1 = resultado['Pts']



# gráfico

barWidth = 0.9

# Criando a barra de pontos conquistados

plt.bar(r, coluna1, color='black', width=barWidth)



# Eixo X

plt.xticks(r, nomes)

plt.xlabel("Anos")

plt.ylabel("Pontos")

plt.title("Pontos Conquistados")



plt.rcParams["figure.figsize"] = [12,6]

 

# Visualizar o gráfico

plt.show()
coluna1 = resultado['% Pts casa']

coluna2 = resultado['% Pts fora']



# gráfico

barWidth = 0.9

# Criando a barra de pontos conquistados em casa

plt.bar(r, coluna1, color='black', edgecolor='white', width=barWidth,label='% Pts Casa')

# Criando a barra de pontos conquistados fora de casa

plt.bar(r, coluna2, bottom=coluna1, color='red', edgecolor='white', width=barWidth,label='% Pts Fora')



# Eixo X

plt.xticks(r, nomes)

plt.xlabel("Anos")

plt.ylabel("Proporção")

plt.title("Dist. de pontos conquistados em casa e fora")

 

plt.rcParams["figure.figsize"] = [12,6]

    

# Visualizar o gráfico

plt.legend()

plt.show()
# Somar as vitórias por adversário

adversarios_vit = df.groupby('Vencedor')['Vencedor'].count()

adversarios_vit = pd.DataFrame(adversarios_vit)

adversarios_vit.columns = ['Vitórias']

adversarios_vit.reset_index(level=0, inplace=True)



# Eliminar as linhas com as vitórias do Flamengo e Empates

adversarios_vit = adversarios_vit.drop(adversarios_vit[(adversarios_vit['Vencedor'] == 'Flamengo')].index)

adversarios_vit = adversarios_vit.drop(adversarios_vit[(adversarios_vit['Vencedor'] == 'Empate')].index)



# Ordenar o Dataframe em ordem decrescente

adversarios_vit.sort_values(by=['Vitórias'], inplace=True, ascending=False)



# Criar gráfico

ax = adversarios_vit.plot.barh(x='Vencedor', y='Vitórias',color ='black',figsize=(10,10))
# Pelo dataframe 'Mandante'

gols_adversarios_casa = mandante.groupby('Visitante')['Visitante Gols'].sum()

gols_adversarios_casa = pd.DataFrame(gols_adversarios_casa)

gols_adversarios_casa.columns = ['Gols Casa']

gols_adversarios_casa.reset_index(level=0, inplace=True)

gols_adversarios_casa.columns = ['Time','Gols Casa']



# Pelo dataframe 'Visitante'

gols_adversarios_fora = visitante.groupby('Mandante')['Mandante Gols'].sum()

gols_adversarios_fora = pd.DataFrame(gols_adversarios_fora)

gols_adversarios_fora.columns = ['Gols Fora']

gols_adversarios_fora.reset_index(level=0, inplace=True)

gols_adversarios_fora.columns = ['Time','Gols Fora']



# Juntando os dataframes

gols_adversarios = gols_adversarios_casa.merge(gols_adversarios_fora, how='left', on='Time')



# Criar coluna total com os gols marcados por adversarios

gols_adversarios['Gols'] = gols_adversarios['Gols Casa'] + gols_adversarios['Gols Fora']



# Ordenar o dataframe

gols_adversarios.sort_values(by=['Gols'], inplace=True, ascending=False)



# Criar gráfico

ax = gols_adversarios.plot.barh(x='Time', y='Gols',color ='black',figsize=(10,10))
# Criar um dataframe com a coluna de perdedores

perdedores = df.copy()



# Criar a coluna de perdedores

condicoes = [(perdedores['Mandante'] == perdedores['Vencedor']),(perdedores['Visitante'] == perdedores['Vencedor'])]

valores = [perdedores['Visitante'], perdedores['Mandante']]

perdedores['Perdedor'] = np.select(condicoes, valores, default='Empate')



# Contar o número de vitórias do Flamengo sobre cada adversário

flamengo_vit = perdedores.groupby('Perdedor')['Perdedor'].count()

flamengo_vit = pd.DataFrame(flamengo_vit)

flamengo_vit.columns = ['Vitórias']

flamengo_vit.reset_index(level=0, inplace=True)



# Eliminar as linhas com as vitórias do Flamengo e Empates

flamengo_vit = flamengo_vit.drop(flamengo_vit[(flamengo_vit['Perdedor'] == 'Flamengo')].index)

flamengo_vit = flamengo_vit.drop(flamengo_vit[(flamengo_vit['Perdedor'] == 'Empate')].index)



# Ordenar o Dataframe em ordem decrescente

flamengo_vit.sort_values(by=['Vitórias'], inplace=True, ascending=False)



# Criar gráfico

ax = flamengo_vit.plot.barh(x='Perdedor', y='Vitórias',color ='red',figsize=(10,10))
# Pelo dataframe 'Mandante'

gols_flamengo_casa = mandante.groupby('Visitante')['Mandante Gols'].sum()

gols_flamengo_casa = pd.DataFrame(gols_flamengo_casa)

gols_flamengo_casa.columns = ['Gols Flamengo Casa']

gols_flamengo_casa.reset_index(level=0, inplace=True)

gols_flamengo_casa.columns = ['Time','Gols Flamengo Casa']



# Pelo dataframe 'Visitante'

gols_flamengo_fora = visitante.groupby('Mandante')['Visitante Gols'].sum()

gols_flamengo_fora = pd.DataFrame(gols_flamengo_fora)

gols_flamengo_fora.columns = ['Gols Flamengo Fora']

gols_flamengo_fora.reset_index(level=0, inplace=True)

gols_flamengo_fora.columns = ['Time','Gols Flamengo Fora']



# Juntando os dataframes

gols_flamengo = gols_flamengo_casa.merge(gols_flamengo_fora, how='left', on='Time')



# Criar coluna total com os gols marcados por adversarios

gols_flamengo['Gols'] = gols_flamengo['Gols Flamengo Casa'] + gols_flamengo['Gols Flamengo Fora']



# Ordenar o dataframe

gols_flamengo.sort_values(by=['Gols'], inplace=True, ascending=False)



# Criar gráfico

ax = gols_flamengo.plot.barh(x='Time', y='Gols',color ='red',figsize=(10,10))
# Função que cria lista dos dataframes de cada ano

def listar_df(dataframe):

    new_df = dataframe.copy()  # Copia o dataframe input

    anos = new_df['Ano'].unique() # Vai criar uma array com cada ano que aparece na coluna

    anos = anos.tolist()    # Transforma a array em uma lista

    

    # Filtrar dataframes

    dbs = []

    for ano in anos:  # Vai pegar os anos na lista e usar como critério para filtrar os dataframes

        db = new_df.loc[new_df['Ano'] == ano]

        dbs.append(db)

       

    return dbs



def conseguir_times(dataframe):

    # Registrar times

    df = dataframe.copy()

    times = df['Mandante'].unique()

    times = times.tolist()

    return times

    

# Função que conta o número de vitórias

def contar_vitórias(dataframe,lista):

    vitorias_time = []

    for item in lista:

        filter = dataframe["Vencedor"] == item

        vitorias = dataframe[filter]['Vencedor'].count().astype(np.int64)

        vitorias_time += [vitorias,]                                                        

    return vitorias_time



# Função que conta o número de empates

def contar_empates(dataframe, lista ):

    empates_time = []

    for item in lista:

        df = dataframe.copy()

        filter1 = (df["Mandante"] == item) | (df["Visitante"] == item)

        filter2 = df["Vencedor"] == 'Empate'

        df = df[(filter1) & (filter2)]

        empates = df['Vencedor'].count().astype(np.int64)

        empates_time += [empates,] 

    return empates_time



# Função que conta o número de derrotas

def contar_derrotas(dataframe, lista):

    derrotas_time = []

    for item in lista:

        df = dataframe.copy()

        filter1 = (df["Mandante"] == item) | (df["Visitante"] == item)

        filter2 = (df["Vencedor"] != item) & (df["Vencedor"] != 'Empate')

        df = df[(filter1) & (filter2)]

        derrotas = df['Vencedor'].count().astype(np.int64)

        derrotas_time += [derrotas,] 

    return derrotas_time



# Função para contar gols

def contar_gp(dataframe, lista):

    gp_time = [] # Gols marcados

    for item in lista:

        df = dataframe.copy()

        # Somar gols marcados em casa

        filter1 = (df["Mandante"] == item)

        df1 = df[(filter1)]

        gp1 = df1['Mandante Gols'].sum().astype(np.int64)

        # Somar gols marcados fora

        filter2 = (df["Visitante"] == item)

        df2 = df[(filter2)]

        gp2 = df2['Visitante Gols'].sum().astype(np.int64)

        gp = gp1 + gp2

        # Somar gols

        gp_time += [gp,]



    return gp_time



def contar_gc(dataframe, lista):

    gc_time = [] # Gols sofridos

    for item in lista:

        df = dataframe.copy()

        # Somar gols sofridos em casa

        filter3 = (df["Mandante"] == item)

        df3 = df[(filter3)]

        gc1 = df3['Visitante Gols'].sum().astype(np.int64)

        # Somar gols sofridos fora

        filter4 = (df["Visitante"] == item)

        df4 = df[(filter4)]

        gc2 = df4['Mandante Gols'].sum().astype(np.int64)

        gc = gc1 + gc2

        # Somar gols

        gc_time += [gc,]

        

    return gc_time



# Criar lista com nossos dataframes

dataframes = listar_df(dados)



# Loop final

tabelas = []

for dataframe in dataframes:

    times = conseguir_times(dataframe)

    vitorias = contar_vitórias(dataframe,times)

    empates = contar_empates(dataframe,times)

    derrotas = contar_derrotas(dataframe,times)

    gp = contar_gp(dataframe,times)

    gc = contar_gc(dataframe,times)

    

    # Transformar listas em um dataframe

    tabela = pd.DataFrame(list(zip(times,vitorias,empates,derrotas,gp,gc)), 

               columns =['times','vitorias','empates','derrotas','GP','GC']) 

    #Criar coluna de pontos e de saldo de gol

    tabela['Pts'] = 3*tabela['vitorias'] + tabela['empates']

    tabela['SG'] = tabela['GP'] + tabela['GC']

    # Ordenar os times

    tabela = tabela.sort_values(['Pts', 'vitorias', 'SG','GP'], ascending=[False, False, False,False])

    # Criar coluna com posição

    posição = len(tabela) + 1

    tabela['Posição'] = [i for i in range(1,posição)]

    # Ajustar ordem das colunas

    tabela = tabela[['times','Pts','vitorias','empates','derrotas','GP','GC','SG','Posição']]

    

    tabelas.append(tabela)

    

# Obter a classificação do Flamengo ao longo dos anos

fla = []

for tabela in tabelas:

    for index,row in tabela.iterrows():

        if row['times'] == 'Flamengo':

            pos = row['Posição']

    fla.append(pos)



def conseguir_ano(dataframe):

    new_df = dataframe.copy()  # Copia o dataframe input

    anos = new_df['Ano'].unique() # Vai criar uma array com cada ano que aparece na coluna

    anos = anos.tolist()    # Transforma a array em uma lista

    return anos



ano = conseguir_ano(df)



campeões = pd.DataFrame(list(zip(ano,fla)),columns =['Ano','Classificação'])



campeões.head(17)
# Vitórias

vitorias = resultado['Vit'].sum()

print('O Flamengo venceu '+str(vitorias) + ' jogos.')



# Derrotas

derrotas = resultado['Der'].sum()

print('O Flamengo perdeu '+str(derrotas) + ' jogos.')



# Empates

empates = resultado['Emp'].sum()

print('O Flamengo empatou '+str(empates) + ' jogos.')



# Gols Marcados

print('O Flamengo marcou '+str(gols) + ' gols.')



# Gols por partida

print('O Flamengo marcou '+str(round(gols_partida,1)) + ' gols por partida.')



# Aproveitamento de pontos

pontos = 3*vitorias+empates

aproveitamento = pontos/(3*(vitorias+derrotas+empates))

print('O Flamengo teve aproveitamento de '+str(100*round(aproveitamento,1)) + '%.')