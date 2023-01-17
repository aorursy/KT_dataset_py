import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # plots

import matplotlib.pyplot as plt 



# Exporting files

scouts_2014   = pd.read_csv('../input/2014_scouts.csv')

scouts_2015   = pd.read_csv('../input/2015_scouts.csv')

scouts_2016   = pd.read_csv('../input/2016_scouts.csv')

scouts_2017   = pd.read_csv('../input/2017_scouts.csv')



partidas_2014 = pd.read_csv('../input/2014_partidas.csv')

partidas_2015 = pd.read_csv('../input/2015_partidas.csv')

partidas_2016 = pd.read_csv('../input/2016_partidas.csv')

partidas_2017 = pd.read_csv('../input/2017_partidas.csv')



atletas_2014  = pd.read_csv('../input/2014_atletas.csv')

atletas_2015  = pd.read_csv('../input/2015_atletas.csv')

atletas_2016  = pd.read_csv('../input/2016_atletas.csv')

atletas_2017  = pd.read_csv('../input/2017_atletas.csv')



clubes_2014   = pd.read_csv('../input/2014_clubes.csv')

clubes_2015   = pd.read_csv('../input/2015_clubes.csv')

clubes_2016   = pd.read_csv('../input/2016_clubes.csv')

clubes_2017   = pd.read_csv('../input/2017_clubes.csv')



posicoes = pd.read_csv('../input/posicoes.csv')

pontuacao= pd.read_csv('../input/pontuacao.csv')

status = pd.read_csv('../input/status.csv')
# Contribuition by Gustavo Bonesso

# Join matches and teams info - 

df_2014 = partidas_2014.set_index('clube_casa_id').join(clubes_2014.set_index('id'), rsuffix='_casa')

df_2014 = df_2014.set_index('clube_visitante_id').join(clubes_2014.set_index('id'), rsuffix='_visitante')

df_2014 = df_2014[['rodada', 'nome', 'placar_oficial_mandante', 'nome_visitante', 'placar_oficial_visitante']]

df_2014['YEAR'] = '2014'

df_2014.rename(columns = {'nome': 'MANDANTE', 

                          'rodada': 'RODADA', 

                          'placar_oficial_mandante': 'GOLS_MANDANTE',

                          'nome_visitante': 'VISITANTE', 

                          'placar_oficial_visitante': 'GOLS_VISITANTE'}, 

                           inplace = True)



df_2015 = partidas_2015.set_index('clube_casa_id').join(clubes_2015.set_index('id'), rsuffix='_casa')

df_2015 = df_2015.set_index('clube_visitante_id').join(clubes_2015.set_index('id'), rsuffix='_visitante')

df_2015 = df_2015[['rodada', 'nome', 'placar_oficial_mandante', 'nome_visitante', 'placar_oficial_visitante']]

df_2015['YEAR'] = '2015'

df_2015.rename(columns = {'nome': 'MANDANTE', 

                          'rodada': 'RODADA', 

                          'placar_oficial_mandante': 'GOLS_MANDANTE',

                          'nome_visitante': 'VISITANTE', 

                          'placar_oficial_visitante': 'GOLS_VISITANTE'}, 

                           inplace = True)



df_2016 = partidas_2016.set_index('clube_casa_id').join(clubes_2016.set_index('id'), rsuffix='_casa')

df_2016 = df_2016.set_index('clube_visitante_id').join(clubes_2016.set_index('id'), rsuffix='_visitante')

df_2016 = df_2016[['rodada', 'nome', 'placar_oficial_mandante', 'nome_visitante', 'placar_oficial_visitante']]

df_2016['YEAR'] = '2016'

df_2016.rename(columns = {'nome': 'MANDANTE', 

                          'rodada': 'RODADA', 

                          'placar_oficial_mandante': 'GOLS_MANDANTE',

                          'nome_visitante': 'VISITANTE', 

                          'placar_oficial_visitante': 'GOLS_VISITANTE'}, 

                           inplace = True)



df_2017 = partidas_2017.set_index('clube_casa_id').join(clubes_2017.set_index('id'), rsuffix='_casa')

df_2017 = df_2017.set_index('clube_visitante_id').join(clubes_2017.set_index('id'), rsuffix='_visitante')

df_2017 = df_2017[['rodada_id', 'nome', 'placar_oficial_mandante', 'nome_visitante', 'placar_oficial_visitante']]

df_2017['YEAR'] = '2017'

df_2017.rename(columns = {'nome': 'MANDANTE', 'rodada_id': 'RODADA', #column name changed...

                          'placar_oficial_mandante': 'GOLS_MANDANTE',

                          'nome_visitante': 'VISITANTE', 'placar_oficial_visitante': 'GOLS_VISITANTE'}, 

                           inplace = True)



# Join scouts and team info -

# dc_2014 = ...



# Join dfs

df0 = df_2014

df1 = df0.append(df_2015, ignore_index=True)

df2 = df1.append(df_2016, ignore_index=True)

df  = df2.append(df_2017, ignore_index=True)



# Aline table by RODADA 

#df = df.sort_values(by='RODADA')
# Select all matches within a team's perfomance

team = 'Flamengo'



df_flamengo = df[(df.MANDANTE == team)]

df_flamengo = df_flamengo.append(df[(df.VISITANTE == team)])



# Aline data by Rodada

df_flamengo = df_flamengo.sort_values(by='RODADA')

df_flamengo
# Analyse to see how good is MANDANTE x VISITANTE

sns.jointplot(x="GOLS_MANDANTE", y="GOLS_VISITANTE", data=df, size = 10, space = 0);
# Analyse scouts from a specific player

# - so in the future we could train the network with one entry_labels = 'atleta_id'

Atleta_id = 36443

scouts_2014_atleta = scouts_2014[scouts_2014.atleta_id == Atleta_id]

scouts_2014_atleta
# Price variation of the player within "rodadas"

# With more years_data...

x = scouts_2014_atleta['rodada']

y = scouts_2014_atleta['preco_num']

plt.plot(x,y)
# Futuramente:

# Incluir "clube_adversario" para cada partida - em analise de cada jogador.



# - Criar um dado com todas as analises de determinado jogador -

# Input labels = atleta_id, clube, "clube_adversario", mandante/visitante(1,0)

# Output Labels = nota

# Treinar 70%

# Validar 20%

# Testar 10%