# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Load all teams

df_teams_2014 = pd.read_csv('../input/2014_clubes.csv')

df_teams_2015 = pd.read_csv('../input/2015_clubes.csv')

df_teams_2016 = pd.read_csv('../input/2016_clubes.csv')

df_teams_2017 = pd.read_csv('../input/2017_clubes.csv')
# Load all matches

df_matches_2014 = pd.read_csv('../input/2014_partidas.csv')

df_matches_2015 = pd.read_csv('../input/2015_partidas.csv')

df_matches_2016 = pd.read_csv('../input/2016_partidas.csv')

df_matches_2017 = pd.read_csv('../input/2017_partidas.csv')
# Join matches and teams info

df_2014 = df_matches_2014.set_index('clube_casa_id').join(df_teams_2014.set_index('id'), rsuffix='_casa')

df_2014 = df_2014.set_index('clube_visitante_id').join(df_teams_2014.set_index('id'), rsuffix='_visitante')

df_2014 = df_2014[['rodada', 'nome', 'placar_oficial_mandante', 'nome_visitante', 'placar_oficial_visitante']]

df_2014['YEAR'] = '2014'

df_2014.rename(columns = {'nome': 'MANDANTE', 'rodada': 'RODADA', 

                          'placar_oficial_mandante': 'GOLS_MANDANTE',

                          'nome_visitante': 'VISITANTE', 'placar_oficial_visitante': 'GOLS_VISITANTE'

                         }, inplace = True)



#df_2014.head()



df_2015 = df_matches_2015.set_index('clube_casa_id').join(df_teams_2015.set_index('id'), rsuffix='_casa')

df_2015 = df_2015.set_index('clube_visitante_id').join(df_teams_2015.set_index('id'), rsuffix='_visitante')

df_2015 = df_2015[['rodada', 'nome', 'placar_oficial_mandante', 'nome_visitante', 'placar_oficial_visitante']]

df_2015['YEAR'] = '2015'

df_2015.rename(columns = {'nome': 'MANDANTE', 'rodada': 'RODADA', 

                          'placar_oficial_mandante': 'GOLS_MANDANTE',

                          'nome_visitante': 'VISITANTE', 'placar_oficial_visitante': 'GOLS_VISITANTE'

                         }, inplace = True)



df_2016 = df_matches_2016.set_index('clube_casa_id').join(df_teams_2016.set_index('id'), rsuffix='_casa')

df_2016 = df_2016.set_index('clube_visitante_id').join(df_teams_2016.set_index('id'), rsuffix='_visitante')

df_2016 = df_2016[['rodada', 'nome', 'placar_oficial_mandante', 'nome_visitante', 'placar_oficial_visitante']]

df_2016['YEAR'] = '2016'

df_2016.rename(columns = {'nome': 'MANDANTE', 'rodada': 'RODADA', 

                          'placar_oficial_mandante': 'GOLS_MANDANTE',

                          'nome_visitante': 'VISITANTE', 'placar_oficial_visitante': 'GOLS_VISITANTE'

                         }, inplace = True)



df_2017 = df_matches_2017.set_index('clube_casa_id').join(df_teams_2017.set_index('id'), rsuffix='_casa')

df_2017 = df_2017.set_index('clube_visitante_id').join(df_teams_2017.set_index('id'), rsuffix='_visitante')

df_2017 = df_2017[['rodada_id', 'nome', 'placar_oficial_mandante', 'nome_visitante', 'placar_oficial_visitante']]

df_2017['YEAR'] = '2017'

df_2017.rename(columns = {'nome': 'MANDANTE', 

                          'rodada_id': 'RODADA', # Column name change in this table...

                          'placar_oficial_mandante': 'GOLS_MANDANTE',

                          'nome_visitante': 'VISITANTE', 'placar_oficial_visitante': 'GOLS_VISITANTE'

                         }, inplace = True)





df0 = df_2014

df1 = df0.append(df_2015, ignore_index=True)

df2 = df1.append(df_2016, ignore_index=True)

df  = df2.append(df_2017, ignore_index=True)



df.head()
df_Santos_Mandante = df[df.MANDANTE=='Santos']

df_Santos_Mandante.head()