import numpy as np
import pandas as pd

# Caminho do Arquivo
file = '/kaggle/input/corona-virus-brazil/brazil_covid19_cities.csv'
# Leitura do Arquivo
df_pa = pd.read_csv(file).rename(columns={'date':   'Data',
                                          'state':  'Estado',
                                          'name':   'Municipio',
                                          'code':   'Cod.Mun.', 
                                          'cases':  'Casos Confirmados', 
                                          'deaths': 'Mortes'})

# Filtrando dados Municipio (Pouso Alegre)
df_pa = df_pa[df_pa.Municipio.eq('Pouso Alegre')]

# Exibindo o resultado (última atualização)
print('Dados do Município - [Pouso Alegre]')
print(df_pa.iloc[-1])

# Destruindo o objeto
del df_pa
