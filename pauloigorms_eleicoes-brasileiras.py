# Última execução
import datetime
import pytz
print(datetime.datetime.now(pytz.timezone('America/Manaus')).strftime('%Y-%m-%d %H:%M'))
# IMPORTE DAS LIBS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# IMPORTANDO DADOS
candidatura = pd.read_csv('../input/eleicao-br/candidatura.csv', low_memory=False)
bem_declarado = pd.read_csv('../input/eleicao-br/bem-declarado.csv', low_memory=False)
candidatura.dtypes
candidatura_2018 = candidatura[candidatura['ano_eleicao'] >= 2014]
# gráfico por raca
pd.DataFrame(candidatura_2018.groupby(['ano_eleicao', 'descricao_cor_raca']).count()['codigo_cor_raca'])
# gráfico por sexo
pd.DataFrame(candidatura_2018.groupby(['descricao_genero']).count()['codigo_genero'])