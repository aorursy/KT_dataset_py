import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

# print(os.listdir("../input/opendata"))

# Datasets orindos do 

#http://www.tse.jus.br/eleicoes/estatisticas/repositorio-de-dados-eleitorais-1/repositorio-de-dados-eleitorais
brasil = pd.DataFrame()

arquivos = os.listdir("../input/opendata")

for arquivo in arquivos:

    estado = pd.read_csv('../input/opendata/'+arquivo, sep=";", encoding='ISO-8859-1', na_values=['#NULO#', -1,'#NE#', -3], low_memory=False)

    brasil = brasil.append(estado, ignore_index=True)
brasil.to_parquet('brasil.parquet.gzip', compression="gzip")