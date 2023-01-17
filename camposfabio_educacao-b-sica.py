# Importação das bibliotecas para uso nas análises

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Código gerado automaticamente pelo kaggle na criação de um kernel. Permite ver a localização dos arquivos 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Leitura do arquivo de turmas a partir do csv utilizando o comando read_csv do pandas

# com parâmetros opcionais para separador e codificação

turmas = pd.read_csv('/kaggle/input/dados-brasil/Educacao_Basica_2018 - Turmas.csv',sep='|',encoding='latin-1' )
# Leitura dos dados sobre a média de alunos por turma do arquivo Excel

# Tive que incluir o parâmetro adicional skiprows para não ler as 8 primeiras linhas do excel

alunosTurma = pd.read_excel('/kaggle/input/dados-brasil/Media_Alunos_Turma - Escolas_2018.xlsx', skiprows =range(8))
# Mostra os 5 primeiros registros do dataSet

turmas.head()
# Leitura dos dados sobre a média de alunos por turma do arquivo Excel

alunosMunicipios = pd.read_excel('/kaggle/input/dados-brasil/Media_Alunos_Turma - Municipios_2018.xlsx', skiprows =range(8))
alunosMunicipios.head()