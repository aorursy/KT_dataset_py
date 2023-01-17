# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Organizando a tabela de dados dos pacientes
grupo_fleury= pd.read_csv("/kaggle/input/grupofleury-pacientes-dataset/GrupoFleury_Pacientes_2.csv")

name='ID_PACIENTE|IC_SEXO|AA_NASCIMENTO|CD_PAIS|CD_UF|CD_MUNICIPIO|CD_CEPREDUZIDO'

grupo_fleury[['ID_PACIANTE','IC_SEXO','AA_NASCIMENTO','CD_PAIS','CD_UF','CD_MUNICIPIO',
                                  'CD_CEPREDUZIDO']]=grupo_fleury[name].str.split("|", expand = True)
del grupo_fleury[name]
grupo_fleury.head()
#Organizando a tabela de dados dos exames
hsl_exames= pd.read_csv("/kaggle/input/hsl-dataset/HSL_Exames_2.csv",usecols=[0])
namecol='ID_PACIENTE|ID_ATENDIMENTO|DT_COLETA|DE_ORIGEM|DE_EXAME|DE_ANALITO|DE_RESULTADO|CD_UNIDADE|DE_VALOR_REFERENCIA'
hsl_exames[namecol.split("|")]=hsl_exames[namecol].str.split("|", expand = True,)
del hsl_exames[namecol]
hsl_exames.head()

#Organizando a tabela de dados desfecho final
desf= pd.read_csv("/kaggle/input/hsl-desfecho/HSL_Desfechos_2.csv")

name='id_paciente|id_atendimento|dt_atendimento|de_tipo_atendimento|id_clinica|de_clinica|dt_desfecho|de_desfecho'

desf[['ID_PACIANTE','ID_ATENDIMENTO','DT_ATENDIMENTO','DE_TIPO_ATENDIMENTO','ID_CLINICA','DE_CLINICA',
              'DT_DESFECHO','DE_DESFECHO']]=desf[name].str.split("|", expand = True)
del desf[name]
desf.head()

df_atr=pd.DataFrame([grupo_fleury.columns,hsl_exames.columns,desf.columns]).T
df_atr.columns = ['ATR_PACIENTE', 'ATR_EXAME','ATR_DESFECHO']
df_atr


#ACHO QUE NÃO DÁ PRA ANALIZAR COM ESSE DATASET DA FAPESP
#Atributos da ficha de atendimento de Mossoró
d_moss= pd.read_csv("/kaggle/input/mossor-dataset/ficha.CSV")
d_moss_atrr=pd.DataFrame([d_moss.columns]).T
d_moss_atrr.columns = ['ATR_FICHA_INVESTIGAÇÃO']
#dados que serão comparados
d_fapesp=pd.DataFrame([grupo_fleury.columns,desf.columns]).T
d_fapesp.columns =['ATR_PACIENTE','ATR_DESFECHO']

d_moss_atrr
#Atributos que também estão do formulário
df=d_moss_atrr.merge(d_fapesp.iloc[:,0:1],left_on=d_moss_atrr['ATR_FICHA_INVESTIGAÇÃO'],
                     right_on=d_fapesp['ATR_PACIENTE'])
dff=d_moss_atrr.merge(d_fapesp.iloc[:,1:2],left_on=d_moss_atrr['ATR_FICHA_INVESTIGAÇÃO'],
                     right_on=d_fapesp['ATR_DESFECHO'])
atrr_iguais=pd.DataFrame([df['ATR_PACIENTE'],dff['ATR_DESFECHO']]).T
atrr_iguais
#a=d_moss_atrr[~d_moss_atrr.isin(atrr_iguais.iloc[:,1:2])].dropna()
d_moss_atrr[(~d_moss_atrr['ATR_FICHA_INVESTIGAÇÃO'].isin(atrr_iguais['ATR_DESFECHO'])) &
            (~d_moss_atrr['ATR_FICHA_INVESTIGAÇÃO'].isin((atrr_iguais['ATR_PACIENTE'])))]