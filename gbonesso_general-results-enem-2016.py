import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
fields = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO',

         'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'TP_PRESENCA_MT',

         'TP_SEXO']

df = pd.read_csv('../input/microdados_enem_2016_coma.csv', encoding='latin-1', 

                 sep=',', usecols=fields)
# Create final grade column

df['GRADE'] = (

    df['NU_NOTA_CN'] + df['NU_NOTA_CH'] + df['NU_NOTA_LC'] + df['NU_NOTA_MT'] + 

    df['NU_NOTA_REDACAO']) / 5.0



# Replace NaN with zeros

df.fillna(0, inplace=True)

    

# Filter grades zero (probabilly the applicant doesn't make the exam...)

df = df[df.GRADE != 0]



# Filter by presence.

# 0 = Missed the test

# 1 = Attend the test

# 2 = Eliminated in the test

#df = df[df.TP_PRESENCA_CN == 1]

#df = df[df.TP_PRESENCA_CH == 1]

#df = df[df.TP_PRESENCA_LC == 1]

#df = df[df.TP_PRESENCA_MT == 1]
# Science of Nature Average Grade

df["NU_NOTA_CN"].mean()
# Humam Science Average Grade

df["NU_NOTA_CH"].mean()
# Language and Codes Average Grade

df["NU_NOTA_LC"].mean()
# Mathematics Average Grade

df["NU_NOTA_MT"].mean()
# Filtra os que receberam nota máxima na Redação (1000).

# No release fala em 77, mas só achei 76... 77 sem os filtros... ???

ESSAY_MAX_df = df[df.NU_NOTA_REDACAO == 1000]

ESSAY_MAX_df.groupby('NU_NOTA_REDACAO')['NU_NOTA_REDACAO'].agg(['count'])