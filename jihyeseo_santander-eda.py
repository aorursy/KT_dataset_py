# https://www.kaggle.com/c/integer-sequence-learning

import math
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


from wordcloud import WordCloud, STOPWORDS

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
%matplotlib inline
import matplotlib.pyplot as plt 
plt.rcParams["figure.figsize"] = [16, 12]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
filenames = check_output(["ls", "../input/"]).decode("utf8").strip()
# helpful character encoding module
import chardet
df = pd.read_csv('../input/train_ver2.csv')
dg = pd.read_csv('../input/test_ver2.csv')
dh = pd.read_csv('../input/sample_submission.csv')
df.sample(5)
dg.sample(5)
dh.sample(5)
set(df.columns).difference(set(dg.columns))
set(dg.columns).difference(set(df.columns))
df.columns
df[['fecha_dato', 'ncodpers', 'ind_empleado', 'pais_residencia', 'sexo',
       'age', 'fecha_alta', 'ind_nuevo', 'antiguedad', 'indrel',
       'ult_fec_cli_1t', 'indrel_1mes', 'tiprel_1mes', 'indresi', 'indext',
       'conyuemp', 'canal_entrada', 'indfall', 'tipodom', 'cod_prov',
       'nomprov', 'ind_actividad_cliente', 'renta', 'segmento']].describe(exclude = 'O').transpose()
df[['fecha_dato', 'ncodpers', 'ind_empleado', 'pais_residencia', 'sexo',
       'age', 'fecha_alta', 'ind_nuevo', 'antiguedad', 'indrel',
       'ult_fec_cli_1t', 'indrel_1mes', 'tiprel_1mes', 'indresi', 'indext',
       'conyuemp', 'canal_entrada', 'indfall', 'tipodom', 'cod_prov',
       'nomprov', 'ind_actividad_cliente', 'renta', 'segmento']].describe(include = 'O').transpose()
dg[['fecha_dato', 'ncodpers', 'ind_empleado', 'pais_residencia', 'sexo',
       'age', 'fecha_alta', 'ind_nuevo', 'antiguedad', 'indrel',
       'ult_fec_cli_1t', 'indrel_1mes', 'tiprel_1mes', 'indresi', 'indext',
       'conyuemp', 'canal_entrada', 'indfall', 'tipodom', 'cod_prov',
       'nomprov', 'ind_actividad_cliente', 'renta', 'segmento']].describe(include = 'O').transpose()
df[['fecha_dato', 'ncodpers', 'ind_empleado', 'pais_residencia', 'sexo',
       'age', 'fecha_alta', 'ind_nuevo', 'antiguedad', 'indrel',
       'ult_fec_cli_1t', 'indrel_1mes', 'tiprel_1mes', 'indresi', 'indext',
       'conyuemp', 'canal_entrada', 'indfall', 'tipodom', 'cod_prov',
       'nomprov', 'ind_actividad_cliente', 'renta', 'segmento']].hist()
df.fecha_dato  = pd.to_datetime(df.fecha_dato, format = '%Y-%m-%d')
dg.fecha_dato  = pd.to_datetime(dg.fecha_dato, format = '%Y-%m-%d')

df.fecha_alta  = pd.to_datetime(df.fecha_alta, format = '%Y-%m-%d')
dg.fecha_alta  = pd.to_datetime(dg.fecha_alta, format = '%Y-%m-%d')

df.ult_fec_cli_1t  = pd.to_datetime(df.ult_fec_cli_1t, format = '%Y-%m-%d')
dg.ult_fec_cli_1t  = pd.to_datetime(dg.ult_fec_cli_1t, format = '%Y-%m-%d')
df.fecha_dato.hist()
dg.fecha_dato.hist()
