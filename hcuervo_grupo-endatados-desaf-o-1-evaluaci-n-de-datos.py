# Import de Librerias

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 
import random 
import os 
import pandas as pd 

# Print de los nombres de archivos del dataset
print(os.listdir('../input'))
filename = '../input/app_alumno.csv'

# app_alumno.csv has 1835710 rows in reality, but we are only loading/previewing the first 3000 rows
n = sum(1 for line in open(filename)) - 1 #number of records in file (excludes header)
s = 3000 #desired sample size
skip = sorted(random.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
alumnos = pd.read_csv('../input/app_alumno.csv', delimiter=',',skiprows=skip)

# asignar la columna 'id' al índice
alumnos.set_index('id')

nRow, nCol = alumnos.shape
print(f'There are {nRow} rows and {nCol} columns')

# imprimir los nombres de columnas para posterior uso
print('Columns in dataframe:\n',alumnos.columns)
# Añadir información de 'escuela'
escuela = pd.read_csv('../input/app_escuela.csv').set_index('id')
alumnos = alumnos.join(escuela,on='escuela_id',rsuffix = '_esc')
# Añadir información de 'departamento'
depto = pd.read_csv('../input/app_departamento.csv').set_index('id')
alumnos = alumnos.join(depto,on='departamento_id',rsuffix = '_dep')
# Añadir información de 'provincia'
prov = pd.read_csv('../input/app_provincia.csv').set_index('id')
alumnos = alumnos.join(prov,on='provincia_id',rsuffix = '_prov')
# Añadir información de 'población'
pob = pd.read_csv('../input/app_poblacion.csv').set_index('id')
alumnos = alumnos.join(pob.groupby('provincia_id').agg('sum')['total'],on='provincia_id',rsuffix = '_pob')
# Seleccionar columnas de injerencia
trainer_mate = alumnos[['sexo', 'indice_socioeconomico', 'nivel_desemp_matematica', 'tiene_notebook',
                         'tiene_pc','tiene_tablet', 'tiene_celular', 'tiene_smartphone', 'tiene_consola',
                         'tiene_smarttv', 'tiene_cable', 'tiene_internet', 'repeticion_primaria',
                         'repeticion_secundaria', 'escuela_id', 'nivel_id', 'year_id', 'ambito',
                        'gestion', 'icse', 'icse_cat', 'indice_socioeconomico_medio',
                        'indice_socioeconomico_medio_cat', 'tecnica', 'tiene_internet_esc',
                        'subvencion', 'dependencia', 'departamento_id','name', 'provincia_id', 'name_prov',
                        'tasa_con_nbi',  'total']]
# Crear training set pada 'matemáticas'
trainer_mate.nivel_desemp_matematica = trainer_mate.nivel_desemp_matematica.fillna('nc')
trainer_mate.tiene_internet_esc = trainer_mate.tiene_internet_esc.fillna('nc')
trainer_mate.indice_socioeconomico_medio_cat = trainer_mate.indice_socioeconomico_medio_cat.fillna('nc')
trainer_mate.indice_socioeconomico = trainer_mate.indice_socioeconomico.fillna('nc')
trainer_mate.icse_cat = trainer_mate.icse_cat.fillna('nc')
trainer_mate.subvencion = trainer_mate.subvencion.fillna('Sin información')
trainer_mate.dependencia = trainer_mate.dependencia.fillna('Sin información')

trainer_mate.tasa_con_nbi = trainer_mate.tasa_con_nbi.fillna(0) 
trainer_mate.departamento_id = trainer_mate.departamento_id.fillna(-1) 

trainer_mate = trainer_mate[trainer_mate.nivel_desemp_matematica != 'nc' ]

print('Contenido del registro número 1000:\n',trainer_mate.iloc[1000:1001])
# Asignar valores del 0 al 4 en 'nive_desemp_*'
desemp_values = ['por_debajo_del_basico', 'basico','satisfactorio', 'avanzado']
trainer_mate.nivel_desemp_matematica = trainer_mate.nivel_desemp_matematica.apply(lambda x: desemp_values.index(x))
# Num Encode variables más importantes
valores=list(['nc', 'no', 'si']) # otra forma sin ordenar sería valores = list(trainer_mate['xxxx'].unique())
trainer_mate.tiene_notebook = trainer_mate.tiene_notebook.apply(lambda x: valores.index(x))
trainer_mate.tiene_pc = trainer_mate.tiene_pc.apply(lambda x: valores.index(x))
trainer_mate.tiene_tablet = trainer_mate.tiene_tablet.apply(lambda x: valores.index(x))
trainer_mate.tiene_celular = trainer_mate.tiene_celular.apply(lambda x: valores.index(x))
trainer_mate.tiene_smartphone = trainer_mate.tiene_smartphone.apply(lambda x: valores.index(x))
trainer_mate.tiene_consola = trainer_mate.tiene_consola.apply(lambda x: valores.index(x))
trainer_mate.tiene_smarttv = trainer_mate.tiene_smarttv.apply(lambda x: valores.index(x))
trainer_mate.tiene_cable = trainer_mate.tiene_cable.apply(lambda x: valores.index(x))
trainer_mate.tiene_internet = trainer_mate.tiene_internet.apply(lambda x: valores.index(x))

trainer_mate.to_csv('trainer_mate.csv',index=False)
