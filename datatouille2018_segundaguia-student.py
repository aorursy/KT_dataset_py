# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_por = pd.read_csv('../input/student-por.csv', low_memory=False)
df_mat = pd.read_csv('../input/student-mat.csv', low_memory=False)
df = pd.concat([df_por,df_mat])
df.sample(3)
df.info()
# La mayoría de las familias tienen hasta 3 miembros
df['famsize'].value_counts()
# Los residentes de áreas rurales registran mas ausencias, lo que los lleva a tener peores calificaciones
df['G_mean'] = df[['G1','G2','G3']].mean(axis=1)
df2 = df[['address','absences','G_mean']]
x = df2.groupby('address')
x.mean()
#  Existe una dependencia entre las notas obtenidas y la cantidad de aplazos previos 
df3 = df[['failures','G_mean']]
df3.groupby('failures').mean().plot()
#  Las aspiraciones a futuro o plan de carrera no influyen en el nivel de consumo de alcohol 
df['alc_mean'] = df[['Walc','Dalc']].mean(axis=1)
df3 = df[['higher','alc_mean']]
df3.groupby('higher').mean()
#  Realizando un histograma de notas finales, se puede observar que existen mas notas entre 12 y 14 que entre 10 y 12. 
g = df['G3'].plot.hist()
#  Hay mas registros de alumnos de Matemática que de Portugués. 
len(df_mat), len(df_por)
#  Los alumnos de la escuela Gabriel Pereira tienen mejor desempeño que los de la escuela Mousinho da Silveira, tanto para Matemática como para Portugués. 
df_por['G_mean'] = df_por[['G1','G2','G3']].mean(axis=1)
df_mat['G_mean'] = df_mat[['G1','G2','G3']].mean(axis=1)
display(df_por[['school','G_mean']].groupby('school').mean(), df_mat[['school','G_mean']].groupby('school').mean())

