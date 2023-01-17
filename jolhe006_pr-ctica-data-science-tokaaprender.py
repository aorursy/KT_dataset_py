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
db_escuelas = pd.read_csv("../input/bd_limpia.csv")
db_escuelas.describe()
db_escuelas.head()
total_hombres = db_escuelas['TOTAL HOMBRES'].sum()
total_hombres
total_mujeres = db_escuelas['TOTAL MUJERES'].sum()
total_mujeres
total_matricula = db_escuelas['TOTAL MATRÍCULA'].sum()
total_matricula
(total_hombres+total_mujeres) == total_matricula
estudiantes = pd.DataFrame(data=
    {
        'Total':[total_hombres, total_mujeres] 
    }, index=["Hombres", "Mujeres"])
estudiantes
estudiantes.plot.pie(subplots=True, figsize=(10,10), autopct='%1.1f%%')
db_escuelas.SOSTENIMIENTO.unique()
sostenimiento = db_escuelas.groupby('SOSTENIMIENTO').SOSTENIMIENTO.count()
type(sostenimiento)
sostenimiento
import matplotlib.cm as cm 
rb_colors = cm.rainbow(np.linspace(0, 1, sostenimiento.count()))
sostenimiento.plot.bar(figsize = (15, 5), title='Sostenimiento en las escuelas de Jalisco', colors=rb_colors)
db_escuelas.NIVEL.unique()
niveles_educativos = db_escuelas.groupby('NIVEL').NIVEL.count()
niveles_educativos.head()
ax.patches[1].get_y()
ax = niveles_educativos.plot.bar(rot=1, figsize = (15, 5), colors=cm.rainbow(np.linspace(0, 1, niveles_educativos.count())), title='Planteles de Jalisco por nivel educativo', logy = True)

# set individual bar lables using above list
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x()+.1, i.get_height(), str(round((i.get_height()), 2)), fontsize=14, color='black')
print('Máximo de estudiantes: ' + str(db_escuelas['TOTAL MATRÍCULA'].max()))
escuelas_sorted = db_escuelas.sort_values(by=['TOTAL MATRÍCULA'], ascending=False).reset_index(drop=True)
escuelas_sorted.head()
escuelas_sorted = escuelas_sorted.loc[:9, ['NOMBRE ESCUELA', 'TOTAL HOMBRES', 'TOTAL MUJERES', 'TOTAL MATRÍCULA', 'DOCENTES TECNOLOGÍAS']]
escuelas_top_index = escuelas_sorted.set_index('NOMBRE ESCUELA')
ax = escuelas_top_index.plot.barh( color=cm.rainbow(np.linspace(0, 1, len(escuelas_top_index.columns))),figsize = (25, 25), title='Top planteles con más estudiantes', logx= True)
# set individual bar lables using above list
for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width() + 1.5, i.get_y()+.05, str(round((i.get_width()), 2)), fontsize=11, color='dimgrey')