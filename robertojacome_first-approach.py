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
#this is the name of the file

data = pd.read_csv("../input/2016_Accidentes Trnsito_BDD.csv")

#Which is the province with more accidents?
step = 0.5
bin_range = np.arange(0.5, 24.5, 1)
out, bins  = pd.cut(data['PROVINCIA'], bins=bin_range, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar()


# get the data of the Guayas province, code 9
guayas = data.loc[data['PROVINCIA'] == 9]

#list of codes of all provinces
#1=AZUAY
#2=BOLIVAR
#3=CAÑAR
#4=CARCHI
#5=COTOPAXI
#6=CHIMBORAZO
#7=EL ORO
#8=ESMERALDAS
#9=GUAYAS
#10=IMBABURA
#11=LOJA
#12=LOS RIOS
#13=MANABI
#14=MORONA SANTIAGO
#15=NAPO
#16=PASTAZA
#17=PICHINCHA
#18=TUNGURAHUA
#19=ZAMORA CHINCHIPE
#20=GALAPAGOS
#21=SUCUMBIOS
#22=ORELLANA
#23=SANTO DOMINGO DE LOS TSACHILAS
#24=SANTA ELENA
print(guayas)
#Let's see the causes of the accidents
step = 0.5
bin_range = np.arange(0.5, 10.5, 1)
out, bins  = pd.cut(guayas['CAUSA'], bins=bin_range, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar()

#1= EMBRIAGUEZ O DROGA
#2= MAL REBASAMIENTO INVADIR CARRIL
#3= EXCESO VELOCIDAD
#4= IMPERICIA E IMPRUDDENCIA DEL CONDUCTOR
#5= IMPRUDENCIA  DEL PEATÓN
#6= DAÑOS MECÁNICOS
#7= NO RESPETA LAS SEÑALES DE TRÁNSITO
#8= FACTORES CLIMÁTICOS
#9= MAL ESTADO DE LA VÍA
#10= OTRAS CAUSAS
#Now the type of accidents
step = 0.5
bin_range = np.arange(0.5, 8.5, 1)
out, bins  = pd.cut(guayas['CLASE'], bins=bin_range, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar()

#1= ATROPELLOS
#2= CAÍDA PASAJEROS
#3= CHOQUES
#4= ESTRELLAMIENTOS
#5= ROZAMIENTOS
#6= VOLCAMIENTOS
#7= PÉRDIDA DE PISTA
#8= OTROS
#Finally, the number of victims
step = 0.5
bin_range = np.arange(0.5, 10.5, 1)
out, bins  = pd.cut(guayas['TOTAL_VICTIMAS'], bins=bin_range, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar()
#What time of the day
step = 0.5
bin_range = np.arange(-0.5, 24.5, 1)
out, bins  = pd.cut(guayas['HORA'], bins=bin_range, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar()


#0= 00:00 A 00:59
#1= 01:00 A 01:59
#2= 02:00 A 02:59
#3= 03:00 A 03:59
#4= 04:00 A 04:59
#5= 05:00 A 05:59
#6= 06:00 A 06:59
#7= 07:00 A 07:59
#8= 08:00 A 08:59#
#9= 09:00 A 09:5#9
#10= 10:00 A 10:#59
#11= 11:00 A 11:#59
#12= 12:00 A 12:#59
#13= 13:00 A 13:#59
#14= 14:00 A 14:#59
#15= 15:00 A 15:59
#16= 16:00 A 16:59
#17= 17:00 A 17:59
#18= 18:00 A 18:59
#19= 19:00 A 19:59
#20= 20:00 A 20:59
#21= 21:00 A 21:59
#22= 22:00 A 22:59
#23= 23:00 A 23:59
