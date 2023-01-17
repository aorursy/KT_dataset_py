#Librerias de analisis de datos
import pandas as pd
#libreria para operaciones matematicas
import numpy as np

#Libreria de graficación
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline
#lectura de datos
datos = pd.read_excel('../input/todoslosmeses.xlsx')
#Primeras 5 registros para ver estructura de la tabla
datos.head()
#Columnas de mi tabla
datos.columns
#Reenombrado de columnas
datos.columns = [u'hora', u'fecha', u'coef_disp', u'coef_abso', u'albedo', u'coef_extinc', u'cn']
# info tabla
datos.info()
#Descriptivos de la tabla
datos.describe()
# localizacion de datos
datos[datos.cn == datos.cn.max()]
# para saber datos nulos
datos.isnull().sum()
# tratamiento de nulos, que hacer y que significan
# rellenando nulos
nulos = pd.DataFrame(datos.isnull().sum())
nulos.columns = ['nul_values']
nulos['porcentaje'] = nulos.nul_values/datos.shape[0] * 100
nulos
# quitar nulos
datos.dropna(inplace=True)
#Chequeo de borrado de nulos
datos.isnull().sum()
# quitar nulos
datos.dropna(inplace=True)
# datos rellenar nulos
datos.fillna(datos.median(), inplace=True)
# distribuciones
datos.cn.hist(bins=100)
plt.xlim(-0.2,7)
#Histogramas de datos
datos.hist(bins=100, figsize=(15,10)) 
plt.tight_layout()
# para checar corelaciones
datos.corr()
# mapa de calor de correlacion
plt.imshow(datos.corr(), cmap='RdYlBu')
plt.colorbar()
plt.xticks(np.arange(5), datos.describe().columns, fontsize=8)
plt.yticks(np.arange(5), datos.describe().columns, fontsize=12)
import seaborn as sns
sns.heatmap(datos.corr(), annot= True, cmap='RdYlBu')
