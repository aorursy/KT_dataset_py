Nombre='Cesar Guevara'
Edad=33
talla=1.73
Casado=False
Edad
tupla1=(Nombre,Edad,talla,Casado)
tupla1
type(tupla1)
n=len(tupla1)
n
lista1=[Nombre,Edad,talla,Casado]
lista1
type(lista1)
n=len(lista1)
n
tupla1
import sys
sys.getsizeof(tupla1)
lista1
sys.getsizeof(lista1)
tupla1
tupla1[0]='César Guevara Quispe'
lista1
lista1[0]='César Guevara Quispe'
lista1
lista1.append('Lima')
lista1
lista1.insert(2,"Cesar.guevaraq@gmail.com")
lista1
del lista1[3]
lista1
tupla2=tuple(lista1)
tupla2
type(tupla2)
lista2=list(tupla1)
lista2
type(lista2)
DatosPersonales={
                 "Nombre":['Cesar Guevara','Jose Honores'],#Caracteres
                 "Edad":[33,35],#Enteros
                 "Talla":[1.73,1.72],#Flotantes
                 "Ciudad":['Lima','Arequipa']#Caracteres
                 }
DatosPersonales
DatosPersonales['Nombre']
import pandas as pd
tabla1 = pd.DataFrame(DatosPersonales)
tabla1
tabla1.columns
registro=pd.Series(['Diego Rojas',29,1.75,'Lima'],index=tabla1.columns)
tabla1=tabla1.append(registro,ignore_index=True) 
tabla1
tabla1.info()
import os
os.listdir("../input")
import pandas as pd
datostxt=pd.read_table('../input/data.txt',encoding="iso-8859-1")
datostxt.head(10)
datostxt.info()
len(datostxt.index)
import pandas as pd
datoscsv = pd.read_csv('../input/Datos_Clientes.csv',sep=";",encoding="iso-8859-1")
datoscsv.head(20)
import pandas as pd
file = pd.ExcelFile('../input/PD1-2018.xlsx',sheetname='Hoja1')
file
datosexcel=file.parse()
datosexcel.head(20)
datosexcel.info()
x=datoscsv['Venta']
x
import numpy as np
np.mean(x)
import numpy as np
np.median(x)
colores=["red", "blue", "blue", "red", "green", "red", "red"]
from collections import Counter
Counter(colores)
from statistics import mode
mode(colores)
x=datoscsv['Venta']
max(x)
min(x)
Rango=max(x)-min(x)
Rango
import numpy as np
np.std(x)
import numpy as np
np.std(x)/np.mean(x)*100
x=datoscsv['Venta']
from scipy.stats import skew 
skew(x)
from scipy.stats import kurtosis 
kurtosis(x)
datoscsv.describe()
Resumen1=datoscsv.groupby(['Segmento'])['Venta'].describe()
Resumen1
Resumen2=datoscsv.groupby(['Segmento'])['Venta'].agg([np.mean])
Resumen2
import numpy as np
Resumen2=datoscsv.groupby(['Segmento'])['Venta'].agg([np.mean,np.median,np.std])
Resumen2
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
plt.subplots(figsize=(10,6))
sns.kdeplot(np.log(x),color='blue', shade = True)
Segmentos=datoscsv['Segmento'].unique()
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
plt.subplots(figsize=(10,6))
sns.kdeplot(datoscsv['Venta'][(datoscsv['Segmento'] =='Bodega')],color="Blue", shade= True)
sns.kdeplot(datoscsv['Venta'][(datoscsv['Segmento'] =='Despensa')],color="Red", shade= True)
sns.kdeplot(datoscsv['Venta'][(datoscsv['Segmento'] =='Reposicion')],color="Green", shade= True)
datosexcel.head(10)
import pandas as pd
pd.crosstab(datosexcel['Categoría'],datosexcel['Condición'])
import pandas as pd
pd.crosstab(datosexcel['Categoría'],datosexcel['Condición'],margins=True)