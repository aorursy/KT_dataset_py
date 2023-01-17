import seaborn as sns

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime as dt

%matplotlib inline
df = pd.read_csv("/kaggle/input/trainparadatos/train.csv")
#Eliminamos las filas con algun valor vacio. Si no sabemos el tipo de propiedad, el precio o

#la cantidad de metros cubiertos, el registro no sirve para el analisis que estamos haciendo.

#En caso de querer saber cuantos registros vienen con vacios, realizaremos el analisis en otro archivo.

df['tipodepropiedad'].dropna(inplace = True)

df['centroscomercialescercanos'].dropna(inplace = True)

df['precio'].dropna(inplace = True)
def cambiarLabelCentrosComercialesCercanos(x):

    

    if (x == 0):

        return 'sin centro comercial cerca'

    if (x == 1):

        return 'con centro comercial cerca'
df['centroscomercialescercanos'] = df['centroscomercialescercanos'].map(lambda x : cambiarLabelCentrosComercialesCercanos(x))
#Agrupo los tipos de propiedad en categorias



categorias_de_tipo_de_propiedad = {



'Apartamento':'residencial',

'Casa en condominio':'residencial',

'Casa':'residencial',

'Quinta Vacacional':'residencial',

'Casa uso de suelo':'residencial',

'Villa':'residencial',

'Duplex':'residencial',

'Departamento Compartido':'residencial',

    

'Terreno comercial':'comercial',

'Local Comercial':'comercial',

'Oficina comercial':'comercial',

'Local en centro comercial':'comercial',

'Bodega comercial':'comercial',

'Nave industrial':'comercial',

'Terreno industrial':'comercial',

'Garage':'comercial',

    

'Rancho':'rural',

'Huerta':'rural',

'Lote':'rural',

'Terreno':'rural',

    

'Edificio':'varios',

'Inmuebles productivos urbanos':'varios',

'Hospedaje':'varios',

'Otros':'varios',

    

}
#Quito los de tipo varios porque en cuanto a piscinas no son significativos



df['categorias_de_tipo_de_propiedad'] = df['tipodepropiedad'].map(categorias_de_tipo_de_propiedad)

#df = df[df['categorias_de_tipo_de_propiedad'] != 'varios']

df['centros comerciales cercanos'] = df['centroscomercialescercanos']
#tipo de prop (filtrado por centro comercial cercano) en violinplot con el precio



plt.figure(figsize = (40, 20))

sns.set(font_scale = 4)



g_14 = sns.violinplot(data = df, x = 'categorias_de_tipo_de_propiedad', y = 'precio',

                      hue = 'centros comerciales cercanos', showfliers = False,

                     scale = 'count', inner = 'quartile', hue_scale = False, split = True)



g_14.set(ylim=(0))



plt.title('Distribucion de precios en funcion de si tiene centro comercial cercano por categoria de propiedad')

plt.xlabel('categorias de propiedad')

plt.ylabel('precio')



plt.ticklabel_format(style='plain', axis='y')
#tipo de prop (filtrado por centro comercial cercano) en boxplot con el precio



plt.figure(figsize = (40, 20))

sns.set(font_scale = 4)



g_14 = sns.boxplot(data = df, x = 'categorias_de_tipo_de_propiedad', y = 'precio', hue = 'centros comerciales cercanos', showfliers = False)



plt.title('Distribucion de precios en funcion de si tiene centro comercial cercano por categoria de propiedad')

plt.xlabel('categorias de propiedad')

plt.ylabel('precio')



plt.ticklabel_format(style='plain', axis='y')