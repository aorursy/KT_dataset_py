import seaborn as sns

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime as dt

%matplotlib inline

df = pd.read_csv("/kaggle/input/trainparadatos/train.csv")
df['tipodepropiedad'].dropna(inplace = True)

df['gimnasio'].dropna(inplace = True)

df['precio'].dropna(inplace = True)
def cambiarLabel(x):

    

    if (x == 0.0):

        return 'sin gimnasio'

    if (x == 1.0):

        return 'con gimnasio'



df['gimnasio'] = df['gimnasio'].map(lambda x : cambiarLabel(x))
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
#En los tipos de propiedad "varios", "comercial" y "rural" no hay gimnasios.

df['categorias_de_tipo_de_propiedad'] = df['tipodepropiedad'].map(categorias_de_tipo_de_propiedad)

df = df[df['categorias_de_tipo_de_propiedad'] != 'varios']

df = df[df['categorias_de_tipo_de_propiedad'] != 'comercial']

df = df[df['categorias_de_tipo_de_propiedad'] != 'rural']
#tipo de prop (filtrado por gimnasio) en violinplot con el precio



plt.figure(figsize = (40, 20))

sns.set(font_scale = 4)



g_9 = sns.violinplot(data = df, x = 'categorias_de_tipo_de_propiedad', y = 'precio', 

                  hue = 'gimnasio', showfliers = False, scale = 'count', inner = 'quartile',

                    hue_scale = False, split = True)





g_9.set(ylim=(0))



plt.title('Distribucion de precios en funcion de gimnasio por categoria de propiedad')

plt.xlabel('Categorias de propiedad')

plt.ylabel('Precio')



plt.ticklabel_format(style='plain', axis='y')
#tipo de prop (filtrado por gimnasio) en boxplot con el precio



plt.figure(figsize = (40, 20))

sns.set(font_scale = 4)



g_9 = sns.boxplot(data = df, x = 'categorias_de_tipo_de_propiedad', y = 'precio', 

                  hue = 'gimnasio', showfliers = False)









plt.title('Distribucion de precios en funcion de gimnasio por categoria de propiedad')

plt.xlabel('Categorias de propiedad')

plt.ylabel('Precio')



plt.ticklabel_format(style='plain', axis='y')