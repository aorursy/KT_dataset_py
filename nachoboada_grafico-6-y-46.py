import seaborn as sns

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime as dt

%matplotlib inline



trainPath = "/kaggle/input/trainparadatos/train.csv"
df = pd.read_csv(trainPath)

df.replace([np.inf, -np.inf], np.nan)

df['tipodepropiedad'].dropna(inplace = True)

df['habitaciones'].dropna(inplace = True)

df['precio'].dropna(inplace = True)
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
df['categorias_de_tipo_de_propiedad'] = df['tipodepropiedad'].map(categorias_de_tipo_de_propiedad)
#Agrupo por metros cuadrados cubiertos



separaciones = [0, 2, 4, 6, 8, 10]

nombre_de_grupos = ['<2', '2-4', '4-6', '6-8', '8-10']



df['Cantidad de Habitaciones'] = pd.cut(df['habitaciones'], separaciones, labels=nombre_de_grupos)



df['Cantidad de Habitaciones']
#tipo de prop (filtrado por habitaciones) en violinplot con el precio



plt.figure(figsize = (40, 20))

sns.set(font_scale = 4)



g_6 = sns.violinplot(data = df, x = 'categorias_de_tipo_de_propiedad', y = 'precio', 

                  hue = 'Cantidad de Habitaciones', showfliers = False, scale = 'count', inner = 'quartile',

                    hue_scale = False)



g_6.set(ylim=(0))









plt.title('Distribucion de precios en funcion de habitaciones por categoria de propiedad')

plt.xlabel('Categorias de propiedad')

plt.ylabel('Precio')



plt.ticklabel_format(style='plain', axis='y')
#tipo de prop (filtrado por habitaciones) en boxplot con el precio



plt.figure(figsize = (40, 20))

sns.set(font_scale = 4)



g_6 = sns.boxplot(data = df, x = 'categorias_de_tipo_de_propiedad', y = 'precio', 

                  hue = 'Cantidad de Habitaciones', showfliers = False)









plt.title('Distribucion de precios en funcion de habitaciones por categoria de propiedad')

plt.xlabel('Categorias de propiedad')

plt.ylabel('Precio')



plt.ticklabel_format(style='plain', axis='y')