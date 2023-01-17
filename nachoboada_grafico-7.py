import seaborn as sns

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime as dt

%matplotlib inline



trainPath = "/kaggle/input/trainparadatos/train.csv"
df = pd.read_csv(trainPath)
df['tipodepropiedad'].dropna(inplace = True)

df['garages'].dropna(inplace = True)
df['garages'] = (df['garages']).astype(int, errors = 'ignore')
#Agrupo los tipos de propiedad en categorias



categorias = {



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
df['categoria'] = df['tipodepropiedad'].map(categorias)
#tipo de prop (filtrado por garages) en violinplot con el precio



plt.figure(figsize = (40, 20))

sns.set(font_scale = 4)



g_7 = sns.violinplot(data = df, x = 'categoria', y = 'precio', hue = 'garages', showfliers = False, scale = 'count', inner = 'quartile',

                    hue_scale = False)



plt.title('Distribucion de precios en funcion de garages por categoria de propiedad')

plt.xlabel('categorias de propiedad')

plt.ylabel('precio')



plt.ticklabel_format(style='plain', axis='y')
#tipo de prop (filtrado por garages) en boxplot con el precio



plt.figure(figsize = (40, 20))

sns.set(font_scale = 4)



g_7 = sns.boxplot(data = df, x = 'categoria', y = 'precio', hue = 'garages', showfliers = False)



plt.title('Distribucion de precios en funcion de garages por categoria de propiedad')

plt.xlabel('categorias de propiedad')

plt.ylabel('precio')



plt.ticklabel_format(style='plain', axis='y')