import seaborn as sns

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime as dt

%matplotlib inline
df = pd.read_csv("/kaggle/input/trainparadatos/train.csv")
df['tipodepropiedad'].dropna(inplace = True)

df['usosmultiples'].dropna(inplace = True)

df['precio'].dropna(inplace = True)
#Agrupo los tipos de propiedad en categorias

#Quito los de tipo varios y rural porque en cuanto a usos multiples no son significativos



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



    

}
df['categorias_de_tipo_de_propiedad'] = df['tipodepropiedad'].map(categorias_de_tipo_de_propiedad)
def cambiarLabel(x):

    if (x == 0.0):

        return 'sin usos multiples'

    if (x == 1.0):

        return 'con usos multiples'



df['usos multiples'] = df['usosmultiples'].map(lambda x : cambiarLabel(x))
#tipo de prop (filtrado por usos multiples) en violinplot con el precio



plt.figure(figsize = (40, 20))

sns.set(font_scale = 4)



g_10 = sns.violinplot(data = df, x = 'categorias_de_tipo_de_propiedad', y = 'precio', 

                  hue = 'usos multiples', showfliers = False, scale = 'count', inner = 'quartile',

                    hue_scale = False, split = True)



g_10.set(ylim=(0))





plt.title('Distribucion de precios en funcion de usos multiples por categoria de propiedad')

plt.xlabel('Categorias de propiedad')

plt.ylabel('Precio')



plt.ticklabel_format(style='plain', axis='y')
#tipo de prop (filtrado por usos multiples) en boxplot con el precio



plt.figure(figsize = (40, 20))

sns.set(font_scale = 4)



g_10 = sns.boxplot(data = df, x = 'categorias_de_tipo_de_propiedad', y = 'precio', 

                  hue = 'usos multiples', showfliers = False)









plt.title('Distribucion de precios en funcion de usos multiples por categoria de propiedad')

plt.xlabel('Categorias de propiedad')

plt.ylabel('Precio')



plt.ticklabel_format(style='plain', axis='y')