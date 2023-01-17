import seaborn as sns

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime as dt

%matplotlib inline



trainPath = "/kaggle/input/trainparadatos/train.csv"
df = pd.read_csv(trainPath)
#Vemos cuales son los valores mas altos de metros cuadrados cubiertos y notamos que no sobrepasan los 500

df.sort_values(by = 'metroscubiertos',ascending = False)['metroscubiertos'].head()
#Eliminamos las filas con algun valor vacio. Si no sabemos el tipo de propiedad, el precio o

#la cantidad de metros cubiertos, el registro no sirve para el analisis que estamos haciendo.

#En caso de querer saber cuantos registros vienen con vacios, realizaremos el analisis en otro archivo.

df['tipodepropiedad'].dropna(inplace = True)

df['metroscubiertos'].dropna(inplace = True)

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



#nombre_de_grupos = ['<50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350-400', '400-450']

#separaciones = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]

separaciones = [0, 100, 200, 300, 400, 500]

nombre_de_grupos = ['<100', '100-200', '200-300', '300-400', '400-500']



df['metroscubiertos_en_grupos'] = pd.cut(df['metroscubiertos'], separaciones, labels=nombre_de_grupos)



df.metroscubiertos_en_grupos
#tipo de prop (filtrado por metros cubiertos) en boxplot con el precio



plt.figure(figsize = (40, 20))

sns.set(font_scale = 4)



g_4 = sns.boxplot(data = df, x = 'categorias_de_tipo_de_propiedad', y = 'precio', hue = 'metroscubiertos_en_grupos', 

                  showfliers = False)



#g_4.set_xticklabels(g_4.get_xticklabels(), rotation = 45)





plt.title('Distribucion de precios en funcion de metros cubiertos por categoria de propiedad')

plt.xlabel('Categorias de propiedad')

plt.ylabel('Precio')



plt.ticklabel_format(style='plain', axis='y')