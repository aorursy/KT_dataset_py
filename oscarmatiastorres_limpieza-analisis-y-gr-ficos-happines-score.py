import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt
df2015 = pd.read_csv('../input/world-happiness/2015.csv')

df2016 = pd.read_csv('../input/world-happiness/2016.csv')
df2015.head()
print(df2015.columns)



print(df2015.shape)
print(df2016.columns)



print(df2016.shape)
"""solo hay 11 Columnas en comun  ('Country', 'Region', 'Happiness Rank', 'Happiness Score',

'Economy (GDP per Capita)','Family', 'Health (Life Expectancy)', 'Freedom', 'Trust (Government 

Corruption)','Generosity', 'Dystopia Residual' """
uniondf = pd.concat([df2016, df2015], ignore_index=True, join='outer')

uniondf.head(5)
print("CANTIDAD de datos nulos por columna en el dataframe") 

print(uniondf.isnull().sum())

print("----------------------------------")

print("PORCENTAJE de datos nulos por columna en el dataframe") 

print(uniondf.isnull().sum()/len(uniondf)*100)

 



#Maximo de Lower Confidence Interval

print(uniondf.iloc[:, 4].max())



#Minimo de Lower Confidence Interval

print(uniondf.iloc[:, 4].min())
#Maximo de Upper Confidence Interval

print(uniondf.iloc[:, 5].max())
#Minimo de Upper Confidence Interval

print(uniondf.iloc[:, 5].min())
## obtener un numero aleatorio para ambos intervalos

import random
import math

def truncate(number, digits) -> float:

    stepper = 10.0 ** digits

    return math.trunc(stepper * number) / stepper



#obtener float aleatorio para Lower Confidence

random.uniform(2.7319999999999998, 7.46)

#valor aleatorio para Lower Confidence

aleatorioLower = truncate(6.167839783480675, 2)





#obtener float aleatorio para Upper Confidence

random.uniform(3.0780000000000003, 7.669)

#valor aleatorio para reemplazar en Upper Confidence

aleatorioUpper = truncate(3.4782926731390473, 2)
#Reemplazar valores nulos por valores obtenidos entre los rangos Max y Min

uniondf['Lower Confidence Interval'] = uniondf['Lower Confidence Interval'].fillna(aleatorioLower)

uniondf['Upper Confidence Interval'] = uniondf['Upper Confidence Interval'].fillna(aleatorioUpper)
uniondf.isnull().sum()
# obtener media al cuadrado de "Standar Error"



union_df_sin_nulos = uniondf["Standard Error"].dropna()

udfsn = union_df_sin_nulos.mean()**2

udfsn
uniondf2 = uniondf.fillna(udfsn)
uniondf2.describe




uniondf2.plot(kind='scatter', x='Family', color = "g", y='Health (Life Expectancy)')

plt.title('Relación entre Familia y Salud')



sns.lmplot(x="Family", y = 'Health (Life Expectancy)', data = uniondf2)



#par= sns.pairplot(faithful)
uniondf2.plot(kind='scatter', x='Happiness Score', color = "b", y='Trust (Government Corruption)')

plt.title('Relación entre Score de felicidad y Confianza en el gobierno')



sns.lmplot(x="Happiness Score", y = 'Trust (Government Corruption)', data = uniondf2)
"""" Segun la grafica expuesta, no existiría una fuerte relacion estadistica entre ambas variables, existiendo una gran 

cantidad de paises con bajos indices de confianza en su gobierno pero altos score en felicidad.""" 
sns.heatmap(uniondf2.corr(), cmap='Oranges')
uniondf3 = uniondf2.sort_values(['Country', 'Happiness Score',])

uniondf3.head()
sns.violinplot(x='Region', y='Dystopia Residual', data=uniondf2)

plt.xticks(rotation=90)  

plt.ylabel('Valor Distopia Residual') 

plt.xlabel('Region')

plt.title('Distribución del grado de distopia por región')

plt.show()