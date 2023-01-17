import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import pandas as pd

#import numpy as np



MAX_ROWS = 10

pd.set_option('display.max_rows', MAX_ROWS)

pd.set_option('display.max_columns', 200)

 

sns.set_style("whitegrid")

sns.set_context("paper")



plt.rcParams['figure.figsize'] = (12,5)
path_dataset = '../input/datos_properati_limpios.csv'

df = pd.read_csv(path_dataset, parse_dates=['created_on'])
df.columns
print("El dataset que vamos a trabajar aquí tiene {} observaciones".format(df.shape[0]))
# Mostrá la figura en esta celda

df.dropna(subset = ['price_usd_per_m2'], inplace = True)

sns.distplot(df['price_usd_per_m2'])

df.shape
# El boxplot debe estar en esta celda

sns.boxplot(x = df['price_usd_per_m2'])
# Describir la columna en esta celda

df.price_usd_per_m2.describe()
# Realizar el filtrado intercuartílico en esta celda

df_filtered = df.copy()

Q1 = df_filtered['price_usd_per_m2'].quantile(0.25)

Q3 = df_filtered['price_usd_per_m2'].quantile(0.75)

IQR = Q3 - Q1

LR = Q1 -(1.5 * IQR)

UR = Q3 + (1.5 * IQR)



print ('el valor Minimo es '+ str(LR))

print ('el valor Maximo es '+ str(UR))





df_filtered.drop(df_filtered[(df_filtered.price_usd_per_m2 > UR) | (df_filtered.price_usd_per_m2< LR)].index, inplace=True)

df_filtered

#Otra forma de hacerlo 



q1_price_usd_per_m2=df.price_usd_per_m2.quantile(0.25)

q3_price_usd_per_m2=df.price_usd_per_m2.quantile(0.75)

iqr = df.price_usd_per_m2.quantile(0.75) - df.price_usd_per_m2.quantile(0.25)

min = q1_price_usd_per_m2 - (iqr*1.5)

max = q3_price_usd_per_m2 + (iqr*1.5)

df_filtered = df[df.price_usd_per_m2.between(min,max)]



df_filtered = df

Q1 = df_filtered['price_usd_per_m2'].quantile(0.25)

Q3 = df_filtered['price_usd_per_m2'].quantile(0.75)

IQR = Q3 - Q1

df_filtered = df.query('(@Q1 - 1.5 * @IQR) <= price_usd_per_m2 <= (@Q3 + 1.5 * @IQR)')

df_filtered

# Hacé el distplot 

sns.distplot(df_filtered['price_usd_per_m2'])
# Hacé el boxplot en esta celda

sns.boxplot(x = df_filtered['price_usd_per_m2'])
df_filtered['price_usd_per_m2'].describe()
df = df_filtered
df.surface_total_in_m2.isna().mean().round(4) # hacer el resto y concatenar
# Mostrá los valores faltantes en esta celda

por_surface_total_in_m2 = ((df.surface_total_in_m2.isna().sum()) / df.shape[0]) * 100

por_surface_covered_in_m2 = ((df.surface_covered_in_m2.isna().sum()) / df.shape[0]) * 100

por_rooms = ((df.rooms.isna().sum()) / df.shape[0] ) * 100

por_price_aprox_usd =((df.price_aprox_usd.isna().sum()) / df.shape[0]) * 100

por_price_usd_per_m2 = ((df.price_usd_per_m2.isna().sum()) / df.shape[0]) * 100

por_expenses = ((df.expenses.isna().sum()) / df.shape[0]) * 100





print('% Sup. Total NULOS = {a}\n% Sup. Cubierta NULOS = {b}\n% Ambientes NULOS = {c}\n% Precio aprox U$D NULOS = {d}\n% Precio aprox U$D/M2 NULOS = {e}\n% Expensas NULOS = {f}' .format(a=por_surface_total_in_m2, b=por_surface_covered_in_m2, c=por_rooms, d=por_price_aprox_usd, e=por_price_usd_per_m2, f=por_expenses))
#Otra forma mas directa de hacerlo es de la siguiente manera

df.isna().mean().round(4) * 100
#Otra forma mas directa de hacerlo es de la siguiente manera

df.isna().mean().sort_values(ascending=False)*100
df = df.drop(['floor', 'expenses'], axis = 1)
# Imputar los valores en esta celda



import numpy as np

from sklearn.impute import SimpleImputer



imp_mean = SimpleImputer(strategy='mean')



imp_mean = imp_mean.fit(df[['surface_total_in_m2']])

df[['surface_total_in_m2']] = imp_mean.transform(df[['surface_total_in_m2']])

df[['surface_covered_in_m2']] = imp_mean.transform(df[['surface_covered_in_m2']])







#Chequeamos que no haya valores nulos. para corroborar que el proceso funciono

df.loc[:,['surface_total_in_m2','surface_covered_in_m2']].isna().sum()

#Otra forma 



imp_mean = SimpleImputer(strategy='mean', missing_values=np.nan)

df[['surface_total_in_m2','surface_covered_in_m2']] = imp_mean.fit_transform(df[['surface_total_in_m2','surface_covered_in_m2']])
# Imputar con la mediana en esta celda

imp_median = SimpleImputer(strategy='median')



imp_median = imp_median.fit(df[['rooms']])

df[['rooms']] = imp_median.transform(df[['rooms']])



df.isna().sum() # chequer que no queden nulos
# Utilizá LabelEncoder en esta celda



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(df["property_type"])



le.transform(df["property_type"]) #ahora las variables caregoricas toman un indicador numerico
# Mostrar la propiedad classes_ del LabelEncoder que creaste

list(le.classes_)
# Utilizá OneHotEncoder en esta celda 



df_encoded = le.transform(df["property_type"]) #Primero asignamos una variable el codigo realizado anteriormente



from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder(sparse=False)



df_encoded = df_encoded.reshape(len(df_encoded), 1)



categoricals_df = onehot_encoder.fit_transform(df_encoded)



categoricals_df = pd.DataFrame (categoricals_df, dtype = int, columns = le.classes_) #lo paso a entero para evitar futuros problemas en el modelado y uso label encoder para pasar el nombre de la columna



categoricals_df





categoricals_df = categoricals_df.set_index(df.index)

df = pd.concat([df, categoricals_df], axis=1)

df.head()
def custom_division(x, y):

    if y > 0:

        res = x / y

    else:

        res = 0

    return res



df['price_m2'] = df.apply(lambda x: custom_division(x['price_aprox_usd'], x['surface_total_in_m2']), axis = 1)

df.drop(['price_usd_per_m2'], axis=1, inplace=True)
# Creamos un dataset con los porcentajes de nulos

df_faltantes = pd.DataFrame(df.isnull().sum() / df.shape[0], columns=['Porcentaje nulos'])

# Solo mostramos los que tengan valores nulos. Si el porcentaje es 0 no se muestra

df_faltantes.loc[~(df_faltantes==0).all(axis=1)]
print("El dataset final luego del procesamiento tiene {} observaciones".format(df.shape[0]))