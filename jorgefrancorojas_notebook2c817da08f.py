import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sales_train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
item_cats = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
submission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')

fig = plt.figure(figsize=(20,10)) # Establecer Tamaño
plt.subplots_adjust(hspace= 0.5) # Ajusta la distancia de posición de la subimagen, hspace (espaciado vertical) = 0.4

#Ver distribución de shop_id
plt.subplot2grid((1,1), (0,0))  
sales_train['shop_id'].value_counts(normalize=True).plot(kind='bar', color='orangered') # Usar histograma
plt.title('Estado de distribución de la identificación de la tienda (Figura 1)')
plt.xlabel('ID de tienda')
plt.ylabel('Apariciones normalizadas')

plt.show()
fig = plt.figure(figsize=(20,10)) # Establecer Tamaño
plt.subplots_adjust(hspace= 0.5) # Ajusta la distancia de posición de la subimagen, hspace (espaciado vertical) = 0.4

#Ver la distribución de item_price
plt.subplot2grid((1,1), (0,0))  
sales_train['item_price'].plot(kind='hist', color='darkorange')  # Usar histograma
plt.title('La distribución de los precios de los productos (Figura 3)')
plt.xlabel('precio del producto')
plt.ylabel('El número de ocurrencias')
plt.show()
fig = plt.figure(figsize=(20,10)) # Establecer Tamaño
plt.subplots_adjust(hspace= 0.5) # Ajusta la distancia de posición de la subimagen, hspace (espaciado vertical) = 0.4

#Ver la distribución de item_cnt_day
plt.subplot2grid((1,1), (0,0))  
sales_train['item_cnt_day'].plot(kind='hist', color='cornflowerblue')  # Usar histograma
plt.title('Estado de distribución de las ventas de productos (Figura 4)')
plt.xlabel('Venta de productos')
plt.ylabel('El número de ocurrencias')
plt.show()
fig = plt.figure(figsize=(20,10)) # Establecer Tamaño
plt.subplots_adjust(hspace= 0.5) # Ajusta la distancia de posición de la subimagen, hspace (espaciado vertical) = 0.4

#Ver la distribución de date_block_num
plt.subplot2grid((1,1), (0,0))  
sales_train['date_block_num'].value_counts(normalize=True).plot(kind='bar', color='darkseagreen') # Usar histograma
plt.title('El número de meses y el estado de distribución de los registros de ventas (Figura 5)')
plt.xlabel('Número de meses')
plt.ylabel('Número de ocurrencias de registros de ventas estandarizados')

plt.show()
#Se revisan los precios de los cinco productos que mas tiene valor.

sales_train['item_price'].sort_values(ascending=False)[:5]
#Se identifica el producto que mas se aporta al valor.

sales_train[sales_train['item_price'] == 307980]
#La información del producto 6066

items[items['item_id'] == 6066]
# Este producto es en español  "Radmin 3 - 522 flash" 

# Necesitamos verificar más a fondo si hay algún registro de este producto en el conjunto de datos

sales_train = sales_train[sales_train['item_price'] < 300000]
#Se revisan los productos de menor valor

sales_train['item_price'].sort_values(ascending=True)[:5]
#Hay un precio de venta negativo en el extremo muy pequeño, revisando en el conjunto de datos

sales_train[sales_train['item_price'] == -1]
#Ver la información correspondiente del producto
sales_train[sales_train['item_id'] == 2973]
#Ver la información de precio correspondiente del producto

price_info = sales_train[sales_train['item_id'] == 2973]['item_price']
price_info.describe()
#Se puede ver que el precio de este producto no es razonable y el precio de venta promedio es superior a 2000, 
#por lo que otros valores deben eliminarse o completarse.

#Teniendo en cuenta que el precio de venta del mismo producto en diferentes tiendas es diferente, 
#se debe utilizar en su lugar el precio medio del producto en la tienda N°32 correspondiente.

price_median = sales_train[(sales_train['shop_id'] == 32) & (sales_train['item_id'] == 2973) & (sales_train['date_block_num'] == 4) & (sales_train['item_price'] > 0)].item_price.median()
sales_train.loc[sales_train['item_price'] < 0, 'item_price'] = price_median
#Ver la cantidad de ventas de los cinco productos principales

sales_train['item_cnt_day'].sort_values(ascending=False)[:5]
#Se revisa la información de datos correspondiente al volumen máximo de ventas de 2169

sales_train[sales_train['item_cnt_day'] == 2169]
#En un día de octubre, el producto No. 11373 se vendió 2169 veces en la tienda número 12.

#Consulta la información correspondiente de este producto.

items[items['item_id'] == 11373]
#Con la ayuda de la traducción, este es un tipo de bienes relacionados con la empresa de transporte rusa "Boxberry".

#Continuar verificando las ventas de este producto en otras tiendas

sales_train[sales_train['item_id'] == 11373]
#La información del volumen de ventas correspondiente al producto

sale_num = sales_train[sales_train['item_id'] == 11373]['item_cnt_day']
sale_num.describe()
#Se puede ver que el 11373 generalmente se vende muy poco y el volumen de ventas es casi de un solo dígito (75% = 8 piezas).

#Por tanto, el número de ventas 2169 puede considerarse un valor anormal y debe eliminarse.

sales_train = sales_train[sales_train['item_cnt_day'] < 2000]
#Además, hay otro producto que se ha vendido 1000 veces, por lo que debe buscar este producto para el seguro.

#La información de datos correspondiente al volumen de ventas de 1000

sales_train[sales_train['item_cnt_day'] == 1000]
#La información correspondiente de este producto.

items[items['item_id'] == 20949]
#Con la ayuda de la traducción, esta es una pequeña camiseta blanca de la marca Mike.

#Verificando las ventas de este producto en otras tiendas

sales_train[sales_train['item_id'] == 20949]
#La información del volumen de ventas correspondiente al producto

sale_num = sales_train[sales_train['item_id'] == 20949]['item_cnt_day']
sale_num.describe()
#Del mismo modo, para 20949 este producto se vende generalmente muy poco, 
#el volumen de ventas es casi de un solo dígito (75% = 7 piezas), e inesperadamente hay un volumen de ventas negativo.

#Por lo tanto, el número de ventas de 1000 puede considerarse un valor anormal y debe eliminarse.
sales_train = sales_train[sales_train['item_cnt_day'] < 1000]
#La información anterior nos recuerda que debemos verificar qué productos tienen la menor cantidad de ventas

sales_train['item_cnt_day'].sort_values(ascending=True)[:10]
#Parece que muchos productos tienen valores de venta negativos, 
#lo que puede significar que estos productos no se venden sino que se compran, por lo que no nos ocupamos de este tema.
fig = plt.figure(figsize=(20,10))  # Tablero
plt.subplots_adjust(hspace=.4)  # Espaciado de subimagen

#Ver la distribución de shop_id
plt.subplot2grid((1,1), (0,0), rowspan=1, colspan=3)
test['shop_id'].value_counts(normalize=True).plot(kind='bar', color='darkviolet')
plt.title('El estado de distribución del ID de tienda del conjunto de prueba (Figura 6)')
plt.xlabel('ID de tienda')
plt.ylabel('Número de ocurrencias de ID de tienda estandarizado')

plt.show()
fig = plt.figure(figsize=(20,10))  # Tablero
plt.subplots_adjust(hspace=.4)  # Espaciado de subimagen


#Ver la distribución de item_id
plt.subplot2grid((1,1), (0,0), rowspan=1, colspan=1)
test['item_id'].plot(kind='hist', color='sienna')
plt.title('El estado de distribución de los ID de productos en el conjunto de prueba (Figura 7)')
plt.xlabel('ID de producto')
plt.ylabel('Número de apariciones de ID de producto estandarizado')

plt.show()
#Aunque el conjunto de entrenamiento tiene más ID que el conjunto de prueba, 
#no hay garantía de que el conjunto de entrenamiento contenga todas las tiendas que aparecen en el conjunto de prueba.

#Por lo tanto, es necesario verificar si el ID del conjunto de prueba es un subconjunto del ID del conjunto de entrenamiento.

def is_subset(set0,set1):
    if set0.issubset(set1):
        print ("Los dos son la relación de inclusión del subconjunto") 
    else:
        print ("Los dos no son un subconjunto de la relación de inclusión")

shops_train_set = set(sales_train['shop_id'].unique())
shops_test_set = set(test['shop_id'].unique())

print('El resultado del juicio es:')
is_subset(shops_test_set,shops_train_set)
#Aquí se determina que todos los ID de tienda en el conjunto de prueba están en el conjunto de entrenamiento.
#Sin embargo, en la discusión de la competencia del proyecto, se mencionó una pregunta sobre tiendas duplicadas, que puede requerir nuestro análisis.
#Comparar el nombre y la identificación de la tienda
shops
#Sorprendentemente, estos nombres de tiendas se basan en ciudades y regiones, lo que puede ser una característica potencial

#Además, un análisis cuidadoso puede encontrar que los nombres de las tiendas con los ID 0 y 1 son casi los mismos que los que tienen los ID 57 y 58. La diferencia es que las tiendas 0 y 1 también tienen la palabra'фран '(Fran) adjunta.

#Además, el nombre de la tienda con ID 10 es casi el mismo que ID 11, ambos son "Жуковский ул. Чкалова 39м" (Zhukovsky Avenue Chkalov 39m)

#La única diferencia entre los dos es que los caracteres del subíndice final son diferentes, que son '? 'Y 2'
#Por lo tanto, creo que estos elementos de identificación casi duplicados deberían fusionarse 
#(tanto el conjunto de entrenamiento como el conjunto de prueba)

sales_train.loc[sales_train['shop_id'] == 0, 'shop_id'] = 57
test.loc[test['shop_id'] == 0, 'shop_id'] = 57

sales_train.loc[sales_train['shop_id'] == 1, 'shop_id'] = 58
test.loc[test['shop_id'] == 1, 'shop_id'] = 58

sales_train.loc[sales_train['shop_id'] == 10, 'shop_id'] = 11
test.loc[test['shop_id'] == 10, 'shop_id'] = 11
#Mire la cantidad de ID de tienda en el conjunto de entrenamiento y el conjunto de prueba después de la fusión

shops_train = sales_train['shop_id'].nunique()
shops_test = test['shop_id'].nunique()
print('Los ID de tienda en el conjunto de entrenamiento son {} 个 '.format(shops_train))
print('Los ID de tienda en el conjunto de prueba son {} 个 '.format(shops_test))
#Extraemos la ciudad en el nombre de la tienda 

#Vemos el nombre de la tienda de los primeros cinco ID
shops['shop_name'][:5]
#Se extra el nombre de la ciudad en el nombre de la tienda

shop_cities = shops['shop_name'].str.split(' ').str[0]
shop_cities.unique()
#Después de una cuidadosa observación, se descubrió que Yakutsk usaba dos expresiones, '! Якутск' y'Якутск '.
#Supongo que su significado debería ser el mismo, así que los fusionaremos en una categoría. 
#y se pone el nombre de la ciudad como una característica nueva en los datos de las tiendas.

shops['city'] = shop_cities
shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'

#Ver datos de tiendas actuales
shops
#Se convierte las características de la ciudad en etiquetas numéricas

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
shops['shop_city'] = label_encoder.fit_transform(shops['city'])
#Ahora ya no necesitamos las dos variables de'hop_name 'y'city', así que se eliminan

shops = shops.drop(['shop_name', 'city'], axis = 1)
shops.head()
#Se realizan operaciones de análisis similares en el ID de artículo "item_ids" ahora aplicado a la cantidad de productos

#Verificación de contención similar

items_train_set = set(sales_train['item_id'].unique())
items_test_set = set(test['item_id'].unique())

print('El resultado del juicio es: ')
is_subset(items_test_set,items_train_set) 
#La cantidad de estos ID de productos que no pertenecen a un subconjunto

len(items_test_set.difference(items_train_set))
#Puede verse que hay 363 elementos en el conjunto de prueba que no están en el conjunto de entrenamiento.
#Pero esto no significa que el pronóstico de ventas para estos productos deba ser cero, 
#porque se pueden agregar nuevos productos a los datos de capacitación, pero cómo predecir su valor es un problema difícil.

#Antes de procesar, necesitamos comprender mejor los productos 5100 en este conjunto de prueba. 

#A qué categoría pertenecen y qué categorías no necesitamos predecir en el conjunto de prueba.

item_in_test = items.loc[items['item_id'].isin(sorted(test['item_id'].unique()))]
cats_in_test = item_in_test.item_category_id.unique()
#La información de categoría en los datos de categoría item_cats que no están en la prueba (categorías comunes en el conjunto de entrenamiento)

item_cats.loc[~item_cats['item_category_id'].isin(cats_in_test)]
#Ver datos de categoría en item_cats

item_cats['item_category_name']

#Separe los caracteres con'- '
cats_ = item_cats['item_category_name'].str.split('-')

#Extrae la categoría principal en item_cats
item_cats['main_category'] = cats_.map(lambda row: row[0].strip())  # Extraiga el carácter anterior, use strip () para eliminar las unidades que no son caracteres

#Extrae subcategorías en item_cats (si no hay una subcategoría, use la categoría principal como subcategoría)
item_cats['sub_category'] = cats_.map(lambda row: row[1].strip() if len(row) > 1 else row[0].strip())
#Se codifica digitalmente la nueva clase

label_encoder = preprocessing.LabelEncoder()

item_cats['main_category_id'] = label_encoder.fit_transform(item_cats['main_category'])
item_cats['sub_category_id'] = label_encoder.fit_transform(item_cats['sub_category'])
item_cats.head()
#Se generan pares de tuplas de datos de artículos de tienda cada mes en los datos de entrenamiento

# Se convierte la proporción de tiempo en los datos de ventas para obtener la hora y la fecha en el formato especificado: 'día / mes / año'

sales_train['date'] = pd.to_datetime(sales_train['date'], format='%d.%m.%Y') 
#Se crea un iterador para generar tuplas que representen el producto cartesiano de elementos en item1, item2, etc.


from itertools import product 
shops_in_jan = sales_train.loc[sales_train['date_block_num']==0, 'shop_id'].unique()  # Ontiene la cantidad de ID de tienda que comienzan en 0
items_in_jan = sales_train.loc[sales_train['date_block_num']==0, 'item_id'].unique()  # Ontiene la cantidad de ID de producto a partir de 0 meses
jan = list(product(*[shops_in_jan, items_in_jan, [0]]))    # Genera una tupla del producto cartesiano del número de ID de tienda y el número de ID de producto, y luego lo convierte en una lista
#Los primeros cinco resultados de la tupla cartesiana, las posiciones de izquierda a derecha 
#representan respectivamente: (ID de tienda, ID de producto, número de mes actual)

print(jan[:5])
#El número total de tuplas cartesianas (que indica 0 mes)

print(len(jan))
#Productos Descartes fabricados en febrero de 2013 (segundo mes)

shops_in_feb = sales_train.loc[sales_train['date_block_num']==1, 'shop_id'].unique()
items_in_feb = sales_train.loc[sales_train['date_block_num']==1, 'item_id'].unique()
feb = list(product(*[shops_in_feb, items_in_feb, [1]]))
#Tupla cartesiana para el segundo mes
print(feb[:5])
#Número de tuplas cartesianas en el segundo mes

print(len(feb))

#Se utiliza el método de apilamiento de matrices 'vstack' de numpy para fusionar los datos de la tupla cartesiana de los dos meses anteriores y crear un formato de marco de datos para facilitar la visualización
cartesian_jf = np.vstack((jan, feb))    # vstack (dirección vertical) apila matrices.
cartesian_jf_df = pd.DataFrame(cartesian_jf, columns=['shop_id', 'item_id', 'date_block_num'])   # Crea un marco de datos y nombra diferentes columnas
cartesian_jf_df.head().append(cartesian_jf_df.tail())
#Se combinan los 33 meses con los mismos datos y crea df (data-frame)

months = sales_train['date_block_num'].unique()
cartesian = []
for month in months:
    shops_in_month = sales_train.loc[sales_train['date_block_num']==month, 'shop_id'].unique()
    items_in_month = sales_train.loc[sales_train['date_block_num']==month, 'item_id'].unique()
    cartesian.append(np.array(list(product(*[shops_in_month, items_in_month, [month]])), dtype='int32'))
    
cartesian_df = pd.DataFrame(np.vstack(cartesian), columns = ['shop_id', 'item_id', 'date_block_num'], dtype=np.int32)
#Forma de datos consolidados para todos los meses

cartesian_df.shape
cartesian_df.head()
#Los objetos de secuencia shop_id, 'item_id' y'date_block_num 'para agrupar el conjunto de datos y luego se extrae la suma del volumen de ventas mensual'item_cnt_day'

#Es decir, se puede obtener las ventas mensuales totales de productos específicos en tiendas específicas

x = sales_train.groupby(['shop_id', 'item_id', 'date_block_num'])['item_cnt_day'].sum().rename('item_cnt_month').reset_index()
x.head()
x.shape
#El método pd.merge () fusiona y une. ''LEFT'' significa que solo se retiene la clave primaria izquierda y no se toman las filas que existen solo en la clave primaria derecha.

new_train = pd.merge(cartesian_df, x, on=['shop_id', 'item_id', 'date_block_num'], how='left').fillna(0) 
#Se usa numpy.clip para escalar las ventas mensuales item_cnt_month a [0,20], que se menciona en la descripción del proyecto

new_train['item_cnt_month'] = np.clip(new_train['item_cnt_month'], 0, 20)

new_train.head()
# sort_values reorganiza new_train de acuerdo con el orden de la clasificación interna de 'date_block_num', 'shop_id' e'item_id '

new_train.sort_values(['date_block_num','shop_id','item_id'], inplace = True)  
new_train.head()
#Elimina listas innecesarias del sistema y libera memoria

del x
del cartesian_df
del cartesian
del cartesian_jf
del cartesian_jf_df
del feb
del jan
del items_test_set
del items_train_set
del sales_train
#Ahora insertamos el atributo date_block_num (el mes 34) y el atributo de volumen de ventas'item_cnt_month '(tentativamente establecido en 0) para el conjunto de prueba.

#Se usa el método de inserción de pandas para colocar esta nueva columna en un índice específico. Esto hace que sea más fácil conectar el equipo de prueba al equipo de entrenamiento más adelante.


test.insert(loc=3, column='date_block_num', value=34)        # Inserta el número de meses en la tercera columna del conjunto de prueba y asigna un valor de 34
test['item_cnt_month'] = 0  # Inserta una nueva columna'item_cnt_month 'en el conjunto de prueba y asigna un valor de 0
test.head()
#Elimina la columna de ID que no está incluida en el conjunto de prueba en relación con new_train y combina con el conjunto de entrenamiento original

new_train = new_train.append(test.drop('ID', axis = 1)) 
new_train.head().append(new_train.tail())
#Combina los datos de la tienda para obtener la categoría de la ciudad codificada con el ID correspondiente

new_train = pd.merge(new_train, shops, on=['shop_id'], how='left') 
new_train.head()
#Combina los datos del nombre del producto para obtener la categoría de producto codificada con el ID correspondiente

new_train = pd.merge(new_train, items.drop('item_name', axis = 1), on=['item_id'], how='left')
new_train.head()
#Combina los datos de la categoría de producto para obtener la categoría padre-hijo del producto del número de código bajo el nombre correspondiente

new_train = pd.merge(new_train,  item_cats.drop('item_category_name', axis = 1), on=['item_category_id'], how='left')
new_train.head()
#Elimina columnas no numéricas

new_train.drop(['main_category','sub_category'],axis=1,inplace=True)
new_train.head()
#Elimina datos inútiles y libera memoria

del items
del item_cats
del shops
del test
#Se generan características de retraso y codificación promedio

#Definir la función de adición de características de histéresis
def generate_lag(train, months, lag_column):
    for month in months:
        # Crear una función de retraso
        train_shift = train[['date_block_num', 'shop_id', 'item_id', lag_column]].copy()
        train_shift.columns = ['date_block_num', 'shop_id', 'item_id', lag_column+'_lag_'+ str(month)]
        train_shift['date_block_num'] += month
        #La nueva lista está conectada al conjunto de entrenamiento.
        train = pd.merge(train, train_shift, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    return train
#Define la función de conversión de tipo de datos descendente.
#La función es convertir el tipo float64 a float16 y convertir int64 a int16
#(usado para reducir la cantidad de memoria)

from tqdm import tqdm_notebook  
def downcast_dtypes(df):   
    # Seleccina las columnas a procesar
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols =   [c for c in df if df[c].dtype == "int64"]
    
    #Inicia conversión de datos
    df[float_cols] = df[float_cols].astype(np.float16)
    df[int_cols]   = df[int_cols].astype(np.int16)
    
    return df
#Funcion de transformación para cambiar los tipos de datos
new_train = downcast_dtypes(new_train)  
%%time
#Agrega la función de retraso de la variable objetivo (atributo de ventas mensuales) y agrega parte de los datos de ventas mensuales
new_train = generate_lag(new_train, [1,2,3,4,5,6,12], 'item_cnt_month')
%%time
#Agrega características de retraso del producto de la media objetivo
#Ordena por mes e identificación de producto y toma el promedio de sus ventas mensuales
group = new_train.groupby(['date_block_num', 'item_id'])['item_cnt_month'].mean().rename('item_month_mean').reset_index()

#Agrega la nueva tabla a la derecha de new_train, correspondiente a los atributos 'date_block_num', 'item_id'
new_train = pd.merge(new_train, group, on=['date_block_num', 'item_id'], how='left')

#Agrega las ventas mensuales atrasadas a [1,2,3,6,12] meses (llenado promedio)
new_train = generate_lag(new_train, [1,2,3,6,12], 'item_month_mean')

#Elimina el atributo'item_month_mean 'no deseado
new_train.drop(['item_month_mean'], axis=1, inplace=True)
%%time
#Agrega la función de retraso medio de destino de tienda
group = new_train.groupby(['date_block_num', 'shop_id'])['item_cnt_month'].mean().rename('shop_month_mean').reset_index()
new_train = pd.merge(new_train, group, on=['date_block_num', 'shop_id'], how='left')
new_train = generate_lag(new_train, [1,2,3,6,12], 'shop_month_mean')
new_train.drop(['shop_month_mean'], axis=1, inplace=True)
%%time
#Agrega la función de retraso de la media de destino de categoría de producto de tienda
group = new_train.groupby(['date_block_num', 'shop_id', 'item_category_id'])['item_cnt_month'].mean().rename('shop_category_month_mean').reset_index()
new_train = pd.merge(new_train, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
new_train = generate_lag(new_train, [1, 2], 'shop_category_month_mean')
new_train.drop(['shop_category_month_mean'], axis=1, inplace=True)
%%time
#Agrega la categoría principal del producto: la característica rezagada de la media objetivo
group = new_train.groupby(['date_block_num', 'main_category_id'])['item_cnt_month'].mean().rename('main_category_month_mean').reset_index()
new_train = pd.merge(new_train, group, on=['date_block_num', 'main_category_id'], how='left')
new_train = generate_lag(new_train, [1], 'main_category_month_mean')
new_train.drop(['main_category_month_mean'], axis=1, inplace=True)
%time
#Agrega características de rezago de subcategoría de producto de la media objetivo
group = new_train.groupby(['date_block_num', 'sub_category_id'])['item_cnt_month'].mean().rename('sub_category_month_mean').reset_index()
new_train = pd.merge(new_train, group, on=['date_block_num', 'sub_category_id'], how='left')
new_train = generate_lag(new_train, [1], 'sub_category_month_mean')
new_train.drop(['sub_category_month_mean'], axis=1, inplace=True)
#Morfología del conjunto de datos después de agregar características de histéresis
new_train.tail()
#Agrega las características de un atributo de mes
new_train['month'] = new_train['date_block_num'] % 12
#Conversión de tipo de datos nuevamente
new_train = downcast_dtypes(new_train)
new_train.head().append(new_train.tail())
#Debido a que no hay una característica de datos de las transacciones de valores en el primer año, se utilizará como entrada a partir del segundo año.
new_train = new_train[new_train.date_block_num > 11]
#Se usa  0 para completar, indicando una muestra sin datos

def fill_nan(df):
    for col in df.columns:
        if ('_lag_' in col) & (df[col].isna().any()):
            df[col].fillna(0, inplace=True)         
    return df

new_train =  fill_nan(new_train)

#Extracción de características de los datos de entrenamiento
train_feature = new_train[new_train.date_block_num < 33].drop(['item_cnt_month'], axis=1)

#Extracción de etiquetas de datos de entrenamiento
train_label = new_train[new_train.date_block_num < 33]['item_cnt_month']
#Extracción de características de datos de verificación
val_feature = new_train[new_train.date_block_num == 33].drop(['item_cnt_month'], axis=1)

#Extracción de etiquetas de datos de verificación
val_label = new_train[new_train.date_block_num == 33]['item_cnt_month']
test_feature = new_train[new_train.date_block_num == 34].drop(['item_cnt_month'], axis=1)
train_feature.shape,train_label.shape,val_feature.shape,val_label.shape
train_feature.head()
import gc
gc.collect()
from xgboost import XGBRegressor
#Se establecen parámetros de modelo

model = XGBRegressor(n_estimators=3000,
                     max_depth=10,
                     colsample_bytree=0.5, 
                     subsample=0.5, 
                     learning_rate = 0.01
                    )
%%time
#Se realiza el entrenamiento del modelo y configura la función de parada anticipada.

model.fit(train_feature.values, train_label.values, 
          eval_metric="rmse", 
          eval_set=[(train_feature.values, train_label.values), (val_feature.values, val_label.values)], 
          verbose=True, 
          early_stopping_rounds = 50)
y_pred = model.predict(test_feature.values)
#Vista de importancia de la función
importances = pd.DataFrame({'feature':new_train.drop('item_cnt_month', axis = 1).columns,'importance':np.round(model.feature_importances_,3)}) 
importances = importances.sort_values('importance',ascending=False).set_index('feature') 
importances = importances[importances['importance'] > 0.01]

importances.plot(kind='bar',
                 title = 'Feature Importance',
                 figsize = (8,6),
                 grid= 'both')
#Exportar resultado
submission['item_cnt_month'] = y_pred
submission.to_csv('future_sales_submission2.csv', index=False)