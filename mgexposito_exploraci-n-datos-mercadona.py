# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Visualización

import seaborn as sns

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# cargamos el datset

data=pd.read_csv('/kaggle/input/mercadona-es-product-pricing/thegurus-opendata-mercadona-es-products.csv')
# muestra información del dataset

print(data.info(), '\n')
# muestra las primeras 3 filas

print(data.head(3), '\n')
# Nos quedamos únicamente con los datos del producto Banana y lo mostramos

data_banana = data[data['name']=='Banana']

print(data_banana)
# agrupamos por nombre y calculamos la media del precio

data_agg = data.groupby("name").agg({"reference_price": "mean"})

data_agg.reset_index(inplace=True)
# mostramos información del dataset con los datos agregados

print(data_agg.info())
# ordenamos por el precio de referencia

data_agg_sort = data_agg.sort_values(by='reference_price')

#mostramos los 5 productos que tienen el precio de referencia más bajo

print(data_agg_sort.head(5))

#mostramos los 5 productos que tienen el precio de referencia más elevado

print(data_agg_sort.tail(5))
# calculamos el número de productos que existen por categoría y los mostramos en un gráfico de barras

figure(num=None, figsize=(18, 16), dpi=80)

data.groupby('category')['name'].count().plot(kind='bar')