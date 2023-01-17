# Importar librerias utiles para la manipulacion y visualizacion de data
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# importing functools for reduce()
import functools
# importing operator for operator functions
import operator
# Importar datos de archivo train.csv
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

figsize_rect=(13,7) # dimensiones para graficos en formato rectangular
figsize_sqr=(10, 10) # dimensiones para graficos en formato cuadrado

mpl.rcParams['font.size'] = 15
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['lines.linestyle'] = '-'
mpl.rcParams['lines.color'] = '#1372B2'
mpl.rcParams["legend.title_fontsize"] = 16
colors = ['#1372B2', "#F19917",'#F76413','#2B6B85','#359CAE']
plt.figure(figsize=figsize_sqr)
frequencyToMszoning = train.groupby('MSZoning')['MSZoning'].count()
frequencyToMszoning.plot(kind='pie', autopct='%1.f%%', textprops=dict(color="#fff"), colors = colors, fontsize=18)
legend = (
    'Commercial', 'Floating Village Residential', 'Residential High Density', 'Residential Low Density',
    'Residential Medium Density')
plt.legend(labels=legend,
           title="General zoning classification of the sale",
           loc="center left",
           bbox_to_anchor=(1, 0, 0.5, 1))
plt.draw()
plt.figure(figsize=figsize_rect)
frequencyToMszoning.plot(kind='bar')
plt.ylabel("Frequency")
plt.draw()
plt.figure(figsize=figsize_sqr)
frequencyToFireplaces = train.groupby('Fireplaces')['Fireplaces'].count()
frequencyToFireplaces.plot(kind='pie', autopct='%1.f%%', textprops=dict(color="#000"), colors = colors, fontsize=18)
legend = ('0', '1', '2', '3')
plt.legend(labels=legend,
           title="Number of fireplaces",
           loc="center left",
           bbox_to_anchor=(1, 0, 0.5, 1))
plt.draw()
plt.figure(figsize=figsize_rect)
frequencyToFireplaces.plot(kind='bar')
plt.ylabel("Frequency")
plt.draw()
plt.figure(figsize=figsize_rect)
yearToPrice = train.groupby('BedroomAbvGr')['BedroomAbvGr'].count().plot.line(figsize=figsize_rect, marker='o')
plt.xlabel('Bedrooms above grade')
plt.ylabel('Frequency')
plt.draw()
x = train.sort_values(by=['TotRmsAbvGrd'])['TotRmsAbvGrd'].tolist()
frequencies = train.groupby('TotRmsAbvGrd')['TotRmsAbvGrd'].count().tolist()
y = functools.reduce(operator.add, [list(range(1, n + 1)) for n in frequencies])
plt.figure(figsize=figsize_rect)
marker_size = 40
plt.scatter(x, y, marker_size)
plt.ylabel("Frequency")
plt.xlabel("Total rooms above grade")
plt.draw()
plt.figure(figsize=figsize_rect)
plt.hist(train.SalePrice, bins=15, rwidth=0.99)
plt.xlabel("Sale Price")
plt.ylabel("Frequency")
plt.draw()
plt.show() # Mostrar graficos
def get_outliers(attr):
    list_data = train[attr].tolist()
    Q1, Q3 = np.quantile(list_data, [0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return [num for num in list_data if num < lower_bound or num > upper_bound]


print(get_outliers('SalePrice'))  # Mediciones atipicas en la variable GarageArea
print(get_outliers('TotRmsAbvGrd'))  # Mediciones atipicas en la variable LotArea