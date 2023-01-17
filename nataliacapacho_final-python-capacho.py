import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import seaborn as sns 

import matplotlib.pyplot as plt 

%matplotlib inline 
data50 = pd.read_csv("../input/50startup/data50.csv")
data50.head()
data50=  data50.dropna()
data50.dtypes
Estado=data50['State']

new_state = Estado.replace('New York', 1).replace('California', 2).replace('Florida', 3)

print(new_state)
print(data50)
data50["State"].value_counts() 
Estados = ("New York", "California", "Florida")

posicion_y = np.arange(len(Estados))

unidades = (17, 17, 16)

plt.bar(Estados, unidades, width=0.5, align="center", color="crimson")

plt.ylabel('Número de empresas', weight="bold")

plt.title("Nuevas Empresas por Estado", weight="bold")
plt.scatter(data50['Profit'],data50['R&D Spend'], alpha=.7, color='springgreen')

plt.ylabel('Gasto en I&D', weight="bold")

plt.xlabel('Ganancia', weight="bold")

plt.title("Ganancia / Gasto en I&D", weight="bold")
data50.rename(columns={'R&D Spend': 'R&D', 'Administration': 'ADM', 'Marketing Spend': 'MKT', 'Profit': 'PRF'}, inplace=True)

corr = data50.corr(method='pearson')

display(corr)

ax = plt.axes()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, mask=mask, square=True,xticklabels=corr.columns.values,yticklabels=corr.columns.values, cmap=sns.diverging_palette(220, 10, as_cmap=True), annot=True, fmt='.4f', linewidths=.4)

ax.set_title('Correlación de las variables', weight="bold")
plt.figure(figsize=(7,7))

sns.barplot(data50['State'],data50['MKT'],linewidth=1, edgecolor="k",color="lightseagreen",label="Gastos en Marketing")

sns.barplot(data50['State'],data50['ADM'],linewidth=1, edgecolor="k",color="darkorange",label="Gastos en Administración")

sns.barplot(data50['State'],data50['R&D'],linewidth=1, edgecolor="k",color="lightpink",label="Gastos en I&D")

plt.legend(loc="left",prop={"size":10})

plt.title("Distribución de los gastos",weight="bold")

plt.ylabel("Gastos administración/I&D/Marketing ", weight="bold")
plt.figure(figsize=(8,7))

box = sns.boxplot(y=data50['PRF'],x=data50['State'], palette="YlOrRd")

plt.title('Ganancias obtenidas',color='black',weight="bold")

plt.xlabel('Estados',weight="bold")

plt.ylabel('Ganancia',weight="bold")