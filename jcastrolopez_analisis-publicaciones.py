import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as datetime

import sys

from normalize_data import getNormalizedDataset

df = getNormalizedDataset()

df['mes'] = df['fecha'].dt.month

df['año'] = df['fecha'].dt.year

# Veamos el comportamiento a nivel general de ZonaProp en Mexico

zpropMexico = df.groupby(['año']).size()

zpropMexico = zpropMexico.reset_index()

zpropMexico.rename(columns={0:'Publicaciones'}, inplace=True)

total = ((zpropMexico.sum()).values)[1]



fig, ax = plt.subplots(figsize=(15 ,5))

ax.plot(zpropMexico['año'], zpropMexico['Publicaciones'], label='Total', color='blue')



plt.xticks([2012,2013,2014,2015,2016])

plt.grid(b=True, which='major', axis='both')

ax.set_xlabel("\n Año", fontsize=18)

ax.set_ylabel("Publicaciones \n", fontsize=18)

ax.legend(loc='best', title_fontsize=16, fontsize=14)

ax.set_title('Cantidad de publicaciones según el año \n', fontdict={'fontsize':20})


## Analisis de variacion entre periodos



nDf = zpropMexico.set_index('año')

dfVariacion = pd.DataFrame(columns=['period', 'variation'])

dfVariacion = dfVariacion.append({ "period": '2012-2013', "variation": (nDf.loc[2013].Publicaciones - nDf.loc[2012].Publicaciones) * 100 / total}, ignore_index=True)

dfVariacion = dfVariacion.append({ "period": '2013-2014', "variation": (nDf.loc[2014].Publicaciones - nDf.loc[2013].Publicaciones) * 100 / total}, ignore_index=True)

dfVariacion = dfVariacion.append({ "period": '2014-2015', "variation": (nDf.loc[2015].Publicaciones - nDf.loc[2014].Publicaciones) * 100 / total}, ignore_index=True)

dfVariacion = dfVariacion.append({ "period": '2015-2016', "variation": (nDf.loc[2016].Publicaciones - nDf.loc[2015].Publicaciones) * 100 / total}, ignore_index=True)



fig, ax = plt.subplots(figsize=(15,5))

ax.plot(dfVariacion['period'], dfVariacion['variation'], label='Crecimiento', color='red')



plt.grid(b=True, which='major', axis='both')

ax.set_xlabel("\n Sub-período", fontsize=18)

ax.set_ylabel("% Porcentaje \n", fontsize=18)

ax.legend(loc='best', title_fontsize=16, fontsize=14)

ax.set_title('Crecimiento porcentual según sub-período \n', fontdict={'fontsize':20})

# Hipotesis:

# En que ciudades vario mas las cantidades de publicaciones durante los 5 años?.

# Armar una tabla tipo pivot de: provincias vs años, y en el valor pongo la cantidad de publicaciones



data = df.groupby(['provincia','año']).size()

data = data.reset_index()

data.rename(columns={0:'count'}, inplace=True)

data = data.pivot(index='provincia', columns='año', values='count')

data = data.reset_index()

data.columns = ['provincia',2012,2013,2014,2015,2016]

data['total'] = data[2012]+data[2013]+data[2014]+data[2015]+data[2016]

data['variacion_total'] = (data[2016]-data[2012])/data['total'] * 100

data['variacion_2012-2013'] = (data[2013]-data[2012])/data['total'] * 100

data['variacion_2013-2014'] = (data[2014]-data[2013])/data['total'] * 100

data['variacion_2014-2015'] = (data[2015]-data[2014])/data['total'] * 100

data['variacion_2015-2016'] = (data[2016]-data[2015])/data['total'] * 100

#data.max()

# data.iloc[data['variacion_2012-2013'].idxmax(),:]

# Vemos que hay publicaciones
# Que provincias presentaron mayor crecimiento en cuanto a cantidad de publicaciones año a año?

# Esta informacion puede servir ya que se puede ver donde va creciendo y teniendo mas lugares para vivir cierta zona





d1 = data['variacion_2012-2013'].idxmax()

d2 = data['variacion_2013-2014'].idxmax()

d3 = data['variacion_2014-2015'].idxmax()

d4 = data['variacion_2015-2016'].idxmax()



# Pasar a valores relativos porcentuales.

d = data.loc[[d1, d2, d3, d4], : ]



d[2012] = d[2012]/d['total']*100

d[2013] = d[2013]/d['total']*100

d[2014] = d[2014]/d['total']*100

d[2015] = d[2015]/d['total']*100

d[2016] = d[2016]/d['total']*100





fig, ax = plt.subplots(figsize=(15,7))

plt.grid(b=True, which='major', axis='both')

d = d.reset_index(drop=True)

for index, row in d.iterrows():

    print(index)

    row = d.loc[[index], :]

    label = row['provincia'].tolist()[0]

    total = row.total

    row = row.loc[:, [2012,2013,2014,2015,2016]].melt()

    año = row.variable.to_list()

    valor = row.value.to_list()

    colors=['r', 'g', 'b', 'm']

    for i, (x1, x2, y1, y2) in enumerate(zip(año, año[1:], valor, valor[1:])):

        if(i == index): 

            ax.plot([x1, x2], [y1, y2], colors[index], label=label, linewidth=4)

        ax.plot([x1, x2], [y1, y2], colors[index], linewidth=1)



#     ax.plot(año, valor, label=label)

    

plt.legend(fontsize=20)

plt.xticks([2012,2013,2014,2015,2016])

#plt.yticks([0,10,20,30,40,50,60,70,80,90,100])

ax.set_xlabel("\n Año", fontsize=18)

ax.set_ylabel("% Porcentaje de crecimiento \n", fontsize=18)   

ax.legend(loc='best', title_fontsize=16, fontsize=14)

ax.set_title('Provincias con mayor crecimiento de publicaciones por sub-período \n', fontdict={'fontsize':20})

    

d
# Que provincias presentaron mayor decrecimiento en cuanto a cantidad de publicaciones año a año?

# Esta informacion puede servir ya que se puede ver donde va creciendo y teniendo mas lugares para vivir cierta zona



d1 = data['variacion_2012-2013'].idxmin()

d2 = data['variacion_2013-2014'].idxmin()

d3 = data['variacion_2014-2015'].idxmin()

d4 = data['variacion_2015-2016'].idxmin()



# Pasar a valores relativos porcentuales.

d = data.loc[[d1, d2, d3, d4], : ]

d = d.reset_index(drop=True)

d[2012] = d[2012]/d['total']*100

d[2013] = d[2013]/d['total']*100

d[2014] = d[2014]/d['total']*100

d[2015] = d[2015]/d['total']*100

d[2016] = d[2016]/d['total']*100



fig, ax = plt.subplots(figsize=(15,7))

plt.grid(b=True, which='major', axis='both')

for index, row in d.iterrows():



    row = d.loc[[index], :]

    label = row['provincia'].tolist()[0]

    total = row.total

    row = row.loc[:, [2012,2013,2014,2015,2016]].melt()

    año = row.variable.to_list()

    valor = row.value.to_list()

    colors=['r', 'g', 'b', 'm']

    for i, (x1, x2, y1, y2) in enumerate(zip(año, año[1:], valor, valor[1:])):

        if(i == index): 

            ax.plot([x1, x2], [y1, y2], colors[index], label=label, linewidth=4)

        ax.plot([x1, x2], [y1, y2], colors[index], linewidth=1)

    



plt.xticks([2012,2013,2014,2015,2016])

#plt.yticks([0,10,20,30,40,50,60,70,80,90,100])

ax.set_xlabel("\n Año", fontsize=18)

ax.set_ylabel("% Porcentaje de crecimiento \n", fontsize=18)

ax.legend(loc='best', title_fontsize=16, fontsize=14)  

ax.set_title('Provincias con mayor decrecimiento de publicaciones por sub-período \n', fontdict={'fontsize':20})

    

# Que provincias presentaron mayor decrecimiento en cuanto a cantidad de publicaciones año a año?

# Esta informacion puede servir ya que se puede ver donde va creciendo y teniendo mas lugares para vivir cierta zona



data = df.groupby(['provincia','año']).size()

data = data.reset_index()

data.rename(columns={0:'count'}, inplace=True)

data = data.pivot(index='provincia', columns='año', values='count')

data = data.reset_index()

data.columns = ['provincia',2012,2013,2014,2015,2016]

data['total'] = data[2012]+data[2013]+data[2014]+data[2015]+data[2016]

data['variacion_total'] = (data[2016]-data[2012])/data['total']

data['variacion_2012-2013'] = (data[2013]-data[2012])/data['total']

data['variacion_2013-2014'] = (data[2014]-data[2013])/data['total']

data['variacion_2014-2015'] = (data[2015]-data[2014])/data['total']

data['variacion_2015-2016'] = (data[2016]-data[2015])/data['total']

d = data.sort_values(by='variacion_total', ascending=False)



d[2012] = d[2012]/d['total']*100

d[2013] = d[2013]/d['total']*100

d[2014] = d[2014]/d['total']*100

d[2015] = d[2015]/d['total']*100

d[2016] = d[2016]/d['total']*100





fig, ax = plt.subplots(figsize=(15,14))



hasRedLabel = False

hasBlueLabel = False

for index, row in d.iterrows():

    row = d.loc[[index], :]

    label = row['provincia'].tolist()[0]

    total = row.total

    meltRow = row.loc[:, [2012,2013,2014,2015,2016]].melt()

    año = meltRow.variable.to_list()

    valor = meltRow.value.to_list()

    colors=['r', 'g', 'b', 'm']

    for i, (x1, x2, y1, y2) in enumerate(zip(año, año[1:], valor, valor[1:])):

        variation = 0

        label = ""

        if(i == 0): variation = row['variacion_2012-2013']

        if(i == 1): variation = row['variacion_2013-2014']

        if(i == 2): variation = row['variacion_2014-2015']            

        if(i == 3): variation = row['variacion_2015-2016']

        if(variation.values[0] < 0 and not hasRedLabel) :

            hasRedLabel = True

            label = 'Decrecimiento'

            

        if(variation.values[0] >= 0 and not hasBlueLabel) :

            hasBlueLabel = True

            label = 'Crecimiento'

            

#         print([x1, x2], [y1, y2])

        ax.plot([x1, x2], [y1, y2], color='#cc3333' if variation.values[0] < 0 else '#225ea8', label=label)

#         ax.plot([x1, x2], [y1, y2], color=variation, linewidth=1)



#         ax.plot(año, valor, label=label, color='c' if index < 15 else 'y')

    

newDf = d.mean().reset_index()



newDf['newIndex'] = 1

newDf = newDf.pivot(columns='index', index="newIndex", values=0).reset_index()

print(newDf.head)

# for index, row in newDf.iterrows():



#     row = newDf.loc[[index], :]

#     label = 'Total'

#     total = row.total

#     row = row.loc[:, [2012,2013,2014,2015,2016]].melt()

#     print(row)

#     año = row['index']

#     valor = row.value

#     ax.plot(año, valor, label=label, color=(0.204, 0.51, 0.51) if index < 15 else (0.34, 0.94, 0.168))

#     plt.legend(['Crecimiento', 'Decrecimiento'])

    

plt.xticks([2012,2013,2014,2015,2016])

#plt.yticks([0,10,20,30,40,50,60,70,80,90,100])

ax.set_xlabel("\n Año", fontsize=18)

ax.set_ylabel("Porcentaje de crecimiento \n", fontsize=18)

ax.legend(loc='best', title_fontsize=16, fontsize=14)   

ax.set_title("Comportamiento porcentual de todas las provincias \n", fontsize=20)

plt.legend(fontsize=14)



data.sort_values(by="total", ascending=False

                ).head(30)