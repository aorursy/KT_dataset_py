# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import warnings

warnings.simplefilter('ignore')

dataset = pd.read_csv('../input/wind-turbine-scada-dataset/T1.csv', 

                      parse_dates = ['Date/Time'], 

                      date_parser = lambda x: pd.to_datetime(x, format = '%d %m %Y %H:%M'), 

                      index_col = 'Date/Time')





dataset.info()



# transformar a frecuencia horaria y agregar mes



dataset_hora = dataset.resample('H').mean().fillna(0)



dataset_hora['mes'] = dataset_hora.index.strftime('%B')



dataset_hora['mes'] = pd.Categorical(dataset_hora['mes'], 

               categories = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],

              ordered = True)



dataset_hora.info()
fig, ax = plt.subplots(2,2, figsize = (15,7.5))

sns.distplot(dataset['LV ActivePower (kW)'], ax = ax[0,0])

sns.distplot(dataset['Wind Speed (m/s)'], ax = ax[0,1])

sns.distplot(dataset['Theoretical_Power_Curve (KWh)'], ax = ax[1,0])

sns.distplot(dataset['Wind Direction (°)'], ax = ax[1,1])

plt.tight_layout()

plt.show()
gb_velocidad = dataset.groupby(pd.qcut(dataset['Wind Speed (m/s)'], q = 25)).mean()



zona_0 = gb_velocidad['Wind Speed (m/s)'] < 4

zona_1 = (gb_velocidad['Wind Speed (m/s)'] >= 3.5) & (gb_velocidad['Wind Speed (m/s)'] <= 15)

zona_2 = gb_velocidad['Wind Speed (m/s)'] >= 14.5



# dataset['Benchmark'] = (dataset['LV ActivePower (kW)'].div(dataset['Theoretical_Power_Curve (KWh)'])).replace(np.inf, np.nan).replace(-np.inf, np.nan).fillna(0)



fig, ax = plt.subplots(1,2, figsize = (15,7.5))



# Power Curve grouped



# fill curve zones

ax[0].fill_between(gb_velocidad[zona_0]['Wind Speed (m/s)'], 0 , gb_velocidad[zona_0]['Theoretical_Power_Curve (KWh)'], color = 'gray')

ax[0].fill_between(gb_velocidad[zona_1]['Wind Speed (m/s)'], 0 , gb_velocidad[zona_1]['Theoretical_Power_Curve (KWh)'], color = 'lightgray')

ax[0].fill_between(gb_velocidad[zona_2]['Wind Speed (m/s)'], 0 , gb_velocidad[zona_2]['Theoretical_Power_Curve (KWh)'], color = '#F2F3F4')



#plot 

sns.lineplot(x = 'Wind Speed (m/s)', y = 'LV ActivePower (kW)', data = gb_velocidad, label = 'Real Power (kW)', marker = 'o', ax = ax[0])

sns.lineplot(x = 'Wind Speed (m/s)', y = 'Theoretical_Power_Curve (KWh)', data = gb_velocidad, label = 'Theoretical Power (kW)', marker = 'o', ax = ax[0])

ax[0].legend()



# Power Curve ungrouped

sns.scatterplot(x = 'Wind Speed (m/s)', y = 'LV ActivePower (kW)', data = dataset, label = 'Real Power (kW)', ax = ax[1])

sns.scatterplot(x = 'Wind Speed (m/s)', y = 'Theoretical_Power_Curve (KWh)', data = dataset, label = 'Theoretical Power (kW)', ax = ax[1])

ax[1].legend()





plt.show()
prod_acumulada = dataset_hora['LV ActivePower (kW)'].sort_values(ascending = False)

prod_acumulada = prod_acumulada.reset_index(drop = True)



plt.figure(figsize= (15,7.5))



sns.lineplot(data = prod_acumulada, linewidth = '5')



plt.fill_between( list(range(0, int(8760*0.25))) ,0, prod_acumulada[:int(8760*0.25)], color = 'gray')

plt.fill_between( list(range(int(8760*0.25), int(8760*0.5))), 0, prod_acumulada[int(8760*0.25):int(8760*0.5)], color = 'lightgray')

plt.fill_between( list(range(int(8760*0.5), int(8760*0.75))), 0, prod_acumulada[int(8760*0.5):int(8760*0.75)], color = '#F2F3F4')



plt.title('Potencia generada Vs Número de horas')

plt.ylabel('Potencia [kW]')

plt.xlabel('Horas en el año')

plt.show()
# Agregar mes



dataset['mes'] = dataset.index.strftime('%B')



dataset['mes'] = pd.Categorical(dataset['mes'], 

               categories = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],

              ordered = True)



# Datos agregados por mes: Produccion de energía por mes y factor de uso 



dataset_mes = dataset_hora.groupby('mes')



agg_mes = dataset_mes.agg( ActivePower_total_MW = ('LV ActivePower (kW)', 'sum'),

                TheoreticalPower_Total_MW = ('Theoretical_Power_Curve (KWh)', 'sum'),

                Velocidad_promedio = ('Wind Speed (m/s)', 'mean'),

                Direccion_promedio = ('Wind Direction (°)', 'mean')

               )

horas_uso = dataset_hora[dataset_hora['LV ActivePower (kW)'] > 0].groupby('mes')['LV ActivePower (kW)'].count()



# horas_uso = dataset[dataset['LV ActivePower (kW)'] > 0].groupby('mes')['LV ActivePower (kW)'].count()



total_horas = dataset_mes['LV ActivePower (kW)'].count()



factor_uso = pd.DataFrame(horas_uso/total_horas).rename(columns = {'LV ActivePower (kW)': 'Factor_uso'})



agg_mes = agg_mes.merge(factor_uso, left_index = True, right_index = True)



agg_mes.sort_index(inplace = True)



agg_mes['ActivePower_total_MW'] = agg_mes['ActivePower_total_MW']/1000 



agg_mes['TheoreticalPower_Total_MW'] = agg_mes['TheoreticalPower_Total_MW']/1000



agg_mes['Perdidas'] = agg_mes['TheoreticalPower_Total_MW'] - agg_mes['ActivePower_total_MW']



# Producción por mes

fig, ax = plt.subplots(2,1, figsize = (15,7.5))



# grafico superior



#top

sns.barplot(x = agg_mes.index , y =  'TheoreticalPower_Total_MW', data = agg_mes, color = 'red', ax = ax[0], label = 'perdidas')

#bottom

sns.barplot(x = agg_mes.index , y =  'ActivePower_total_MW', data = agg_mes, color = 'blue', ax = ax[0], label = 'Energia producida')

ax[0].legend()

ax[0].set_ylabel('Energia Producida_MWh')



# grafico inferior

sns.barplot(x = agg_mes.index, y = 'Factor_uso', data = agg_mes, ax=ax[1], color = 'gray', label = 'Porcentaje de operación')

ax[1].legend()

plt.show()



# velocidad promedio por mes

plt.figure(figsize = (15,7.5))

sns.boxplot(x = 'mes', y = 'Wind Speed (m/s)', data = dataset_hora, color = '#F2F3F4')

velocidad_promedio = dataset_hora['Wind Speed (m/s)'].mean()

sns.lineplot(x = list(range(0,12)), y = velocidad_promedio, color = 'black', label = f'Velocidad promedio {round(velocidad_promedio,1)} m/s')

plt.legend()

plt.show()
# Datos fuera de servicio



potencias = []

for i in range(0,110,10):

    

    umbral_potencia = i



    filtros = (dataset_hora['Wind Speed (m/s)'] <= 25) & (dataset_hora['Wind Speed (m/s)'] >= 3.5) & (dataset_hora['LV ActivePower (kW)'] <= umbral_potencia)



    fuera_servicio = dataset_hora[filtros]

    

    potencias.append(fuera_servicio['Theoretical_Power_Curve (KWh)'].sum())



# sns.lineplot(x = range(0,110,10), y = potencias)



# filtrando por fuera de servicio



filtros = (dataset_hora['Wind Speed (m/s)'] <= 25) & (dataset_hora['Wind Speed (m/s)'] >= 3.5) & (dataset_hora['LV ActivePower (kW)'] <= 0)



fuera_servicio = dataset_hora[filtros]



fuera_servicio['Delta_Prod'] = fuera_servicio['Theoretical_Power_Curve (KWh)'] - fuera_servicio['LV ActivePower (kW)']

    

fuera_servicio_gb = fuera_servicio.loc[:,['mes','Theoretical_Power_Curve (KWh)','Delta_Prod']].groupby('mes').sum()



fuera_servicio_gb['Theoretical_Power_Curve (KWh)'] = fuera_servicio_gb['Theoretical_Power_Curve (KWh)']/1000



fuera_servicio_gb['Delta_Prod'] = fuera_servicio_gb['Delta_Prod']/1000



# perdidas por mes



plt.figure(figsize =(15,7.5))

sns.barplot(x = fuera_servicio_gb.index, y = 'Theoretical_Power_Curve (KWh)', data = fuera_servicio_gb, color = 'gray')

plt.ylabel('Potencia perdida [MW]')

plt.title('Pérdidas por fuera de servicio')

plt.show()
# perdidas de energia turbina en operación

filtros = (dataset_hora['Wind Speed (m/s)'] <= 25) & (dataset_hora['Wind Speed (m/s)'] >= 3.5) & (dataset_hora['LV ActivePower (kW)'] > 0) 

turbina_op = dataset_hora[filtros]



turbina_op['Delta_prod'] = turbina_op['Theoretical_Power_Curve (KWh)'] - turbina_op['LV ActivePower (kW)']



# sns.lineplot(x = turbina_op.index, y =  'Delta_prod', data = turbina_op)

# plt.show()



turbina_op_gb = turbina_op.loc[:,['mes','Delta_prod']].groupby('mes').sum()



turbina_op_gb['Delta_prod'] = turbina_op_gb['Delta_prod']/1000



plt.figure(figsize =(15,7.5))

sns.barplot(x = turbina_op_gb.index, y = 'Delta_prod', data = turbina_op_gb, color = 'gray')

plt.title('Potencia perdida en operación [MW]')

plt.ylabel('MW')

plt.show()



# Composicion de pérdidas



perdidas_totales = turbina_op_gb['Delta_prod'] + fuera_servicio_gb['Delta_Prod'] 



plt.figure(figsize =(15,7.5))

plt.style.use('seaborn-white')

plt.stackplot(perdidas_totales.index, [turbina_op_gb['Delta_prod'],fuera_servicio_gb['Delta_Prod']], labels = ['Pérdidas en operación', 'Pérdidas por fuera de servicio'])

plt.title('Diferencias entre la potencia teórica y real por mes')

plt.ylabel('MW')

plt.legend()

plt.show()
def ratio_perd(mes): 

    

    turbina_op_mes = turbina_op[turbina_op['mes'] == mes]

    sns.boxplot(turbina_op_mes['Delta_prod']/turbina_op_mes['LV ActivePower (kW)'])

    plt.xlim(0,1)

    plt.show()

    

fig, ax = plt.subplots(4,3, sharex = True, figsize = (15, 7.5))



subplots = [item for sublist in ax for item in sublist]



meses = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']



fig.suptitle('Razón de pérdidas/Potencia esperada por mes')

for mes in meses: 

    axis = subplots.pop(0)

    turbina_op_mes = turbina_op[turbina_op['mes'] == mes]

    sns.boxplot(np.abs(turbina_op_mes['Delta_prod'])/turbina_op_mes['Theoretical_Power_Curve (KWh)'], ax  = axis)

    axis.set_title(mes)

    axis.set_xlim(0,1)

#     axis.set_ylim(0,10)

    axis.grid(linewidth=0.25)

    axis.spines['left'].set_visible(False)

    axis.spines['top'].set_visible(False)

    axis.spines['right'].set_visible(False)

    
shape = []

ratio_list = list(np.arange(0,1.1,0.1))

for i in ratio_list:

    ratio_lim  = np.abs(turbina_op['Delta_prod'])/turbina_op['Theoretical_Power_Curve (KWh)'] >= i

    shape.append(turbina_op[ratio_lim].shape[0])



fig, ax = plt.subplots(1,2, figsize=(15,7.5))

sns.lineplot(x = ratio_list, y = shape, marker = "o", markersize = 8, ax = ax[0])

ax[0].set_title('Número de datos filtrados for umbral de razón de pérdidas')

ax[0].set_xlabel('Razón de pérdidas')



for x,y in zip(ratio_list, shape): 

    

    label = "{:.0f}".format(y)

    ax[0].annotate(label, 

                (x,y), 

                textcoords = 'offset points', 

                xytext = (5,5),

                ha = 'left')

    

    

ratio_lim  = np.abs(turbina_op['Delta_prod'])/turbina_op['Theoretical_Power_Curve (KWh)'] >= 0.3

sns.barplot(x = turbina_op[ratio_lim].groupby('mes').count()['Delta_prod'], 

            y = turbina_op[ratio_lim].groupby('mes').count().index, 

            ax = ax[1], 

           color = 'lightgrey')

ax[1].set_title('Número de datos por encima del umbral de 30% por mes')

ax[1].set_xlabel('Conteo de operaciones filtradas')

plt.show()