# Cargar librerías
import os
import pandas as pd
import numpy as np

# Necesario para presentar los gráficos Jupyter Notebook
%matplotlib inline

# Requerido para realizar gráficos simples
import matplotlib.pyplot as plt

# Requerido para dar formato de fecha más adelante
import datetime
import matplotlib.dates as mdates

# Requerido para versiones recientes de matplotlib
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Requerido para mostrar una imagen en el notebook
from IPython.display import Image

# Gráficos avanzados usando seaborn
import seaborn as sns
sns.set(style="whitegrid") # Se puede definir el estilo según preferencias
# Carga de datos en Python
energy_consumption_file = os.path.join(os.getcwd(),'energy_consumption.csv')        
electricity_generation_file = os.path.join(os.getcwd(),'electricity_generation.csv')

energy_df = pd.read_csv(energy_consumption_file)
electricity_df = pd.read_csv(electricity_generation_file)
# Vistazo a los datos de consumo de energía
energy_df.head()
# Obtener todas las descripciones únicas registradas
print(energy_df['Description'].unique())
# Vistazo a los datos de generación de energía eléctrica
electricity_df.head()
# Obtener todas las descripciones únicas registradas
print(electricity_df['Description'].unique())
# Extraer el mes
energy_df['MM'] = energy_df['YYYYMM'].apply(lambda x: int(str(x)[-2:]))
electricity_df['MM'] = electricity_df['YYYYMM'].apply(lambda x: int(str(x)[-2:]))

# Extraer el año
energy_df['YYYY'] = energy_df['YYYYMM'].apply(lambda x: int(str(x)[:-2]))
electricity_df['YYYY'] = electricity_df['YYYYMM'].apply(lambda x: int(str(x)[:-2]))
# Renonmbrar las descripciones de los datos de energía
energy_short_dict = {'Primary Energy Consumed by the Commercial Sector': 'PEC Commercial Sector',
              'Primary Energy Consumed by the Electric Power Sector': 'PEC Electric Power Sector',
              'Primary Energy Consumed by the Industrial Sector': 'PEC Industrial Sector',
              'Primary Energy Consumed by the Residential Sector': 'PEC Residential Sector',
              'Primary Energy Consumed by the Transportation Sector': 'PEC Transportation Sector',
              'Primary Energy Consumption Total': 'PEC Total',
              'Total Energy Consumed by the Commercial Sector': 'TEC Commercial Sector',
              'Total Energy Consumed by the Industrial Sector': 'TEC Industrial Sector',
              'Total Energy Consumed by the Residential Sector': 'TEC Residential Sector',
              'Total Energy Consumed by the Transportation Sector': 'TEC Transportation Sector'}

# Limpiar nombres acortando la descripción
clean_energy_df = energy_df.copy()
clean_energy_df['Description'] = clean_energy_df['Description'].apply(lambda x: energy_short_dict[x])
clean_energy_df.head()
# Renombrar descripciones de los datos de generación
electricity_short_dict = {'Electricity Net Generation From Coal, All Sectors': 'ENG Coal',
                          'Electricity Net Generation From Conventional Hydroelectric Power, All Sectors': 'ENG HE Power',
                          'Electricity Net Generation From Geothermal, All Sectors': 'ENG Geothermal',
                          'Electricity Net Generation From Hydroelectric Pumped Storage, All Sectors': 'ENG HE Pumped Storage',
                          'Electricity Net Generation From Natural Gas, All Sectors': 'ENG Natural Gas',
                          'Electricity Net Generation From Nuclear Electric Power, All Sectors': 'ENG Nuclear Electric Power',
                          'Electricity Net Generation From Other Gases, All Sectors': 'ENG Other Gases',
                          'Electricity Net Generation From Petroleum, All Sectors': 'ENG Petroleum',
                          'Electricity Net Generation From Solar, All Sectors': 'ENG Solar',
                          'Electricity Net Generation From Waste, All Sectors': 'ENG Waste',
                          'Electricity Net Generation From Wind, All Sectors': 'ENG Wind',
                          'Electricity Net Generation From Wood, All Sectors': 'ENG Wood',
                          'Electricity Net Generation Total, All Sectors': 'ENG Total'}

# Limpiar los nombres acortando la descripción
clean_electricity_df = electricity_df.copy()
clean_electricity_df['Description'] = clean_electricity_df['Description'].apply(lambda x: electricity_short_dict[x])
clean_electricity_df.head()
# Definir las categorías de consumo y generación que nos interesan
consume_category = 'PEC Electric Power Sector'
generate_category = 'ENG Nuclear Electric Power'

# Seleccionar el sector de energía eléctrica en consumo de energía
consume_df = clean_energy_df[clean_energy_df['Description'] == consume_category][['YYYYMM','Value']].reset_index(drop=True)

# seleccionar la generación de energía nuclear para todos los sectores
generate_df = clean_electricity_df[clean_electricity_df['Description'] == generate_category][['YYYYMM','Value']].reset_index(drop=True)

# Unificar en un solo dataframe para graficar
merged_df = pd.merge(consume_df, generate_df, how='left', on=['YYYYMM'], suffixes=('_CONSUME','_GENERATE'))

merged_df.head()
# Crear un diagrama de dispersión básico para ver la relación de las 2 variables
plt.figure(figsize=(15, 7))
plt.scatter(merged_df['Value_GENERATE'], merged_df['Value_CONSUME'], s=1)
plt.title('Nuclear Electric Power Analysis');
plt.xlabel(generate_category);
plt.ylabel(consume_category);
consume_category = 'PEC Commercial Sector'
generate_category = 'ENG Nuclear Electric Power'

# seleccionar energía eléctrica dentro de las categorías de consumo
consume_df = clean_energy_df[clean_energy_df['Description'] == consume_category][['YYYYMM','Value']].reset_index(drop=True)

# seleccionar todos los sectores de generación de energía eléctrica
generate_df = clean_electricity_df[clean_electricity_df['Description'] == generate_category][['YYYYMM','Value']].reset_index(drop=True)

# Unificar en un solo dataframe para graficar
merged_df = pd.merge(consume_df, generate_df, how='left', on=['YYYYMM'], suffixes=('_CONSUME','_GENERATE'))

# Diagrama de dispersión básico
plt.figure(figsize=(15, 7))
plt.scatter(merged_df['Value_GENERATE'], merged_df['Value_CONSUME'], s=1)
plt.title('Nuclear Electric Power Analysis');
plt.xlabel(generate_category);
plt.ylabel(consume_category);
consume_category = 'PEC Electric Power Sector'
generate_category = 'ENG Nuclear Electric Power'

# seleccionar energía eléctrica dentro de las categorías de consumo
consume_df = clean_energy_df[clean_energy_df['Description'] == consume_category][['YYYYMM','Value']].reset_index(drop=True)

# seleccionar todos los sectores de generación de energía eléctrica
generate_df = clean_electricity_df[clean_electricity_df['Description'] == generate_category][['YYYYMM','Value']].reset_index(drop=True)

# Unificar en un solo dataframe para graficar
merged_df = pd.merge(consume_df, generate_df, how='left', on=['YYYYMM'], suffixes=('_CONSUME','_GENERATE'))
# Gráfico de líneas de consumo de energía a través del tiempo
plt.figure(figsize=(15,3))
plt.plot(merged_df['Value_CONSUME'])
plt.title('Gráfico de líneas: ' + consume_category)
plt.xlabel('Índice');
plt.ylabel(consume_category);

# Gráfico de líneas de generación de energía a través del tiempo
plt.figure(figsize=(15,3))
plt.plot(merged_df['Value_GENERATE'])
plt.title('Gráfico de líneas: ' + generate_category)
plt.xlabel('Índice');
plt.ylabel(generate_category);
# Convertir la cadena YYYYMM a formato datetime
merged_df['YYYYMM_dt'] = merged_df['YYYYMM'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y%m"))
merged_df.head()
# Gráfico mejorado de consumo de energía en el tiempo
fig, ax = plt.subplots(figsize=(15,3))
ax.plot(merged_df['YYYYMM_dt'], merged_df['Value_CONSUME'])
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')) # formatear el eje X
plt.title('Gráfico de líneas mejorado: ' + consume_category)
plt.xlabel('Fecha');
plt.ylabel(consume_category);

# Gráfico mejorado de generación de energía en el tiempo
fig, ax = plt.subplots(figsize=(15,3))
ax.plot(merged_df['YYYYMM_dt'], merged_df['Value_GENERATE'])
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')) # formatear el eje X
plt.title('Gráfico de líneas mejorado: ' + generate_category)
plt.xlabel('Fecha');
plt.ylabel(generate_category);
fig, ax = plt.subplots(figsize=(15,3))
ax.plot(merged_df['YYYYMM_dt'], merged_df['Value_CONSUME'].pct_change())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')) # formatear eje X
plt.title('Cambio porcentual mensual de consumo de energía')
plt.xlabel('Fecha');
plt.ylabel(consume_category);
# Adicionar la característica 'Ratio' (proporción): energía consumida / energía generada
merged_df['Ratio'] = merged_df['Value_CONSUME'] / merged_df['Value_GENERATE']
merged_df['MM'] = merged_df['YYYYMM'].apply(lambda x: int(str(x)[-2:])) # Añadir el mes para agrupar en gráfico de cajas
merged_df['Ratio'].describe()
plt.hist(merged_df['Ratio'], bins=50);
# Selecionar meses a usar
unique_months = [1,2,3,4,5,6,7,8,9,10,11,12]


# Ejecutar ciclo sobre todos los meses y guardar cada dataframe en una lista
df_list = []
for month_int in unique_months:
    temp_df = merged_df[merged_df['MM'] == month_int][['Ratio']].reset_index(drop=True) # Seleccionar mes
    temp_df = temp_df.rename(columns={'Ratio':'Prop_'+str(month_int)}) # renombrar para el gráfico
    df_list.append(temp_df) # guardar para concatenar después

# Consolidar datos
plot_df = pd.concat(df_list, axis=1)

# Diagramas de caja
fig, ax = plt.subplots(figsize=(15,5))
plot_df.boxplot(ax=ax, showfliers=False)
ax.set_xlabel('Mes del Año');
ax.set_ylabel('Proporción');
ax.set_title('Distribución de Proporción por Mes');
# Añadir un marcador para los meses pico
customized_df = merged_df.copy()
customized_df['ES_PICO'] = customized_df['YYYYMM'].apply(lambda x: 'PICO' if str(x)[-2:] in ['07','08'] else 'NO PICO')

sns.pairplot(customized_df, hue='ES_PICO', x_vars=['YYYYMM','Value_CONSUME','Value_GENERATE','Ratio'], y_vars=['YYYYMM','Value_CONSUME','Value_GENERATE','Ratio'], plot_kws={'s':10});
# Extraer año para ser usado en el mapa de calor
customized_df['YYYY'] = customized_df['YYYYMM'].apply(lambda x: str(x)[:-2])

# Crear tabla "dinámica" (formatear los datos para hacer más fácil su visualización en meses y años)
pivot_elec_df = customized_df.pivot('MM','YYYY','Value_GENERATE')
pivot_ener_df = customized_df.pivot('MM','YYYY','Value_CONSUME')
# Mapa de calor de energía consumida por mes y año
fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(pivot_ener_df, cmap="coolwarm", ax=ax);
ax.set_title('PEC Electric Power Sector');

# Mapa de calor de energía generada por mes y año
fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(pivot_elec_df, cmap="coolwarm", ax=ax);
ax.set_title('ENG Nuclear Electric Power');
pivot_radio_df = customized_df.pivot('MM','YYYY','Ratio')

# Mapa de calor de la proporción por mes y año
fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(pivot_radio_df, cmap="coolwarm", ax=ax);
ax.set_title('Proporción en el Tiempo');
Image("EjemploCajas.png")
# Definir tamaño del gráfico, crear diagrama
fig, ax = plt.subplots(figsize=(15,8))
m = sns.boxplot(x="YYYY",y="Value_CONSUME",hue='ES_PICO',data=customized_df,orient='vertical',showfliers=False)

# Format plot
plt.legend(loc='upper left')
plt.title('Análisis de Consumo de Meses Pico')
plt.xticks(rotation=90);
plt.xlabel('Año')
plt.ylabel('Consumo de Electricidad: PEC Electric Power Sector');
# Diagrama de cajas de consumo de energía de diversos sectores
fig, ax = plt.subplots(figsize=(15,5))
sns.boxplot(x="Description", y="Value", data=clean_energy_df, palette="Set3", ax=ax)
plt.xticks(rotation=90);
plt.title('Distribución del Consumo de Energía por Sector');
# Crea un diagrama de cintas de los datos para una vista desagregada
fig, ax = plt.subplots(figsize=(15,5))
m = sns.stripplot(x="Description", y="Value", data=clean_energy_df, palette="Set3", s=1, ax=ax)
plt.xticks(rotation=90);
plt.title('Distribuciones del Consumo de Energía por Sector');
# Construir dataframe para graficas
plot_df = clean_energy_df.copy()
plot_df['YYYYMM_dt'] = plot_df['YYYYMM'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y%m"))
plot_df.head()
# Crear nueva carpeta para guardar los gráficos
plot_dir = os.path.join(os.getcwd(), 'Graficos')

# Solo crear la carpeta si no existe aún
if not(os.path.isdir(plot_dir)):
    os.mkdir(plot_dir) # crea la nueva carpeta
# One can quickly generate and save plots with ease in Juypter
unique_desc = sorted(plot_df['Description'].unique())
for i in unique_desc:
    fig, ax = plt.subplots(figsize=(15,4))
    temp_df = plot_df[plot_df['Description'] == i]
    ax.plot(temp_df['YYYYMM_dt'], temp_df['Value'])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y%m')) # format x-axis display

    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.title('Descripción de Datos: ' + str(i))
    plt.tight_layout()
    
    # Guardar como PNG
    file_name = 'Grafico ' + str(i) + '.png'
    print("Guardando: " + file_name)
    fig.savefig(os.path.join(plot_dir,file_name)) # Guardar en png (en el directorio plot_dir)
    plt.close(fig) # No genere los gráficos en el notebook (muchos gráficos pueden ocupar mucha memoria RAM)
# Ejemplo de uno de los diagramas de cajas para 2000
Image("DiagramaCajas_2000.png")
# Crear nueva carpeta
boxplot_dir = os.path.join(os.getcwd(), 'DiagramaCajasAnual')

# Solo crear la carpeta si no existe aún
if not(os.path.isdir(boxplot_dir)):
    os.mkdir(boxplot_dir) # crea la nueva carpeta

# Obtener lista de años para el ciclo
unique_years = sorted(clean_energy_df['YYYY'].unique())

for i in unique_years:
    # Diagrama de cajas de los consumos de energía por sector
    fig, ax = plt.subplots(figsize=(15,10))
    sns.boxplot(x="Description", y="Value", data=clean_energy_df[clean_energy_df['YYYY'] == i], palette="Set3", ax=ax)
    plt.xticks(rotation=90);
    plt.title('Distribuciones de Consumo de Energía por Sector')
    plt.tight_layout()
    
    # Guardar en png
    file_name = 'DiagramaCajas_' + str(i) + '.png'
    print("Guardando: " + file_name)
    fig.savefig(os.path.join(boxplot_dir,file_name)) # guardar en png (en la carpeta plot_dir)
    plt.close(fig) # No genere los gráficos en el notebook (puede requerir demasiada memoria)