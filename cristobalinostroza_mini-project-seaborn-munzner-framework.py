%matplotlib inline

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# V01: Visualización que permite ver la cantidad total de infectados por país
# Dataset: V01.csv 
df1 = pd.read_csv("../input/unreal-pandemic/V01.csv")
print(df1.head(10))
print(df1.shape)
# Ordeno la cantidad de contagios en orden decreciente (20 Paises) y luego agrupo por País sin volver a ordenar  
Contagios = df1.sort_values('Contagiados',ascending=0).groupby('Paises', sort=0)['Contagiados'].sum()
print(Contagios.head())

# Se codifica la visualización usando Matplotlib
ax = Contagios.plot.bar(color = 'navy', title = 'Contagiados por País a la fecha')
ax.set_xlabel("País afectado")
ax.set_ylabel("Cantidad de contagiados")
# V02: Visualización que permita ver la evolución del total de infectados en el tiempo (fecha).
# Dataset: V02.csv 
df2 = pd.read_csv("../input/unreal-pandemic/V02.csv")
print(df2.head(10))
print(df2.shape)
# Se codifica la visualización usando Seaborn y grilla cuadriculada
sns.set()
sns.set_style("whitegrid")
ax = sns.lineplot(x="Año", y="Total Contagiados", color="navy", data=df2, linewidth=2, marker = 's')

# Se configuran los límites, ticks y etiquetas de la codificación
ymax = int(df2[["Total Contagiados"]].max()+0.5e8)
ax.set_ylim(0,ymax)
ax.set_xticks(range(2013,2018))
ax.set_xticklabels(range(2013,2018))
ax.set_title("Evolución Total de Infectados en 5 Años")
plt.show()
# V03: Visualización que permita ver el total de infectados en el tiempo por país (fecha).
# Dataset: V03.csv 
df3 = pd.read_csv("../input/unreal-pandemic/V03.csv")
print(df3.head(10))
print(df3.shape)
#Funcion Melt pivotea columnas hacia filas (proceso inverso de funcion Pivot)
df_aux = pd.melt(df3, id_vars=['Pais'], value_vars=['2013','2014','2015','2016','2017'], var_name="Año", value_name="Total Infectados")
print(df_aux.head(),'\n\n', df_aux.tail())

# Esta función solo genera los gráficos sin datos, ahora falta indicar con qué llenar cada gráfico
grid = sns.FacetGrid(data=df_aux, col="Pais", hue="Pais", col_wrap=5, palette="muted")

# Método map se encarga de llenar cada gráfico de la grilla como codificacion de los datos usando small multiple de lineas y puntos
grid.map(plt.plot, "Año", "Total Infectados", marker="o")
plt.show()
# V04: Visualización que permite ver la distribución de número de contagios de países en función al tiempo de infección.
# Dataset: V04.csv 
df4 = pd.read_csv("../input/unreal-pandemic/V04.csv")
print(df4.head(10))
print(df4.shape)
# Se codifica con boxplot para visualizar la mediana, outliers y extremos estadísticos para cada mes de infección agregando cada país
plt.rcParams['figure.figsize'] = (20.0, 10.0)
ax2 = sns.boxplot(x="Mes", y="Contagiados", data=df4, palette="Set3")
ax2.set_yticks(range(0,11000,1000))
plt.show()
# V05: Dos visualizaciones (uno por hemisferio) que permita visualizar la cantidad de fallecimientos por mes y año. 
# De modo que permita determinar si hubo algunos meses en particular donde aumentó la cantidad de fallecidos o
# siempre se mantuvo de forma homogénea
# Datasets: V05-hem-norte.csv y V05-hem-sur.csv
df5n = pd.read_csv("../input/unreal-pandemic/V05-hem-norte.csv")
df5s = pd.read_csv("../input/unreal-pandemic/V05-hem-sur.csv")
print(df5n.head(10),'\n\n', df5n.shape, '\n\n', df5s.head(10), '\n\n', df5s.shape)
# Se manipula cada dataframe para dejarlo en formato matriz y poder codificarlo con función Heatmap
df5n = df5n.pivot(index="Año", columns="Mes", values="Fallecidos")
df5s = df5s.pivot(index="Año", columns="Mes", values="Fallecidos")
# Se codifica un mapa de calor con paleta de color divergente (Hemisferio Norte)
plt.rcParams['figure.figsize'] = (10.0, 5.0)
ax3 = sns.heatmap(df5n, annot=False, cmap='coolwarm', linewidths=1, square=True)
plt.show()
# Se codifica un mapa de calor con paleta de color divergente (Hemisferio Sur)
plt.rcParams['figure.figsize'] = (10.0, 5.0)
ax3 = sns.heatmap(df5s, annot=False, cmap='coolwarm', linewidths=1, square=True)
plt.show()
# V06: Dos visualizaciones que permitan mostrar la proporción de infectados por edad y la proporción de síntomas posibles.
# Dataset: V06.csv 
df6 = pd.read_csv("../input/unreal-pandemic/V06.csv")
print(df6.head(10),'\n\n', df6.shape, '\n\n', df6.tail(10))
# Se Cuentan los valores de cada categoría de Edad y de síntomas para calcular las proporciones
Sobre50 = df6[df6["Edad"]==">=50"].count()
Bajo18 = df6[df6["Edad"]=="<=18"].count()
Entremedio = df6["Edad"].count()-(Sobre50 + Bajo18)

s_base = df6[df6["Sintomas"]=="base"].count()
s_ext = df6[df6["Sintomas"]=="extendido"].count()
# Se codifica un gráfico de torta para visualizar las proporciones en 2 atributos diferentes (edad y síntomas detectados)
fig1, (a1, a2) = plt.subplots(1,2,figsize=(10,15))

# Se genera el gráfico para cada subplot manteniendo los colores y propiedades
pie_plot1 = a1.pie([Sobre50[0], Entremedio[0], Bajo18[0]], labels=[">=50 Años",">18 y <50 Años","<=18 Años"] ,colors=['navy','lime','red'], autopct='%1.2f%%',startangle=90)
pie_plot2 = a2.pie([s_base[0], s_ext[0]], labels=["Síntomas Base","Síntomas Extendido"] ,colors=['navy','lime','red'], autopct='%1.2f%%',startangle=90)

# Se asigna un título general a los subplot y se demarca en blanco para el segmento de color mas oscuro 
fig1.suptitle("Cantidad de Infectados por rango de edad y por tipo de síntomas", fontsize=15, fontweight="bold")
fig1.subplots_adjust(top=1.5)
pie_plot1[2][0].set_color('white')
pie_plot2[2][0].set_color('white')

# Se agrega un título a cada subplot y se muestra sólo el gráfico
a1.title.set_text('Proporción de Infectados por Edad')
a2.title.set_text('Proporción de Infectados por Tipo de Síntomas')
plt.show()
# V07: Visualización que permita apreciar distribución y potencialmente correlación entre pares de atributos de 
# dataset: edad, peso y altura; con categorización por tipos de síntomas presentados.
# Dataset: V07.csv 
df7 = pd.read_csv("../input/unreal-pandemic/V07.csv")
print(df7.head(10),'\n\n', df7.shape, '\n\n', df7.tail(10))
# Seleccionamos el atributo numérico de menor tamaño para transformarlo a categórico y codificarlo con el tamaño de la burbuja
print(len(df7["Edad"].unique()))
print(len(df7["Peso"].unique()))
print(len(df7["Altura"].unique()))

# Se pasa de atributos categórico a atributo numérico para codificar síntomas
#sintoma_num = df7.Sintomas.astype("category").cat.codes
# Se codifica un gráfico de burbujas con 2 atributos numéricos (eje x: altura, eje y: edad), 1 atributo categórico a numerico (sintoma) y 1 atributo numerico (peso)
# Por aspectos de percepción, el atributo peso puede ser más pre-atentivo para el usuario que la edad como canal de tamaño
plt.rcParams['figure.figsize'] = (20.0, 10.0)
sns.set_style("whitegrid")

markers = {"base": "s", "extendido": "o"}
axis = sns.scatterplot(x="Altura", y="Edad",hue="Sintomas", size="Peso", markers=markers, sizes=(20,250), alpha=1.0, cmap ='Set2',legend="brief", data=df7)

axis.set_title('Distribución y Correlación de Atributos Físicos \n (Tamaño de burbuja: Peso de Infectado) \n (Color: Síntoma) ')
axis.set_xlabel("Altura")
axis.set_ylabel("Edad")
plt.show()