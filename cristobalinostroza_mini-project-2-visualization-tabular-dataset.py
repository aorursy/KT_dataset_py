import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Llamar a dataset tabular videoconferencia.csv  
df = pd.read_csv("../input/short-tabular-data/videoconferencia.csv")

# Se aplica formato de matriz al dataset con la función Pivot 
df_vc = df.pivot(index="Día", columns="Semana", values="Horas")

# Se ordenan las filas (index) del dataset según orden indicado por lista "Dias_Ordenados" 
Dias_Ordenados = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
df_vc = df_vc.reindex(Dias_Ordenados)

# Se visualiza el dataset en formato matricial para cada día de la semana
df_vc.head(len(df_vc))
# Se personaliza el fondo del gráfico (con el fin de mejorar el constraste) y se especifican sus dimensiones con plt.rcParams
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (20.0, 10.0)

# Se genera mapa de calor personalizado por:
# - annot: específica si se escribe o no el valor en la celda
# - fmt: específica el formato de escritura del valor (d para valores enteros)
# - annot_kws: específica argumentos para el texto de las anotaciones, como el tamaño
# - linewidths: para marcar espaciado entre las celdas
# - cbar_kws: especifica parámetros de la barra de color, en este caso la orientación de la barra
# - cmap: especifica la paleta de colores de la visualización, en este caso usamos una paleta "secuencial"
ax = sns.heatmap(df_vc, annot=True, fmt="d", annot_kws={'size':15}, linewidths=1.0, cbar_kws={"orientation": "horizontal"}, cmap='Blues')
ax.set_title('Horas de Llamadas con Paleta de Color Secuencial "Blues"', pad=20, fontsize=20, fontweight="bold")
plt.show()
# - cmap: en este caso usamos una paleta "divergente"
ax = sns.heatmap(df_vc, annot=True, fmt="d", annot_kws={'size':15}, linewidths=1.0, cbar_kws={"orientation": "horizontal"}, cmap='BrBG')
ax.set_title('Horas de Llamadas con Paleta de Color Divergente "BrBG"', pad=20, fontsize=20, fontweight="bold")
plt.show()
# Identificar valor mínimo y máximo de dataset matricial 
# Resultado: Mínimo corresponde al primer domingo y el Máximo al último lunes (la visualizacion secuencial lo valida)
print(df_vc.min().min())
print(df_vc.max().max())
# Llamar a dataset titanic son Seaborn
df_titanic = pd.read_csv("../input/short-tabular-data/titanic.txt")

# Se visualizan los primeros 10 registros del dataset (Nos interesa el atributo class y survived)
df_titanic.head(10)
# Se crea una figura en matplotlib para poner el gráfico de tamaño 5x5 pulgadas y se agrega un gráfico con add_subplot()
plt.style.use('default')
fig = plt.figure(figsize=(5, 5))
plot = fig.add_subplot()

#Se filtran los datos para cada clase y luego se cuentan los pasajeros en cada uno formando una lista 
c1 = df_titanic[df_titanic["class"]=='First']
c2 = df_titanic[df_titanic["class"]=='Second']
c3 = df_titanic[df_titanic["class"]=='Third']
clases = [len(c1),len(c2),len(c3)]

# Se crea un gráfico de torta con el método pie y se le entrega la lista de valores (clase de cada pasajero)
# El primer output del método es una lista de cada segmento circular (patches), en este caso, son 3 segmentos (Clases)
# Se añade el argumento autopct, el cual calcula el porcentaje de pasajeros por segmento con 2 cifras sigificativas en este caso
pie_plot = plot.pie(clases, labels=['First','Second','Third'] ,colors=['navy','lime','red'], autopct='%1.2f%%',startangle=90)

#Se asigna un título al gráfico y se demarca en blanco para el segmento de color mas oscuro 
plot.set_title("Pasajeros del Titanic por Clase", fontsize=15, fontweight="bold")
pie_plot[2][0].set_color('white')
plt.show()
# Se generan 1 figura con 2 subplots. Con cada subplot de tamaño personalizable
fig1, (ax1, ax2) = plt.subplots(1,2,figsize=(10,10))

# Se filtran los datos de los pasajeros que NO sobrevivieron utilizando el dataframe previamente filtrado por clases
c1_0 = c1[c1["survived"]==0]
c2_0 = c2[c2["survived"]==0]
c3_0 = c3[c3["survived"]==0]
no_survived = [len(c1_0),len(c2_0),len(c3_0)]

# Se filtran los datos de los pasajeros que SI sobrevivieron utilizando el dataframe previamente filtrado por clases
c1_1 = c1[c1["survived"]==1]
c2_1 = c2[c2["survived"]==1]
c3_1 = c3[c3["survived"]==1]
survived = [len(c1_1),len(c2_1),len(c3_1)]

# Se genera el gráfico para cada subplot manteniendo los colores y propiedades
pie_plot1 = ax1.pie(no_survived, labels=['First','Second','Third'] ,colors=['navy','lime','red'], autopct='%1.2f%%',startangle=90)
pie_plot2 = ax2.pie(survived, labels=['First','Second','Third'] ,colors=['navy','lime','red'], autopct='%1.2f%%',startangle=90)

# Se asigna un título general a los subplot y se demarca en blanco para el segmento de color mas oscuro 
fig1.suptitle("Pasajeros del Titanic por Supervivencia (0/1) y por Clase (T/F/S)", fontsize=15, fontweight="bold")
fig1.subplots_adjust(top=1.25)
pie_plot1[2][0].set_color('white')
pie_plot2[2][0].set_color('white')

# Se agrega un título a cada subplot y se muestra sólo el gráfico
ax1.title.set_text('0: No Sobrevivientes')
ax2.title.set_text('1: Sobrevivientes')
plt.show()
# Llamar a dataset tabular peliculas.csv  
df_pel = pd.read_csv("../input/short-tabular-data/peliculas.csv")

# Se visualizan los primeros 10 registros del dataset y sus atributos
df_pel.head(20)
# Generar una grilla en donde los argumentos son:
# data: indicar de donde provienen los datos
# col: indicar qué columna utilizar para definir cuantos gráficos hay
# hue: indicar qué columna utilizar para definir los colores de cada gráfico
# col_wrap: indicar cuantas columnas pueden haber.
# Esta función solo genera los gráficos sin datos, ahora falta indicar con qué llenar cada gráfico
grid = sns.FacetGrid(data=df_pel, col="género", hue="género", col_wrap=3)

# Llamamos al método MAP que se encarga de llenar cada gráfico de la grilla. Sus argumentos son:
# Cada gráfico se llenara en este caso con línea y marcas para la cantidad de estrenos por año en cada género 
# Coordenada en el eje X será el años de estreno
# Coordenada en el eje Y será la cantidad de estrenos
# marker: indicar cómo se va a marcar cada punto del gráfico. En este caso usaremos asterisco
grid.map(plt.plot, "año", "cantidad", marker="*")

# Visualizar el gráfico
plt.show()
# modificamos el atributo categórico para este caso al año de estreno
grid2 = sns.FacetGrid(data=df_pel, col="año", hue="año", col_wrap=3)

# Para este caso usamos la función BARH de matplotlib con la categoría género como EJE Y y la cantidad como EJE X
grid2.map(plt.barh, "género", "cantidad", height=0.8)
grid2.set_axis_labels("Cantidad","Género");

# Visualizar el gráfico
plt.show()
