import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
%matplotlib inline
x = np.linspace(0, 5, 13)

y = x ** 2
print('Datos de x: ', x)

print('Datos de y: ', y)
grafica = plt.plot(x, y)


# Crear Grafica

plt.plot(x, y, 'r') # 'r' is el color rojo

# Asignar Titulos

plt.xlabel('Titulo de X')

plt.ylabel('Titulo de Y')

plt.title('Titulo de la Grafica')



# plt.subplot(numero de filas, numero de columnas, numero de plot)





# Layout

plt.subplot(1,2,1)

# Gráfico

plt.plot(x, y, 'r--') # Es posible agregar estilo a las lineas aparte del color





# Layout

plt.subplot(1,2,2)

# Gráfico

plt.plot(y, x, 'g*-');
# plt.subplot(nÚmero de filas, nÚmero de columnas, lugar del gráfico)

plt.subplot(2,2,1)

plt.subplot(2,2,2)

plt.subplot(2,2,3)

plt.subplot(2,2,4)
fig = plt.figure()



# Add set of axes to figure

ax = fig.add_axes([0.05, 0.6, 1, 1]) 



t = np.arange(0.0, 1.0, 0.01)

s = np.sin(2*np.pi*t)

line, = ax.plot(t, s, color='blue', lw=3)



xtext = ax.set_xlabel('my xdata') 

ytext = ax.set_ylabel('my ydata')
fig = plt.figure()



axes1 = fig.add_axes([0.3, 0.1, 0.8, 0.8]) # Gráfico Base

axes2 = fig.add_axes([0.4, 0.5, 0.4, 0.3]) # Gráfico Insertado



# 1

axes1.plot(x, y, 'b')

axes1.set_xlabel('X_label_axes1')

axes1.set_ylabel('Y_label_axes1')

axes1.set_title('Axes 1 Title')



# 2

axes2.plot(y, x, 'r')

axes2.set_xlabel('X_label_axes2')

axes2.set_ylabel('Y_label_axes2')

axes2.set_title('Axes 2 Title')


fig, axes = plt.subplots()



# Usamos el axes para agregar la gráfica

axes.plot(x, y, 'r')

axes.set_xlabel('x')

axes.set_ylabel('y')

axes.set_title('title');
# Empty canvas of 1 by 2 subplots

fig, axes = plt.subplots(1, 2)
type(axes)
for ax in axes:

    ax.plot(x, y, 'b')

    ax.set_xlabel('x')

    ax.set_ylabel('y')

    ax.set_title('title')



# la siguiente función muestra el resultado de la iteración    

fig

fig, axes = plt.subplots(nrows=1, ncols=2)



for ax in axes:

    ax.plot(x, y, 'g')

    ax.set_xlabel('x')

    ax.set_ylabel('y')

    ax.set_title('title')



fig    

plt.tight_layout()
fig = plt.figure(figsize=(8,4), dpi=100)
fig, axes = plt.subplots(figsize=(8,3), dpi = 150)



axes.plot(x, y, 'r')

axes.set_xlabel('x')

axes.set_ylabel('y')

axes.set_title('title');
fig = plt.figure()



ax = fig.add_axes([0,0,1,1])



ax.plot(x, x**2, 'b.-', label="x**2")

ax.plot(x, x**3, 'g-.', label="x**3")





Legend = ax.legend(loc=0)



fig = plt.figure()



ax = fig.add_axes([0,0,1,1])



ax.plot(x, x**2, 'b.-', label="x**2")

ax.plot(x, x**3, 'g-.', label="x**3")

ax.plot(x, x**2/3, 'c+')

ax.plot(x, x**4, 'rH')

ax.plot(x, x**5/3, 'm|')



Legend = ax.legend(loc=0)
fig, ax = plt.subplots()



linea_1 = ax.plot(x, x+1, color="blue", alpha=0.1) # alpha ajusta la transparencia

linea_2 = ax.plot(x, x+2, color="#8B008B")         

linea_3 = ax.plot(x, x+3, color="#FF8C00")         
fig, ax = plt.subplots(figsize=(12,6))



ax.plot(x, x+1, color="red", linewidth=0.25)



# Tipos de lineas ‘-‘, ‘–’, ‘-.’, ‘:’, ‘steps’

ax.plot(x, x+3, color="green", lw=3, linestyle='-')

ax.plot(x, x+4, color="green", lw=3, ls=':')



# Simbolos = '+', 'o', '*', 's', ',', '.', '1', '2', '3', '4', ...

ax.plot(x, x+ 6, color="blue", lw=3, ls='-', marker='+')

ax.plot(x, x+7, color="blue", lw=3, ls='--', marker='o')

ax.plot(x, x+8, color="blue", lw=3, ls='-', marker='*')





# marker size and color

ax.plot(x, x+10, color="purple", lw=1, ls='-', marker='o', markersize=2)

ax.plot(x, x+11, color="purple", lw=1, ls='-', marker='o', markersize=4)

ax.plot(x, x+12, color="purple", lw=1, ls='-', marker='o', markersize=8, markerfacecolor="red")

ax.plot(x, x+16, color="purple", lw=1, ls='-', marker='s', markersize=8, 

        markerfacecolor="yellow", markeredgewidth=3, markeredgecolor="b");
fig, axes = plt.subplots(1, 3, figsize=(12, 4))



axes[0].plot(x, x**2, x, x**3)

axes[0].set_title("Rangos por defult")



axes[1].plot(x, x**2, x, x**3)

axes[1].axis('tight')

axes[1].set_title("Ajuste automático")



axes[2].plot(x, x**2, x, x**3)

axes[2].set_ylim([0, 60])

axes[2].set_xlim([2, 5])

axes[2].set_title("Rango personalizado");
plt.scatter(x,y)
data = np.random.randint(2, 20, 30)

p = plt.hist(data)
data = [np.random.normal(0, std, 100) for std in range(1, 4)]

plt.boxplot(data,vert=True,patch_artist=True); 