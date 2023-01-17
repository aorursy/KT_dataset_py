import matplotlib.pyplot as plt 

import numpy as np

x=np.arange(0,10,0.2)

y=np.sin(x)

# Generamos el gráfico

plt.plot(x,y) 

# Mostramos el gráfico

plt.show()
# generamos el gráfico 

fig=plt.figure() # genera la figura 

# add_subplot(221) x filas, y columnas celda z 

ax=fig.add_subplot(111) # adiciona un eje a esa figura 

ax.plot(x,y) # adiciona gráfico al eje

plt.show()
z=np.cos(x)

fig=plt.figure() # genera la figura 

ax=fig.add_subplot(121) # adiciona un eje a esa figura 

ax.plot(x,y) # adiciona gráfico de la función seno al eje 

ax=fig.add_subplot(122) # adiciona un eje a esa figura 

ax.plot(x,z,'r'); # adiciona grafico al otro eje 
from pylab import * 

x=arange(0,10,0.2) 

y=sin(x)

plot(x,y)

show()
# Import the required packages

import matplotlib.pyplot as plt

import numpy as np

# se genera la figura y los ejes 

fig , axs = plt.subplots(nrows=2,ncols=1) 

# en el primer eje se grafica el seno y se pone etiqueta al eje y 

axs[0].plot(x,y)

axs[0].set_ylabel('Seno')

axs[0].set_xlabel('valores x') 



# en el segundo eje se grafica el coseno y se pone etiqueta al eje y 

axs[1].plot(x,z,'r')

axs[1].set_ylabel('Coseno')

axs[1].set_xlabel('valores x') 

# se muestra el gráfico resultante 

plt.show()
#import matplotlib.pyplot as plt

fig= plt.figure()

fig.add_subplot(221)   #top left

fig.add_subplot(222)   #top right

fig.add_subplot(223)   #bottom left

fig.add_subplot(224)   #bottom right 

plt.show()
ax1=plt.subplot(2, 2, 1)
ax2=plt.subplot(222, frameon=False)
plt.subplot(224, sharex=ax1, facecolor='red')
np.random.randn(10, 3)

#x.shape()