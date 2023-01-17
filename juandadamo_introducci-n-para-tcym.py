!pip install ht;
# Definimos el tipo de documento, "matplotlib notebook"
# Importamos las librerías necesarias: numpy para los cálculos, 
# matplotlib para los gráficos.
# como la función erfc no está incluida en numpy, debe importarse desde otra librería
#%matplotlib notebook
%matplotlib inline
import ht as ht
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 14;
mpl.rcParams['ytick.labelsize'] = 14;
mpl.rcParams['font.family'] = 'serif';
# Función 
def Dirichlet1(T0,dT,t,x,alpha):
    T = T0 + dT*erfc(abs(x)/(2.*np.sqrt(alpha*t)))
    return T
### Cambios

# Condiciones y parámetros
T0 = 0.2             # Temperatura inicial
dT = 15.                      # Cambio de temperatura
t1 = 10.                       # Tiempo
t2 = 1000.                        # Tiempo 2  
xs = np.arange(0,0.10,0.001)    # Distancias
alpha_m = 9.19e-8              # Difusividad térmica. Pej, madera de pino

#########################################################################################

# Ejecuta la función
T = Dirichlet1(T0,dT,t1,xs,alpha_m)
T2 = Dirichlet1(T0,dT,t2,xs,alpha_m)




fig0,ax0 = plt.subplots(1);
# Gráfico. Definimos una figura fig0 y sus ejes ax0

fig0.set_size_inches((5,5));  #tamaño de la figura
ax0.plot(xs,T,label='t=%.0f'%t1)  # plot sobre los ejes
ax0.plot(xs,T2,label='t=%.0f'%t2)
ax0.set_xlim([min(xs),max(xs)]);#definimos límites
ax0.set_xlabel('distancia (m)',fontsize=18); #nombre para ordenadas
ax0.set_ylabel('Temperature ($^\circ$C)',fontsize=18); #nombre para abcisas
ax0.grid(True); #grilla de coordenadas
ax0.legend()

# función armónica en la superficie
def Dirichlet_armonica(T0,Ta,t,x,omega,alpha=1e-6):

    T = T0 + Ta * np.exp(-x*np.sqrt(np.pi*omega/(alpha))) * np.sin((2.*np.pi*omega*t)-x*np.sqrt(np.pi*omega/(alpha)))
    return T
# Constantes y parámetros
T0 = 10.                      # temperatura media
Ta = 1.                       # amplitud de la variación de temperatura
ts = np.linspace(0,1e1,200)   # tiempo
xs = np.arange(0,40e-3,1e-4)  # distancias
omega = 0.2                   # frecuencia (2 pi freq)
alpha_l = 1.5e-5              #difusividad térmica del latón
n_profiles = 5                # número de perfiles a plotear

#########################################################################################


fig1,ax1 = plt.subplots(1)
# Graficamos la condición de borde
fig1.set_size_inches((4,4))
#plt.title(r'Condición de borde')
ax1.plot(ts,T0+Ta*np.sin(2.*np.pi*omega*ts),'k');
ax1.set_ylabel('Temperatura en la superficie ($^\circ$C)',fontsize=12)
ax1.set_xlabel('Tiempo',fontsize=12);
#plt.tight_layout()

fig2,ax2 = plt.subplots(1,2,figsize=(10,4));
# Genera un loop para realizar los n perfiles
for t in ts[::len(ts)//n_profiles]:
    # evalúa la función
    T = Dirichlet_armonica(T0,Ta,t,xs,omega,alpha_l)
    # Plot
    ax2[0].plot(xs,T,label='%.2f'%(t))
    
ax2[0].legend(title='Tiempos (s)')
ax2[0].set_xlabel('Distancia (mm)');
ax2[0].set_ylabel('Temperature ($^\circ$C)');
ax2[0].set_xlim([min(xs),max(xs)]);
ax2[0].set_title('Perfiles para un tiempo dado')


# Gráfico de las series temporales

for x in xs[::len(xs)//n_profiles]:
    T = Dirichlet_armonica(T0,Ta,ts,x,omega,alpha_l)
    # Plot
    ax2[1].plot(T,ts,label='%0d'%(x*1000))
    
ax2[1].legend(title='Distancias (mm)',bbox_to_anchor=(1, 1))
ax2[1].set_xlabel('Temperatura ($^\circ$C)');
ax2[1].set_ylabel('Tiempo');
ax2[1].set_title('Series temporales para una distancia');

madera = ht.nearest_material('wood')
acero = ht.nearest_material('steel')

print('El coeficiente de conducción de la madera es %.3f W/m K'%ht.k_material(madera))
print('El coeficiente de conducción del acero es %.3f W/ m K'%ht.k_material(acero))
a_madera = ht.k_material(madera)/(ht.rho_material(madera)*ht.Cp_material(madera))
a_acero = ht.k_material(acero)/(ht.rho_material(acero)*ht.Cp_material(acero))
print('La difusividad térmica de la madera es %.2g m2/s'%a_madera)
print('La difusividad térmica del acero es %.2g m2/s'%a_acero)

