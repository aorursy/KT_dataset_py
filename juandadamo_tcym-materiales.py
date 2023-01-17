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

madera = ht.nearest_material('wood')
acero = ht.nearest_material('steel')

print('El coeficiente de conducción de la madera es %.3f W/m K'%ht.k_material(madera))
print('El coeficiente de conducción del acero es %.3f W/ m K'%ht.k_material(acero))
a_madera = ht.k_material(madera)/(ht.rho_material(madera)*ht.Cp_material(madera))
a_acero = ht.k_material(acero)/(ht.rho_material(acero)*ht.Cp_material(acero))
print('La difusividad térmica de la madera es %.2g m2/s'%a_madera)
print('La difusividad térmica del acero es %.2g m2/s'%a_acero)
aire = ht.nearest_material('Gases, air')
aire
ht.k_material(aire,T=600)
ht.rho_material(aire)
ht.Cp_material(aire)
