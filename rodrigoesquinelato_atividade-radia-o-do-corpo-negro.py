'''.. :codeauthor:: Rodrigo Esquinelato <resquinelato@gmail.com>'''

h = 6.6207015e-34
# h[J/s] constante de planck
c = 299792458.00
# c[m/s] velocidade da luz
kB = 1.38649e-23
# kB[J/K] contante be Boltzmann
T=5800
T2 = 7300

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


xs = np.arange(1e-8, 1.6e-6, 1e-11)
ys = ((2*h)*(c**2))/((xs**5)*((np.exp((h*c)/(xs*kB*T))) - 1))
ys2 = ((2*h)*(c**2))/((xs**5)*((np.exp((h*c)/(xs*kB*T2))) - 1))
dcnr={ ((2*h)*(c**2))/((i**5)*(np.exp((h*c)/(i*kB*T)) - 1)):i  for i in xs}
y_max = max(dcnr.keys())
x_max = dcnr[y_max]

fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(14,7))

ax.set_xscale('linear')
ax2.set_xscale('linear')

ax.set_ylim([-0.5e+12, 31.5e+12])
ax.set_xlim([-1e-8, 1.601e-6])
ax2.set_ylim([-1.5e+12, 100e+12])
ax2.set_xlim([-1e-8, 1.601e-6])

# Teste de formatação via outros códigos
ax.set_title('Radiação do Corpo Negro\nLei de Plank\nT = 5800K',fontsize=15)
ax2.set_title('Radiação do Corpo Negro\nLei de Plank\nT = 7300K',fontsize=15)
def nm(x, pos):
    return '%1.f nm' % (x*1e+9)
def Jm3(x,pos):
    return '%1.f t' % (x*1e-12)
ax.xaxis.set_major_formatter(FuncFormatter(nm))
ax.yaxis.set_major_formatter(FuncFormatter(Jm3))
ax2.xaxis.set_major_formatter(FuncFormatter(nm))
ax2.yaxis.set_major_formatter(FuncFormatter(Jm3))


ax.plot(xs, ys,linestyle='solid',linewidth=4, c="#BCFF00")
ax.plot(xs, ys,linestyle='solid',linewidth=0.5)
ax.plot(x_max,y_max, 'o')
ax2.plot(xs, ys2,linestyle='solid',linewidth=4)
ax2.plot(xs, ys2,linestyle='solid',linewidth=0.5)

ax.set_xlabel('Comprimento de onda [m]',fontsize=15)
ax.set_ylabel('Radiância Espectral B(λ)T',fontsize=15)
ax2.set_xlabel('Comprimento de onda [m]',fontsize=15)
ax2.set_ylabel('Radiância Espectral B(λ)T',fontsize=15)
ax.annotate('{} nm , {} t'.format(round(x_max*1e9,3),round(y_max*1e-12,2)), xy=(x_max, y_max), xytext=(x_max-2e-7, y_max+1e+12), fontsize=15)
ax.grid(linestyle="--", linewidth=0.8, color='.5')
ax2.grid(linestyle="--", linewidth=0.8, color='.5')

#plt.yticks(ticks=np.arange(0, 32e+12, 4e+12),fontsize=13)
#plt.xticks(ticks=np.arange(0, 1.601e-6,2e-7),rotation=-5,fontsize=13)
plt.tight_layout()
plt.show()

