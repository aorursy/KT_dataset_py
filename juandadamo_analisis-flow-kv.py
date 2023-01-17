import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
from scipy import interpolate,optimize,integrate,signal
import matplotlib.pyplot as plt
import pylab,six,matplotlib 
from matplotlib import markers,lines,colors
import os,six,socket
mk = ['s','o','*','d','v','>','x']

matplotlib.rc('font',family='serif',size=15)   
from scipy.signal import find_peaks
import sys
sys.path.insert(0, '../input/circuito-respirador')
from flowmeter_class import ensayo_class
ck = [p['color'] for p in plt.rcParams['axes.prop_cycle']] 
dirw = '../input/circuito-respirador/'
def fun_fit1(x,a,b):
    return a*(1-np.exp(-b*(x-5)))
#from statsmodels.nonparametric.smoothers_lowess import lowess
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filtfilt(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

dir_data = dirw+"flowmeter/flowmeter/New design/manguera 22/"
lista_files = np.sort(glob.glob(dir_data+'*Signals.txt'))
rho = 1.2; Area = ((22.2-6.25)*1e-3)**2/4*np.pi
Area22 = np.copy(Area)
delta_x = []
salida = []
pressure_i = []
for i,filei in enumerate(lista_files):
    ensayo = ensayo_class(filei,10)
    ensayo.pres_caudal()
    ensayo.P_media
    desplazamiento = np.float(ensayo.name.split('Position ')[1].split(' ')[0].replace('_','.'))
    delta_x.append(desplazamiento)
    pressure_i.append(ensayo.P_media)
    deltas = np.ones_like(ensayo.P_media[:,0])*desplazamiento
    salida.append((deltas,ensayo.Flow_media[:,0],ensayo.P_media[:,0]))
salida = np.asarray(salida)    
index_delta = np.argsort(np.array(delta_x))
#index_pressure = np.argsort(np.array(delta_x))

Deltax_0 = 5.536
Deltax_err = 7.4
D22= 22.2e-3
nu = 1.5e-5
kD = 0.01
fig0,ax0 = plt.subplots(3,1,sharex=True,figsize=(6,8))
ax0 = ax0.ravel()
for i,filei in enumerate(lista_files[:1]):
    ensayo = ensayo_class(filei,10)
    ensayo.pres_caudal()
    ensayo.Kv_calc()
    t,P1,Q1,iKv = (ensayo.time,ensayo.Paw,ensayo.Flow,ensayo.iKv)
    ax0[0].plot(t,P1,linewidth=1,color=ck[0])
    ax0[1].plot(t,Q1,linewidth=1,color=ck[1])    
    ax0[2].plot(t,iKv**.5,linewidth=1,color=ck[2])
for ax0i in ax0:
    ax0i.grid()
ax0[0].set_xlim([ensayo.tiempos_peaks[0],ensayo.tiempos_peaks[5]]) 
ax0[0].set_ylabel('$P$ [cmH$_2$O]',fontsize=11)
ax0[1].set_ylabel('$Q$ [lt/min]',fontsize=11)
ax0[2].set_ylabel('$(Q^2/P)^{1/2}$ \n[lt/min/cmH$_2$O$^{1/2}$]',fontsize=11)
ax0[2].set_xlabel('time [s]',fontsize=11)
fig0.tight_layout()
