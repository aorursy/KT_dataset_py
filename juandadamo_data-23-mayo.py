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
def plot_flow(ensayoi,name='Patient 1'):
    fig0,ax0 = plt.subplots(2,1,sharex = True,figsize=(6.5,6))
    ax0[0].plot(ensayoi.time,ensayoi.Paw,color=ck[0])
    ax0[0].set_ylabel('p [cmH$_2$O]')
    ax0[1].plot(ensayoi.time,ensayoi.Flow,color=ck[1])
    ax0[1].set_ylabel('Q [lt/min]')
    ax0[1].set_xlabel('$t$ [s]')
    ax0[0].grid()
    ax0[1].grid()
    ax0[1].set_xlim([0,15])
    ax0[0].set_ylim([0,30])
    ax0[1].set_ylim([-35,82])
    fig0.suptitle(name,fontsize=10)
    fig0.tight_layout()

    fig0.savefig('PQ_'+name+'.png')
    return
    
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

def fun_fit1(x,a,b):
    return a*(1-np.exp(-b*(x-5)))
rho = 1.2;
Diam = 22.2
Area = ((Diam-2*6.25)*1e-3)**2/4*np.pi
dir_data = dirw+"flowmeter/flowmeter/2020_05_23/"
lista_files = np.sort(glob.glob(dir_data+'*Signals.txt'))


Area22 = np.copy(Area)
delta_x = []
salida = []
pressure_i = []

time_stamp = np.array([nombrei.split('20_05_23-')[1].split(' ')[0] for  nombrei in lista_files])
ind_ord = time_stamp.argsort()
lista_files = lista_files[ind_ord]
fig0,ax0 = plt.subplots(1,2,figsize=(8,3),sharex = True)

for i,filei in enumerate(lista_files[:5][:]):
    ensayo_pat = ensayo_class(filei)
    ensayo_res = ensayo_class(lista_files[6+i])
    lin, = ax0[0].plot(ensayo_pat.time,ensayo_pat.Paw,label = '%.1f'%(ensayo_pat.Paw.max()))
    ax0[0].plot(ensayo_res.time,ensayo_res.Paw,color=lin.get_color(),linestyle='--')
    
    lin, = ax0[1].plot(ensayo_pat.time,ensayo_pat.Flow,label = '%.1f'%(ensayo_pat.Paw.max()))
    ax0[1].plot(ensayo_res.time,ensayo_res.Flow,color=lin.get_color(),linestyle='--')
    
ax0[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=9,title='P[cmH$_2$O]',title_fontsize=10)

ax0[0].set_xlim([0,4])
ax0[0].set_title('Pressure',fontsize=11)
ax0[1].set_title('Flow',fontsize=11)
fig0.tight_layout()


fig0,ax0 = plt.subplots(1,2,figsize=(8,3),sharex = True)

for i,filei in enumerate(lista_files[12:17][:]):
    ensayo_pat = ensayo_class(filei)
    ensayo_res = ensayo_class(lista_files[21-i])
    lin, = ax0[0].plot(ensayo_pat.time,ensayo_pat.Paw,label = '%.1f'%(ensayo_pat.Paw.max()))
    ax0[0].plot(ensayo_res.time,ensayo_res.Paw,color=lin.get_color(),linestyle='--')
    
    lin, = ax0[1].plot(ensayo_pat.time,ensayo_pat.Flow,label = '%.1f'%(ensayo_pat.Paw.max()))
    ax0[1].plot(ensayo_res.time,ensayo_res.Flow,color=lin.get_color(),linestyle='--')
    
ax0[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=9,title='P[cmH$_2$O]',title_fontsize=10)

ax0[0].set_xlim([0,4])
ax0[0].set_title('Pressure',fontsize=11)
ax0[1].set_title('Flow',fontsize=11)
fig0.tight_layout()
fig0,ax0 = plt.subplots(1,2,figsize=(8,3),sharex = True)

for i,filei in enumerate(lista_files[23:28][:]):
    ensayo_pat = ensayo_class(filei)
    ensayo_res = ensayo_class(lista_files[39-i])
    lin, = ax0[0].plot(ensayo_pat.time,ensayo_pat.Paw,label = '%.1f'%(ensayo_pat.Paw.max()))
    ax0[0].plot(ensayo_res.time,ensayo_res.Paw,color=lin.get_color(),linestyle='--')
    
    lin, = ax0[1].plot(ensayo_pat.time,ensayo_pat.Flow,label = '%.1f'%(ensayo_pat.Paw.max()))
    ax0[1].plot(ensayo_res.time,ensayo_res.Flow,color=lin.get_color(),linestyle='--')
    
ax0[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=9,title='P[cmH$_2$O]',title_fontsize=10)

ax0[0].set_xlim([0,4])
ax0[0].set_title('Pressure',fontsize=11)
ax0[1].set_title('Flow',fontsize=11)
fig0.tight_layout()
fig0,ax0 = plt.subplots(1,2,figsize=(8,3),sharex = True)

for i,filei in enumerate(lista_files[30:35][:]):
    ensayo_pat = ensayo_class(filei)
    ensayo_res = ensayo_class(lista_files[39-i])
    lin, = ax0[0].plot(ensayo_pat.time,ensayo_pat.Paw,label = '%.1f'%(ensayo_pat.Paw.max()))
    ax0[0].plot(ensayo_res.time,ensayo_res.Paw,color=lin.get_color(),linestyle='--')
    
    lin, = ax0[1].plot(ensayo_pat.time,ensayo_pat.Flow,label = '%.1f'%(ensayo_pat.Paw.max()))
    ax0[1].plot(ensayo_res.time,ensayo_res.Flow,color=lin.get_color(),linestyle='--')
    
ax0[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=9,title='P[cmH$_2$O]',title_fontsize=10)

ax0[0].set_xlim([0,4])
ax0[0].set_title('Pressure',fontsize=11)
ax0[1].set_title('Flow',fontsize=11)
fig0.tight_layout()
fig0,ax0 = plt.subplots(1,2,figsize=(8,3),sharex = True)

for i,filei in enumerate(lista_files[46:51][:]):
    ensayo_pat = ensayo_class(filei)
    ensayo_res = ensayo_class(lista_files[41+i])
    lin, = ax0[0].plot(ensayo_pat.time,ensayo_pat.Paw,label = '%.1f'%(ensayo_pat.Paw.max()))
    ax0[0].plot(ensayo_res.time,ensayo_res.Paw,color=lin.get_color(),linestyle='--')
    
    lin, = ax0[1].plot(ensayo_pat.time,ensayo_pat.Flow,label = '%.1f'%(ensayo_pat.Paw.max()))
    ax0[1].plot(ensayo_res.time,ensayo_res.Flow,color=lin.get_color(),linestyle='--')
    
ax0[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=9,title='P[cmH$_2$O]',title_fontsize=10)

ax0[0].set_xlim([0,4])
ax0[0].set_title('Pressure',fontsize=11)
ax0[1].set_title('Flow',fontsize=11)
fig0.tight_layout()
fig0,ax0 = plt.subplots(1,2,figsize=(8,3),sharex = True)

for i,filei in enumerate(lista_files[52:57][:]):
    ensayo_pat = ensayo_class(filei)
    ensayo_res = ensayo_class(lista_files[41+i])
    lin, = ax0[0].plot(ensayo_pat.time,ensayo_pat.Paw,label = '%.1f'%(ensayo_pat.Paw.max()))
    ax0[0].plot(ensayo_res.time,ensayo_res.Paw,color=lin.get_color(),linestyle='--')
    
    lin, = ax0[1].plot(ensayo_pat.time,ensayo_pat.Flow,label = '%.1f'%(ensayo_pat.Paw.max()))
    ax0[1].plot(ensayo_res.time,ensayo_res.Flow,color=lin.get_color(),linestyle='--')
    
ax0[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=9,title='P[cmH$_2$O]',title_fontsize=10)

ax0[0].set_xlim([0,4])
ax0[0].set_title('Pressure',fontsize=11)
ax0[1].set_title('Flow',fontsize=11)
fig0.tight_layout()
fig0,ax0 = plt.subplots(1,2,figsize=(8,3),sharex = True)

for i,filei in enumerate(lista_files[65:70][:]):
    ensayo_pat = ensayo_class(filei)
    ensayo_res = ensayo_class(lista_files[70+i])
    lin, = ax0[0].plot(ensayo_pat.time,ensayo_pat.Paw,label = '%.1f'%(ensayo_pat.Paw.max()))
    ax0[0].plot(ensayo_res.time,ensayo_res.Paw,color=lin.get_color(),linestyle='--')
    
    lin, = ax0[1].plot(ensayo_pat.time,ensayo_pat.Flow,label = '%.1f'%(ensayo_pat.Paw.max()))
    ax0[1].plot(ensayo_res.time,ensayo_res.Flow,color=lin.get_color(),linestyle='--')
    
ax0[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=9,title='P[cmH$_2$O]',title_fontsize=10)

ax0[0].set_xlim([0,4])
ax0[0].set_title('Pressure',fontsize=11)
ax0[1].set_title('Flow',fontsize=11)
fig0.tight_layout()
fig0,ax0 = plt.subplots(1,2,figsize=(8,3),sharex = True)

for i,filei in enumerate(lista_files[60:65][:]):
    ensayo_pat = ensayo_class(filei)
    ensayo_res = ensayo_class(lista_files[70+i])
    lin, = ax0[0].plot(ensayo_pat.time,ensayo_pat.Paw,label = '%.1f'%(ensayo_pat.Paw.max()))
    ax0[0].plot(ensayo_res.time,ensayo_res.Paw,color=lin.get_color(),linestyle='--')
    
    lin, = ax0[1].plot(ensayo_pat.time,ensayo_pat.Flow,label = '%.1f'%(ensayo_pat.Paw.max()))
    ax0[1].plot(ensayo_res.time,ensayo_res.Flow,color=lin.get_color(),linestyle='--')
    
ax0[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=9,title='P[cmH$_2$O]',title_fontsize=10)

ax0[0].set_xlim([0,4])
ax0[0].set_title('Pressure',fontsize=11)
ax0[1].set_title('Flow',fontsize=11)
fig0.tight_layout()
lista_resp = lista_files[[80,81,87]]
lista_rama1 = lista_files[[78,83,84]]
lista_rama2 = lista_files[[79,82,86,85]]
fig0,ax0 = plt.subplots(1,2,figsize=(8,3),sharex = True)

for i,filei in enumerate(lista_rama1[:]):
    ensayo_pat = ensayo_class(filei)
    ensayo_res = ensayo_class(lista_resp[i])
    lin, = ax0[0].plot(ensayo_pat.time,ensayo_pat.Paw,label = '%.1f'%(ensayo_pat.Paw.max()))
    ax0[0].plot(ensayo_res.time,ensayo_res.Paw,color=lin.get_color(),linestyle='--')
    
    lin, = ax0[1].plot(ensayo_pat.time,ensayo_pat.Flow,label = '%.1f'%(ensayo_pat.Paw.max()))
    ax0[1].plot(ensayo_res.time,ensayo_res.Flow,color=lin.get_color(),linestyle='--')
    
ax0[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=9,title='P[cmH$_2$O]',title_fontsize=10)

ax0[0].set_xlim([0,4])
ax0[0].set_title('Pressure',fontsize=11)
ax0[1].set_title('Flow',fontsize=11)
fig0.tight_layout()
fig0,ax0 = plt.subplots(1,2,figsize=(8,3),sharex = True)

for i,filei in enumerate(lista_rama2[:-1]):
    ensayo_pat = ensayo_class(filei)
    ensayo_res = ensayo_class(lista_resp[i])
    lin, = ax0[0].plot(ensayo_pat.time,ensayo_pat.Paw,label = '%.1f'%(ensayo_pat.Paw.max()))
    ax0[0].plot(ensayo_res.time,ensayo_res.Paw,color=lin.get_color(),linestyle='--')
    
    lin, = ax0[1].plot(ensayo_pat.time,ensayo_pat.Flow,label = '%.1f'%(ensayo_pat.Paw.max()))
    ax0[1].plot(ensayo_res.time,ensayo_res.Flow,color=lin.get_color(),linestyle='--')
    
ax0[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=9,title='P[cmH$_2$O]',title_fontsize=10)

ax0[0].set_xlim([0,4])
ax0[0].set_title('Pressure',fontsize=11)
ax0[1].set_title('Flow',fontsize=11)
fig0.tight_layout()

lista_files[60]
