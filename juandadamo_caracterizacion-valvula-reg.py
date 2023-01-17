import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
from scipy import interpolate,optimize,integrate,signal
import matplotlib.pyplot as plt
import pylab,six,matplotlib 
from matplotlib import markers,lines,colors
import os,six,socket
mk=np.tile(markers.MarkerStyle.filled_markers,2)
mk = ['s','o','*','d','v','>','x']
ck=list(six.iteritems(colors.cnames))
matplotlib.rc('font',family='serif',size=15)   
from scipy.signal import find_peaks
import sys
sys.path.insert(0, '../input/circuito-respirador')
from flowmeter_class import ensayo_class
from matplotlib.ticker import FormatStrFormatter

dirw = '../input/circuito-respirador/'
#el uso es muy simple, tan solo se crea la clase "ensayo" a partir del nombre de un archivo de señal .txt
A22 = pd.read_csv(dirw+'Carac_valv_reg_manguera22.csv', delimiter=',',decimal=",",skiprows=2)
A19 = pd.read_csv(dirw+'Carac_valv_reg_manguera19.csv', delimiter=',',decimal=",",skiprows=2)
desplazamiento22,Presion22,Volume22,Flow22,Velocity22,Kv22 = A22.to_numpy().T[1:7] 
desplazamiento19,Presion19,Volume19,Flow19,Velocity19,Kv19 = A19.to_numpy().T[1:7] 
posiciones22 = np.unique(desplazamiento22)
posiciones22 = np.sort(np.array([np.float(posi) for posi in posiciones22]))
posiciones19 = np.unique(desplazamiento19)
posiciones19 = np.sort(np.array([np.float(posi) for posi in posiciones19]))

fig0,ax0 = plt.subplots(1,2,figsize=(14,5),sharey = True)

for i,pos_i in enumerate(posiciones22):
    index = [n for n, x in enumerate(desplazamiento22) if '%.1f'%pos_i in x]
    line1, = ax0[1].plot(Flow22[index],Presion22[index],marker=mk[i],linestyle='',
                       markersize=10,markerfacecolor='none',label=r'$\Delta x$ %.1f'%pos_i)
    Qi,Pi = [np.array(Flow22[index],dtype='float'),np.array(Presion22[index],dtype='float')]
    polP = np.polyfit(np.append([0,0.0001],Qi),np.append([0,0.0001],Pi),2)
    P_pol = np.poly1d(polP)
    flow_s = np.linspace(0,Qi.max()+5)
    ax0[1].plot(flow_s,P_pol(flow_s),linestyle='--',color = line1.get_color())
for i,pos_i in enumerate(posiciones19):
    index = [n for n, x in enumerate(desplazamiento19) if '%.2f'%pos_i in x]
    line1, = ax0[0].plot(Flow19[index],Presion19[index],marker=mk[i],linestyle='',
                       markersize=10,markerfacecolor='none',label=r'$\Delta x$ %.1f'%pos_i)
    Qi,Pi = [np.array(Flow19[index],dtype='float'),np.array(Presion19[index],dtype='float')]
    polP = np.polyfit(np.append([0,0.0001],Qi),np.append([0,0.0001],Pi),2)
    P_pol = np.poly1d(polP)
    flow_s = np.linspace(0,Qi.max()+5)
    ax0[0].plot(flow_s,P_pol(flow_s),linestyle='--',color = line1.get_color())

ax0[0].legend(fontsize=10,title_fontsize=11,title='$\Delta x [mm]$')
ax0[1].legend(fontsize=10,title_fontsize=11,title='$\Delta x [mm]$',ncol=2)
ax0[0].set_ylabel('P [cmH$_2$O]')
ax0[1].set_ylabel('P [cmH$_2$O]')

for ax0i in ax0.ravel():
    ax0i.set_xlabel('Q [lt/min]')
    ax0i.set_xlim([0,90]);
    ax0i.grid()
ax0[0].set_title('D=19mm')
ax0[1].set_title('D=22mm');

#fig0.savefig('images_valve/PQ_respirador.pdf')
dir_data = dirw+"flowmeter/flowmeter/New design/manguera 22/"
lista_files = np.sort(glob.glob(dir_data+'*Signals.txt'))
rho = 1.2; Area = ((22.2-6.25)*1e-3)**2/4*np.pi
filei = lista_files[0]
ensayo = ensayo_class(filei,10)
ensayo.pres_caudal()

fig1,ax1 = plt.subplots(2,1,figsize=(14,6),sharex=True)
ax1[0].plot(ensayo.time,ensayo.Paw,label=r'señal')
ax1[1].plot(ensayo.time,ensayo.Flow)

ax1[0].plot([0,ensayo.time[-1]],[ensayo.P_max[0],ensayo.P_max[0]],'k--',linewidth=2, dashes=(5,3),label=r'valor máximo')
ax1[0].plot([0,ensayo.time[-1]],[ensayo.P_media[0],ensayo.P_media[0]],'r-.',linewidth=2,label = r'valor promedio')

ax1[1].plot([0,ensayo.time[-1]],[ensayo.Flow_max[0],ensayo.Flow_max[0]],'k--',linewidth=2, dashes=(5,3))
ax1[1].plot([0,ensayo.time[-1]],[ensayo.Flow_media[0],ensayo.Flow_media[0]],'r-.',linewidth=2)
ax1[0].set_ylabel('P[cmH$_2$O]')
ax1[1].set_ylabel('Q [lt/min]')
ax1[0].set_xlim([0,15])
ax1[0].legend(fontsize=11,ncol=3);

#fig1.savefig('images_valve/salida_tipica.pdf')

delta_x = []
salida = []
for i,filei in enumerate(lista_files):
    ensayo = ensayo_class(filei,10)
    ensayo.pres_caudal()
    ensayo.P_media
    desplazamiento = np.float(ensayo.name.split('Position ')[1].split(' ')[0].replace('_','.'))
    delta_x.append(desplazamiento)
    deltas = np.ones_like(ensayo.P_media[:,0])*desplazamiento
    salida.append((deltas,ensayo.Flow_media[:,0],ensayo.P_media[:,0]))
salida = np.asarray(salida)    
index_delta = np.argsort(np.array(delta_x))
fig1a,ax1a = plt.subplots(1,1,figsize=(7,4))
fig1b,ax1b = plt.subplots(1,1,figsize=(7,4))
for i,salida_i in enumerate(salida[index_delta]):
    deltas,flow,Pm = salida_i
    line2, = ax1b.plot(flow,Pm,marker=mk[i],linestyle='',
             markerfacecolor='none',markersize=10)
    
    Qi,Pi = [np.array(flow,dtype='float'),np.array(Pm,dtype='float')]
    Ui = Qi/Area/1000/60
    Kv = Pi/rho/Ui**2
    ax1a.plot(flow,Kv,marker=mk[i],linestyle='',label='$\Delta x= %.1f mm$'%deltas[0],
             markerfacecolor='none',markersize=10)
    
    polP = np.polyfit(np.append([0,0.0001],Qi),np.append([0,0.0001],Pi),2)
    P_pol = np.poly1d(polP)
    flow_s = np.linspace(0,Qi.max()+5)
    ax1b.plot(flow_s,P_pol(flow_s),linestyle='--',color = line2.get_color())
    
#ax3[1].set_ylim([0,12])
ax1a.legend(fontsize=12,ncol=2,loc='best')
ax1a.set_ylim([0,3]);
ax1a.set_xlim([0,110]);
ax1b.set_ylim([0,14]);

ax1a.set_title('D=22.2mm')
ax1b.set_title('D=22.2mm')
ax1a.set_ylabel('$K_v$')
ax1b.set_ylabel('P [cmH$_2$O]')
for ax1i  in [ax1a,ax1b]:
    ax1i.grid()
    ax1i.set_xlim([0,110])
    ax1i.set_xlabel('Q [lt/min]')
fig1a.tight_layout()
fig1b.tight_layout()
#fig1a.savefig('images_valve/KV_flowmeter_media.pdf')
#fig1b.savefig('images_valve/PQ_flowmeter_media.pdf')
delta_x = []
salida = []
for i,filei in enumerate(lista_files):
    ensayo = ensayo_class(filei,10)
    ensayo.pres_caudal()
    ensayo.P_media
    desplazamiento = np.float(ensayo.name.split('Position ')[1].split(' ')[0].replace('_','.'))
    delta_x.append(desplazamiento)
    deltas = np.ones_like(ensayo.P_media[:,0])*desplazamiento
    salida.append((deltas,ensayo.Flow_max[:,0],ensayo.P_max[:,0]))
salida = np.asarray(salida)    
index_delta = np.argsort(np.array(delta_x))
fig1a,ax1a = plt.subplots(1,1,figsize=(7,4))
fig1b,ax1b = plt.subplots(1,1,figsize=(7,4))
for i,salida_i in enumerate(salida[index_delta]):
    deltas,flow,Pm = salida_i
    line2, = ax1b.plot(flow,Pm,marker=mk[i],linestyle='',
             markerfacecolor='none',markersize=10)
    
    Qi,Pi = [np.array(flow,dtype='float'),np.array(Pm,dtype='float')]
    Ui = Qi/Area/1000/60
    Kv = Pi/rho/Ui**2
    ax1a.plot(flow,Kv,marker=mk[i],linestyle='',label='$\Delta x= %.1f mm$'%deltas[0],
             markerfacecolor='none',markersize=10)
    
    polP = np.polyfit(np.append([0,0.0001],Qi),np.append([0,0.0001],Pi),2)
    P_pol = np.poly1d(polP)
    flow_s = np.linspace(0,Qi.max()+5)
    ax1b.plot(flow_s,P_pol(flow_s),linestyle='--',color = line2.get_color())
    
#ax3[1].set_ylim([0,12])
ax1a.legend(fontsize=12,ncol=2,loc='best')
ax1a.set_ylim([0,3]);
ax1a.set_xlim([0,110]);
ax1b.set_ylim([0,14]);
ax1a.set_ylabel('$K_v$')
ax1b.set_ylabel('P [cmH$_2$O]')

ax1a.set_title('D=22.2mm')
ax1b.set_title('D=22.2mm')
for ax1i  in [ax1a,ax1b]:
    ax1i.grid()
    ax1i.set_xlim([0,110])
    ax1i.set_xlabel('Q [lt/min]')
fig1a.tight_layout()
fig1b.tight_layout()
#fig1a.savefig('images_valve/KV_flowmeter_max.pdf')
#fig1b.savefig('images_valve/PQ_flowmeter_max.pdf')
dir_data = dirw+"flowmeter/flowmeter/New design/manguera 19/"
lista_files = np.sort(glob.glob(dir_data+'*Signals.txt'))
rho = 1.2; Area = ((22.2-6.25)*1e-3)**2/4*np.pi
delta_x = []
salida = []
for i,filei in enumerate(lista_files):
    ensayo = ensayo_class(filei,10)
    ensayo.pres_caudal()
    ensayo.P_media
    desplazamiento = np.float(ensayo.name.split('Position ')[1].split(' ')[0].replace('_','.'))
    delta_x.append(desplazamiento)
    deltas = np.ones_like(ensayo.P_media[:,0])*desplazamiento
    salida.append((deltas,ensayo.Flow_media[:,0],ensayo.P_media[:,0]))
salida = np.asarray(salida)    
index_delta = np.argsort(np.array(delta_x))
fig1a,ax1a = plt.subplots(1,1,figsize=(7,4))
fig1b,ax1b = plt.subplots(1,1,figsize=(7,4))
for i,salida_i in enumerate(salida[index_delta]):
    deltas,flow,Pm = salida_i
    line2, = ax1b.plot(flow,Pm,marker=mk[i],linestyle='',
             markerfacecolor='none',markersize=10)
    
    Qi,Pi = [np.array(flow,dtype='float'),np.array(Pm,dtype='float')]
    Ui = Qi/Area/1000/60
    Kv = Pi/rho/Ui**2
    ax1a.plot(flow,Kv,marker=mk[i],linestyle='',label='$\Delta x= %.2f mm$'%deltas[0],
             markerfacecolor='none',markersize=10)
    
    polP = np.polyfit(np.append([0,0.0001],Qi),np.append([0,0.0001],Pi),2)
    P_pol = np.poly1d(polP)
    flow_s = np.linspace(0,Qi.max()+5)
    ax1b.plot(flow_s,P_pol(flow_s),linestyle='--',color = line2.get_color())
    
#ax3[1].set_ylim([0,12])
ax1a.legend(fontsize=12,ncol=2,loc='best')
ax1a.set_ylim([0,3]);
ax1a.set_xlim([0,110]);
ax1b.set_ylim([0,14]);

ax1a.set_title('D=19.mm')
ax1b.set_title('D=19.mm')
ax1a.set_ylabel('$K_v$')
ax1b.set_ylabel('P [cmH$_2$O]')
for ax1i  in [ax1a,ax1b]:
    ax1i.grid()
    ax1i.set_xlim([0,110])
    ax1i.set_xlabel('Q [lt/min]')
fig1a.tight_layout()
fig1b.tight_layout()
#fig1a.savefig('images_valve/19_KV_flowmeter_media.pdf')
#fig1b.savefig('images_valve/19_PQ_flowmeter_media.pdf')
delta_x = []
salida = []
for i,filei in enumerate(lista_files):
    ensayo = ensayo_class(filei,10)
    ensayo.pres_caudal()
    ensayo.P_media
    desplazamiento = np.float(ensayo.name.split('Position ')[1].split(' ')[0].replace('_','.'))
    delta_x.append(desplazamiento)
    deltas = np.ones_like(ensayo.P_media[:,0])*desplazamiento
    salida.append((deltas,ensayo.Flow_max[:,0],ensayo.P_max[:,0]))
salida = np.asarray(salida)    
index_delta = np.argsort(np.array(delta_x))
fig1a,ax1a = plt.subplots(1,1,figsize=(7,4))
fig1b,ax1b = plt.subplots(1,1,figsize=(7,4))
for i,salida_i in enumerate(salida[index_delta]):
    deltas,flow,Pm = salida_i
    line2, = ax1b.plot(flow,Pm,marker=mk[i],linestyle='',
             markerfacecolor='none',markersize=10)
    
    Qi,Pi = [np.array(flow,dtype='float'),np.array(Pm,dtype='float')]
    Ui = Qi/Area/1000/60
    Kv = Pi/rho/Ui**2
    ax1a.plot(flow,Kv,marker=mk[i],linestyle='',label='$\Delta x= %.2f mm$'%deltas[0],
             markerfacecolor='none',markersize=10)
    
    polP = np.polyfit(np.append([0,0.0001],Qi),np.append([0,0.0001],Pi),2)
    P_pol = np.poly1d(polP)
    flow_s = np.linspace(0,Qi.max()+5)
    ax1b.plot(flow_s,P_pol(flow_s),linestyle='--',color = line2.get_color())
    
#ax3[1].set_ylim([0,12])
ax1a.legend(fontsize=12,ncol=2,loc='best')
ax1a.set_ylim([0,3]);
ax1a.set_xlim([0,110]);
ax1b.set_ylim([0,14]);
ax1a.set_ylabel('$K_v$')
ax1b.set_ylabel('P [cmH$_2$O]')

ax1a.set_title('D=19 mm')
ax1b.set_title('D=19 mm')
for ax1i  in [ax1a,ax1b]:
    ax1i.grid()
    ax1i.set_xlim([0,110])
    ax1i.set_xlabel('Q [lt/min]')
fig1a.tight_layout()
fig1b.tight_layout()
#fig1a.savefig('images_valve/19_KV_flowmeter_max.pdf')
#fig1b.savefig('images_valve/19_PQ_flowmeter_max.pdf')