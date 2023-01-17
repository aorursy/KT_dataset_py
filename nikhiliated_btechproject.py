import numpy as np
from numpy import diff
import pandas as pd
import os
import matplotlib.pyplot as plt
#Mole Fractions Data
m15=pd.read_csv('../input/molefraction/mole15.csv')
m16=pd.read_csv('../input/molefraction/mole16.csv')
m17=pd.read_csv('../input/molefraction/mole17.csv')
m18=pd.read_csv('../input/molefraction/mole18.csv')
m19=pd.read_csv('../input/molefraction/mole19.csv')
m20=pd.read_csv('../input/molefraction/mole20.csv')
m21=pd.read_csv('../input/molefraction/mole21.csv')
m22=pd.read_csv('../input/molefraction/mole22.csv')
m23=pd.read_csv('../input/molefraction/mole23.csv')
m24=pd.read_csv('../input/molefraction/mole24.csv')
m25=pd.read_csv('../input/molefraction/mole25.csv')
m26=pd.read_csv('../input/molefraction/mole26.csv')
m27=pd.read_csv('../input/molefraction/mole27.csv')
m28=pd.read_csv('../input/molefraction/mole28.csv')
m29=pd.read_csv('../input/molefraction/mole29.csv')
m30=pd.read_csv('../input/molefraction/mole30.csv')
trpath='../input/trdata/trdata/trData/'
#Transconductance for Mole Fractions Data
trm15=pd.read_csv(trpath+'molefraction/csv/m15.csv')
trm16=pd.read_csv(trpath+'molefraction/csv/m16.csv')
trm17=pd.read_csv(trpath+'molefraction/csv/m17.csv')
trm18=pd.read_csv(trpath+'/molefraction/csv/m18.csv')
trm19=pd.read_csv(trpath+'molefraction/csv/m19.csv')
trm20=pd.read_csv(trpath+'molefraction/csv/m20.csv')
trm21=pd.read_csv(trpath+'molefraction/csv/m21.csv')
trm22=pd.read_csv(trpath+'molefraction/csv/m22.csv')
trm23=pd.read_csv(trpath+'molefraction/csv/m23.csv')
trm24=pd.read_csv(trpath+'molefraction/csv/m24.csv')
trm25=pd.read_csv(trpath+'molefraction/csv/m25.csv')
trm26=pd.read_csv(trpath+'molefraction/csv/m26.csv')
trm27=pd.read_csv(trpath+'molefraction/csv/m27.csv')
trm28=pd.read_csv(trpath+'molefraction/csv/m28.csv')
trm29=pd.read_csv(trpath+'molefraction/csv/m29.csv')
trm30=pd.read_csv(trpath+'molefraction/csv/m30.csv')
#Thickness Data
t15=pd.read_csv('../input/thickness/t0.015.csv')
t16=pd.read_csv('../input/thickness/t0.016.csv')
t17=pd.read_csv('../input/thickness/t0.017.csv')
t18=pd.read_csv('../input/thickness/t0.018.csv')
t19=pd.read_csv('../input/thickness/t0.019.csv')
t20=pd.read_csv('../input/thickness/t0.020.csv')
t21=pd.read_csv('../input/thickness/t0.021.csv')
t22=pd.read_csv('../input/thickness/t0.022.csv')
t23=pd.read_csv('../input/thickness/t0.023.csv')
t24=pd.read_csv('../input/thickness/t0.024.csv')
t25=pd.read_csv('../input/thickness/t0.025.csv')
#Transconductance for Thickness Data
trt15=pd.read_csv(trpath+'thickness/csv/t0.015.csv')
trt16=pd.read_csv(trpath+'thickness/csv/t0.016.csv')
trt17=pd.read_csv(trpath+'thickness/csv/t0.017.csv')
trt18=pd.read_csv(trpath+'thickness/csv/t0.018.csv')
trt19=pd.read_csv(trpath+'thickness/csv/t0.019.csv')
trt20=pd.read_csv(trpath+'thickness/csv/t0.020.csv')
trt21=pd.read_csv(trpath+'thickness/csv/t0.021.csv')
trt22=pd.read_csv(trpath+'thickness/csv/t0.022.csv')
trt23=pd.read_csv(trpath+'thickness/csv/t0.023.csv')
trt24=pd.read_csv(trpath+'thickness/csv/t0.024.csv')
trt25=pd.read_csv(trpath+'thickness/csv/t0.025.csv')
#Data corresponding to dependence on mole, thickness
thm=pd.read_csv('../input/threshold/thresholdvsmole.csv')
tht=pd.read_csv('../input/threshold/thresholdvsthickness.csv')
plt.plot(thm['Mole Fraction'], thm['Threshold Voltage'], marker='', linewidth=1, alpha=0.9, label='Threshold Voltage')
plt.legend(loc=1, ncol=2)
plt.title("Threshold Voltage For Different Mole Fractions", loc='left', fontsize=12, fontweight=0, color='red')
plt.xlabel("Mole Fraction")
plt.ylabel("Threshold Voltage")
mole=pd.DataFrame()
molegm=pd.DataFrame()

mole['Gate Voltage']=m15['Gate Voltage']
mole['m15']=m15['Drain Current']
mole['m16']=m16['Drain Current']
mole['m17']=m17['Drain Current']
mole['m18']=m18['Drain Current']
mole['m19']=m19['Drain Current']
mole['m20']=m20['Drain Current']
mole['m21']=m21['Drain Current']
mole['m22']=m22['Drain Current']
mole['m23']=m23['Drain Current']
mole['m24']=m24['Drain Current']
mole['m25']=m25['Drain Current']
mole['m26']=m26['Drain Current']
mole['m27']=m27['Drain Current']
mole['m28']=m28['Drain Current']
mole['m29']=m29['Drain Current']
mole['m30']=m30['Drain Current']

x=molegm['Gate Voltage']=trm30['gate bias']
molegm['m15']=trm15['dydx 1']
molegm['m16']=trm16['dydx 1']
molegm['m17']=trm17['dydx 1']
molegm['m18']=trm18['dydx 1']
molegm['m19']=trm19['dydx 1']
molegm['m20']=trm20['dydx 1']
molegm['m21']=trm21['dydx 1']
molegm['m22']=trm22['dydx 1']
molegm['m23']=trm23['dydx 1']
molegm['m24']=trm24['dydx 1']
molegm['m25']=trm25['dydx 1']
molegm['m26']=trm26['dydx 1']
molegm['m27']=trm27['dydx 1']
molegm['m28']=trm28['dydx 1']
molegm['m29']=trm29['dydx 1']
molegm['m30']=trm30['dydx 1']
#for c in mole.drop('Gate Voltage', axis=1):
#    dy = np.zeros(mole['Gate Voltage'].shape,np.float)
#    dy[0:-1]=diff(mole[c])/diff(x)
#    molegm[c]=dy
num_plots = 16
plt.figure(figsize=(12,8))
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
for column in mole.drop('Gate Voltage', axis=1):
    plt.plot(mole['Gate Voltage'], mole[column], marker='', linewidth=1, alpha=0.9, label=column)
plt.legend(loc=2, ncol=2)
plt.title("Effect Of Mole Fraction On Threshold Voltage", loc='left', fontsize=12, fontweight=0, color='red')
plt.xlabel("Gate Voltage")
plt.ylabel("Drain Current")
num_plots = 16
plt.figure(figsize=(12,8))
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
for column in molegm.drop('Gate Voltage', axis=1):
    plt.plot(molegm['Gate Voltage'], molegm[column], marker='', linewidth=1, alpha=0.9, label=column)
plt.legend(loc=2, ncol=2)
plt.title("Transconductance For Different Mole Fractions", loc='left', fontsize=12, fontweight=0, color='red')
plt.xlabel("Gate Voltage")
plt.ylabel("Transconductance")
plt.plot(tht['Thickness'], tht['Threshold Voltage'], marker='', linewidth=1, alpha=0.9, label='Threshold Voltage')
plt.legend(loc=1, ncol=2)
plt.title("Threshold Voltage For Different Thicknesses", loc='left', fontsize=12, fontweight=0, color='red')
plt.xlabel("Thickness")
plt.ylabel("Threshold Voltage")
thickness=pd.DataFrame()
thicknessgm=pd.DataFrame()

thickness['Gate Voltage']=t25['Gate Voltage']
thickness['t15']=t15['Drain Current']
thickness['t16']=t16['Drain Current']
thickness['t17']=t17['Drain Current']
thickness['t18']=t18['Drain Current']
thickness['t19']=t19['Drain Current']
thickness['t20']=t20['Drain Current']
thickness['t21']=t21['Drain Current']
thickness['t22']=t22['Drain Current']
thickness['t23']=t23['Drain Current']
thickness['t24']=t24['Drain Current']
thickness['t25']=t25['Drain Current']

x=thicknessgm['Gate Voltage']=trt25['gate bias']
thicknessgm['t15']=trt15['dydx 1']
thicknessgm['t16']=trt16['dydx 1']
thicknessgm['t17']=trt17['dydx 1']
thicknessgm['t18']=trt18['dydx 1']
thicknessgm['t19']=trt19['dydx 1']
thicknessgm['t20']=trt20['dydx 1']
thicknessgm['t21']=trt21['dydx 1']
thicknessgm['t22']=trt22['dydx 1']
thicknessgm['t23']=trt23['dydx 1']
thicknessgm['t24']=trt24['dydx 1']
thicknessgm['t25']=trt25['dydx 1']
#for c in thickness.drop('Gate Voltage', axis=1):
#    dy = np.zeros(thickness['Gate Voltage'].shape,np.float)
#    dy[0:-1]=diff(thickness[c])/diff(x)
#    thicknessgm[c]=dy
num_plots = 16
plt.figure(figsize=(12,8))
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
for column in thickness.drop('Gate Voltage', axis=1):
    plt.plot(thickness['Gate Voltage'], thickness[column], marker='', linewidth=1, alpha=0.9, label=column)
plt.legend(loc=2, ncol=2)
plt.title("Effect Of thickness On Threshold Voltage", loc='left', fontsize=12, fontweight=0, color='red')
plt.xlabel("Gate Voltage")
plt.ylabel("Drain Current")
num_plots = 16
plt.figure(figsize=(12,8))
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
for column in thicknessgm.drop('Gate Voltage', axis=1):
    plt.plot(thicknessgm['Gate Voltage'], thicknessgm[column], marker='', linewidth=1, alpha=0.9, label=column)
plt.legend(loc=2, ncol=2)
plt.title("Transconductance For Different Thicknesses", loc='left', fontsize=12, fontweight=0, color='red')
plt.xlabel("Gate Voltage")
plt.ylabel("Transconductance")
plt.plot(thickness['Gate Voltage'], thicknessgm['t20'], marker='', linewidth=1, alpha=0.9, label='m27')
plt.legend(loc=2, ncol=2)
# Add titles
plt.title("Transconductance For Different Mole Fractions", loc='left', fontsize=12, fontweight=0, color='red')
plt.xlabel("Gate Voltage")
plt.ylabel("Transconductance")
molefractions=[i for i in range(15,31)]
maxmolegm=[]
for column in molegm.drop('Gate Voltage', axis=1):
    maxmolegm.append(max(molegm[column]))
plt.figure(figsize=(12,8))
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
plt.plot(molefractions, maxmolegm, marker='', linewidth=1)
plt.title("Maximum Transconductance VS Mole Fractions", loc='left', fontsize=12, fontweight=0, color='red')
plt.xlabel("Mole Fractions")
plt.ylabel("Maximum Transconductance")
thicknesses=[i for i in range(15,26)]
maxthickgm=[]
for column in thicknessgm.drop('Gate Voltage', axis=1):
    maxthickgm.append(max(thicknessgm[column]))
plt.figure(figsize=(12,8))
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
plt.plot(thicknesses, maxthickgm, marker='', linewidth=1)
plt.title("Maximum Transconductance VS Thicknesses", loc='left', fontsize=12, fontweight=0, color='red')
plt.xlabel("Thicknesses")
plt.ylabel("Maximum Transconductance")
molefractions=[i for i in range(15,31)]
maxmole=[]
for column in mole.drop('Gate Voltage', axis=1):
    maxmole.append(max(mole[column]))
plt.figure(figsize=(12,8))
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
plt.plot(molefractions, maxmole, marker='', linewidth=1)
plt.title("Maximum Drain Current VS Mole Fractions", loc='left', fontsize=12, fontweight=0, color='red')
plt.xlabel("Mole Fractions")
plt.ylabel("Maximum Drain Current")
thicknesses=[i for i in range(15,26)]
maxthick=[]
for column in thickness.drop('Gate Voltage', axis=1):
    maxthick.append(max(thickness[column]))
plt.figure(figsize=(12,8))
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
plt.plot(thicknesses, maxthick, marker='', linewidth=1)
plt.title("Maximum Drain Current VS Thicknesses", loc='left', fontsize=12, fontweight=0, color='red')
plt.xlabel("Thicknesses")
plt.ylabel("Maximum Drain Current")
