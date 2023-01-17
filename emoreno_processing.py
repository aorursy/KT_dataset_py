# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import struct
import numpy as np
import matplotlib.pyplot as plt
from struct import unpack
import os

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/referencia/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
filere=open("/kaggle/input/datafor/scintillator10.txt","rb")
a=filere.read()
ltraza,ruido,inicio,fin=750,150,200,400 # 500,100,120,350
ndat=int((len(a)/ltraza))
filer=open("scintillator10_0.dat","a")
charge=[]
#os.remove("patron00.dat")
ii=0
for j in range(ndat):
    t10,t50,t90=0,0,0
    vwave=a[j*ltraza:j*ltraza+ltraza]
    ADC_wave = (np.array(unpack('%sB' % len(vwave),vwave)))*(.5/25.6)
    noise=sum(ADC_wave[:ruido])/ruido
    nADC_wave=(ADC_wave-noise)*-1
    nADC_wavec=nADC_wave[inicio:fin]
    vcharge=[]
    for i in range(len(nADC_wavec)):
        vcharge.append(sum(nADC_wavec[0:i]))
    qm=max(nADC_wave)
    charge.append(qm)
    for i in range(len(vcharge)): 
        if vcharge[i]>qm*.1: 
            t10=i
            break
    for i in range(len(vcharge)): 
        if vcharge[i]>qm*.5: 
            t50=i
            break
    for i in range(len(vcharge)): 
        if vcharge[i]>qm*.9: 
            t90=i
            break
    #print (qm,vcharge,t50,t90)
    filer.write(str(qm)+" "+str(max(vcharge))+" "+str(t50-t10)+" "+str(t90-t10)+"\n")
    ii=ii+1
print ("Fin",ii)
plt.plot(nADC_wave)
print (max(nADC_wave))
print (qm,ndat)
plt.title("Pulso")
plt.xlabel("t [ua]") 
plt.ylabel("Voltaje [mV]")
plt.figure(2)

plt.title("Pulso")
plt.xlabel("t [ua]") 
plt.ylabel("Voltaje [mV]")
plt.plot(nADC_wavec)
plt.figure(3)
plt.title("Carga integrada")
plt.xlabel("t [ua]") 
plt.ylabel("Carga [ua]")
plt.plot(vcharge)
plt.figure(4)
plt.title("Histograma de carga")
plt.xlabel("Carga [ua]") 
plt.ylabel("Frecuencia")
plt.hist(charge,bins=50);
