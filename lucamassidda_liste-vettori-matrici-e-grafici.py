# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
primes = [2, 3, 5, 7]
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
hands = [
    ['J', 'Q', 'K'],
    ['2', '2', '2'],
    ['6', 'A', 'K'], # (Questa virbola è opzionale)
]
# (Possiamo anche scrivere la lista su un unica riga, se preferiamo)
hands = [['J', 'Q', 'K'], ['2', '2', '2'], ['6', 'A', 'K']]
my_favourite_things = [32, 'raindrops on roses', help]
planets[0]
# Quale pianeta dopo Mercurio?


print(len(planets))
# l'ultimo pianeta è... (ricordate che gli indici partono da 0!)

planets[7]
# possiamo trovare l'ultimo elemento usando l'indice -1

print(planets[-1])
# e il penultimo?


planets[0:3]
planets[:3]
planets[3:]
# e se omettessi sia il primo indice che l'ultimo?


planets[:-1]
planets[2]
planets[2] = 'Terra'
planets
planets[2] = 'Earth'
import numpy as np
a = np.array([10, 20, 30, 40])
print(a)
type(a)
a
a.ndim # numero di dimensioni dell'array
a.shape  # shape of the array
len(a) # questo funziona sempre
a.dtype # come detto il tipo degli elementi è unico per tutto l'array, a differenza delle liste
help(np.arange)
print(np.arange(1,10,1))
print(np.arange(1,10))
print(np.arange(10))
np.arange(-1,1,0.1)
np.arange(-1,1,0.1).dtype
(1 - 0) / (10 -1)
np.linspace(0,1,10)
np.ones(213456)
a = np.linspace(0,1,11)
print(a)
print(a[2])
print(a[-1])
print(a[0:3])
a = np.arange(10)
b = np.ones(10)
print(a,b)
2*b
a+b
2*a**2
x = np.arange(0,10,0.5)
print(x)

y1 = x**2
print(y1)

y2 = 1/x
print(y2)

y3 = np.log10(x)
print(y3)

import matplotlib.pyplot as plt
plt.plot(x, y1);
plt.figure(figsize=(16,8)) # per cambiare le dimensioni della figura

plt.plot(x, x)
plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)

plt.grid()
plt.figure(figsize=(15,8)) # per cambiare le dimensioni della figura

plt.plot(x,x, label='x')
plt.plot(x, y2, label='1/x')
plt.legend()
plt.grid()
T = 273.16 + 15 # Temperatura in Kelvin

Ras = 287.05 # per l'aria secca

p = 101325 * np.arange(1, 10, 0.1) # un array di pressioni in Pascal

# Legge dei gas: p = rho*R*T

rho = p / Ras / T
print(rho)
print(rho[0], p[0])
plt.plot(1./rho, p, marker='.');
import numpy as np
So = 1367
Save = So/4
sigma = 5.67e-8
alfa = 0.3
emiss = 1
Tpr = (So/4/sigma)**(1/4)
print('T pianeta roccioso: ' + str(round(Tpr, 3)) + 'K')
Ta = (So/4/sigma*(1-alfa))**(1/4)
print('T con albedo: ' + str(round(Ta, 3)) + 'K')
T1 = (So/4/sigma*(1-alfa) / (1 - 0.5*emiss))**(1/4)
print('T 1 strato assorbente: ' + str(round(T1, 2)) + 'K')
emiss = np.linspace(0,1,101)
print(emiss)
print(alfa, Save, sigma)
Tp = ( (1 - alfa) * Save / sigma / (1 - 0.5*emiss) )**(1./4.)
print(Tp)
Tp.shape
Tp_C = Tp - 273.15
print(Tp_C)
import matplotlib.pyplot as plt
plt.figure(figsize=(16,8));

plt.plot(100*emiss, Tp_C);

plt.xlabel('emissivita [%]');
plt.ylabel('Tp [°C]');

plt.xticks(np.arange(0, 101, 5));
plt.yticks(np.arange(-20, 35, 5));

plt.grid();
emiss = 0.78
alfa = np.linspace(0, 0.9, 91)
print(alfa)
print(alfa)
print(Save)
print(sigma)
print(emiss)
Tp = ( (1 - alfa) * Save/sigma/ (1 -0.5 * emiss))**(1/4)
print(Tp)
plt.figure(figsize=(16,8))

plt.plot(100*alfa, Tp - 273.15)

plt.xticks(np.arange(0,95,5))
plt.yticks(np.arange(-100,45,5))
plt.xlabel('Albedo [%]')
plt.ylabel('Tp [C]')
plt.title("Temperatura pianeta in funzione dell'albedo")
plt.grid()
emiss = np.zeros([91, 101])
alfa = np.ones([91, 101])
print(emiss.shape, alfa.shape)
print(alfa)
for i in range(12):
    print(i)
alfa[0,:2]
for i in range(91):
    for j in range(101):
        emiss[i,j] = 0.01*j
        alfa[i,j] = 0.01*i
print(emiss)
print(alfa)
plt.imshow(emiss, origin='lower');
plt.colorbar();
plt.imshow(alfa, origin='lower');
plt.colorbar();
Tp = ( (1 - alfa) * Save/sigma/ (1 -0.5 * emiss))**(1/4)
print(Tp)
print(Tp.shape)
plt.figure(figsize=(10,8))
plt.contourf(Tp - 273.15, origin='lower',extent=[0,100,0,90], cmap='rainbow', levels=25)
plt.colorbar()
plt.title('Tp [C] funzione di albedo ed emissivita');
plt.xlabel('Emissivita [%]');
plt.ylabel('Albedo [%]');
plt.grid();
day = np.linspace(0,365,365)
delta = 23.45 * np.sin(2*np.pi*(day+284)/365)
plt.figure(figsize=(15,8));
plt.plot(day, delta);
plt.grid();
S_0 = 1367
np.sin(np.pi)
S_40 = S_0*(np.sin(np.radians(40))*np.sin(np.radians(delta))+np.cos(np.radians(40))*np.cos(np.radians(delta)))
print(S_40)
S_50 = S_0*(np.sin(np.radians(50))*np.sin(np.radians(delta))+np.cos(np.radians(50))*np.cos(np.radians(delta)))
print(S_50)
plt.figure(figsize=(16,8));
plt.plot(day,S_40,label="lat=40");
plt.plot(day,S_50,label="lat=50");
plt.xlim(0,365);
plt.ylim(0,1400);
plt.title('Radiazione solare giornaliera TOA');
plt.xlabel('Giorno anno');
plt.ylabel('Irradianza [W/m^2]');
plt.grid();
plt.legend();
phi = np.radians(np.linspace(-89, 89, 180))
print(phi)
len(delta)
plt.plot(delta);
cos_H_0 = - np.outer(np.tan(np.radians(delta)), np.tan(phi))
cos_H_0.shape
plt.figure(figsize=(16,8))
plt.imshow(cos_H_0.T);
plt.colorbar();
cos_H_0[cos_H_0 < -1] = -1
cos_H_0[cos_H_0 > 1] = np.nan
plt.figure(figsize=(16,8))
plt.imshow(cos_H_0.T);
plt.colorbar();
H_0 = np.arccos(cos_H_0)
H_ins = H_0 / np.pi * 24
print(H_ins)
plt.figure(figsize=(16,8))
plt.imshow(H_ins.T, origin='lower', aspect='auto', extent=[0,360,0,180], cmap='rainbow');
#plt.yticks(np.arange(0,180,15), np.arange(-90,89,15))
plt.grid();
plt.colorbar();
H_0.shape
S_d = S_0 / np.pi * (H_0*np.outer(np.sin(np.radians(delta)), np.sin(phi)) + np.outer(np.cos(np.radians(delta)), np.cos(phi))*np.sin(H_0))
print(S_d.shape)
plt.figure(figsize=(16,8));
plt.imshow(S_d.T, origin='lower', aspect='auto', extent=[0,12,-90,90], cmap='rainbow')
plt.grid();
plt.colorbar();
plt.title('Radiazione solare giornaliera TOA [W/m^2]');
plt.xlabel('Mese anno');
plt.ylabel('Latitudine');


