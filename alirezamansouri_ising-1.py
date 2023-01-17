# *****************************************************************
# * Ising Model, Exercise 9, ver 1.5                              *
# * Date: 1399.06.06                                              *
# * Copyleft 2020 (ɔ) F.Bolhasani Far, all lefts reserved.        *
# * student ID: 981118                                            *
# * Following function written in python 3                        *
# *****************************************************************


#--------------------------------------------------------#
#                    Part B: random spin                 #                  
#--------------------------------------------------------#


##### Moduls ######

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt


#### functions ####


def initial(N):
    "considering a random spin"
    # 2d array for random spin: s = +1 or s = -1
    sit = 2*np.random.randint(2, size=(N,N))-1
    return sit


def M_C (config, beta):         # beta = 1/T
    "Monte Carlo move with using of Metropolice algoritem"

    for i in range (N):        
        for j in range (N):
            x = np.random.randint(0, N)
            y = np.random.randint(0, N)

            # spin
            s =  config[x, y]      
            nbh = config[(x+1)%N,y] + config[x,(y+1)%N] + config[(x-1)%N,y] + config[x,(y-1)%N]
            # moving toward the neighbor spin
            
            delta_E = 2*s*nbh
            if delta_E <= 0:
                s = -s
            elif rand() <= np.exp(-delta_E * beta ):
                s = -s

            config[x,y] = s

    return config


def Energy(config):
    "calculating energy of a configuration"
    energy = 0
    for i in range(len(config)):
        for j in range(len(config)):
            S = config[i,j]
            nbh = config[(i+1)%N, j] + config[i,(j+1)%N] + config[(i-1)%N, j] + config[i,(j-1)%N]
            energy += -nbh*S
    return energy/4.


def Mag (config):
    "Magnetization of a configuration"
    Mag = np.sum(config)
    return Mag


##### parameters #####

L       = 10           # size of lattice 
N       = L**2        # number of spins
N1_step = 100           # number of M_C sweeps for equilibration
N2_step = 100           # number of M_C sweeps for calculation
N_T     = 105           # number of temperature points
T0      = 10            # initial tempetarure
T_final = 0.5           # final temperature
    

# T[n+1] = (1-epsilon)*T[n],    epsilon = 0.01
T = np.linspace(T0, T_final, N_T) 
M = np.zeros(N_T)       # magnetization
X = np.zeros(N_T)       # magnetic susceptibility
E = np.zeros(N_T)       # energy of a configuration

# divide by number of samples
n1 = 1.0/(N1_step * N**2)
n2 = 1.0/(N2_step**2 * N**2)


###### main part of program ######

for i in range (N_T):
    E1 = M1 = E2 = M2 = 0
    config = initial(N)
    iT = 1.0/T[i]        # beta

    # Monte Carlo move to reach equilibriumc
    for j in range (N1_step): 
        M_C(config, iT)
    
    # Monte Carlo move, calculation
    for i in range(N2_step):
        M_C(config, iT)           
        Ene = Energy(config)     # calculate the energy
        Magn = Mag(config)       # calculate the magnetisation

        E1 = E1 + Ene
        M1 = M1 + Magn
        M2 = M2 + Magn*Magn 
        E2 = E2 + Ene*Ene


        E[i] = n1*E1
        M[i] = n1*M1
        X[i] = (n1*M2 - n2*M1*M1)*iT


######### plot ##########

pic = plt.figure(figsize=(18, 10))   

## energy:
diagram1 = pic.add_subplot(2,2,1)
plt.scatter(T, E, s=50, marker='.', color='Red')
plt.xlabel("Temperature")
plt.ylabel("Energy")
plt.axis('tight')

## magnetization:
diagram1 = pic.add_subplot(2,2,2)
plt.scatter(T, M, s=50, marker='.', color='Blue')
plt.xlabel("Temperature")
plt.ylabel("Magnetization")
plt.axis('tight')

## magnetization:
diagram1 = pic.add_subplot(2,2,3)
plt.scatter(T, X , s=50, marker='.', color='green')
plt.xlabel("Temperature")
plt.ylabel("Susceptibility")
plt.axis('tight')

plt.show()