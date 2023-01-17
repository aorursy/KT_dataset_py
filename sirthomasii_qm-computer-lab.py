import numpy as np

import matplotlib.pyplot as plt

import scipy.constants as const

import os

import math



# Parameters

ximin=-8

ximax=-ximin

Nsteps=1001

nu_0=4.0

ϵ=-4



ξ_vector=np.linspace(ximin,ximax,Nsteps)

h=ξ_vector[1]-ξ_vector[0]

nu=-nu_0*np.exp(-ξ_vector**2)



#Bisection method for positive wave function tails 

def findOddEnergy(ϵ_min,ϵ_max,noplot=None):

    

    while(abs(ϵ_min-ϵ_max)>1e-13):

        

        ϵ_avg=(ϵ_min+ϵ_max)/2        

        k=math.sqrt(abs(ϵ_avg))

        ϕ=[]

        ϕ.append(1)

        ϕ.append(np.exp(k*h))



        for i in range(2,Nsteps):

            ϕ.append(((2+h**2*(nu[i]-ϵ_avg))*ϕ[i-1])-ϕ[i-2])



        ϕ_950=ϕ[Nsteps-2]

        

        if(ϕ_950)<0:

            ϵ_max=ϵ_avg

        else:

            ϵ_min=ϵ_avg

            

    if noplot is None:

        plt.plot(ϕ, '-')

        plt.show()



    return ϵ_avg, ϕ;



#Bisection method for negative wave function tails

def findEvenEnergy(ϵ_min,ϵ_max,noplot=None):

    

    while(abs(ϵ_min-ϵ_max)>1e-13):

        

        ϵ_avg=(ϵ_min+ϵ_max)/2

        k=math.sqrt(abs(ϵ_avg))

        ϕ=[]

        ϕ.append(1)

        ϕ.append(np.exp(k*h))



        for i in range(2,Nsteps):

            ϕ.append(((2+h**2*(nu[i]-ϵ_avg))*ϕ[i-1])-ϕ[i-2])



        ϕ_950=ϕ[Nsteps-2]



        if((ϕ_950)<0):

            ϵ_min=ϵ_avg

        else:

            ϵ_max=ϵ_avg

    if noplot is None:

        plt.plot(ϕ, '-')

        plt.show()



    return ϵ_avg, ϕ;



#choosing arbritrary point for evaluating the energy value

def shootingMethod(ϵ,nu):        

    ϵ_avg=ϵ

    k=math.sqrt(abs(ϵ_avg))

    ϕ=[]

    ϕ.append(1)

    ϕ.append(np.exp(k*h))



    for i in range(2,Nsteps):

        ϕ.append(((2+h**2*(nu[i]-ϵ_avg))*ϕ[i-1])-ϕ[i-2])

    

    return ϕ;



plt.plot(nu, '-')

plt.show()

print("Potential Well")



E_0,φ_0  = findOddEnergy(-2.5,-2)

print("Energy: "+str(E_0))



E_1,φ_1 = findEvenEnergy(E_0,0)

print("Energy: "+str(E_1))



E_2,φ_2 = findOddEnergy(E_1,0)

print("Energy: "+str(E_2))

# Parameters

ximin=-16

ximax=-ximin

Nsteps=2001

nu_0=2

ϵ=-4

ξ_0=4



ξ_vector=np.linspace(ximin,ximax,Nsteps)

h=ξ_vector[1]-ξ_vector[0]



nu=-nu_0*(np.exp(-(ξ_vector-ξ_0)**2)+np.exp(-(ξ_vector+ξ_0)**2))



plt.plot(nu, '-')

plt.show()

print("Potential Wells")



nu=-nu_0*(np.exp(-(ξ_vector-ξ_0)**2))



E_0,φ_0  = findOddEnergy(-2,-.5)

print("Energy: "+str(E_0))



nu=-nu_0*(np.exp(-(ξ_vector+ξ_0)**2))



E_0,φ_0  = findOddEnergy(-2,-.5)

print("Energy: "+str(E_0))

from sympy import symbols, Eq, solve



# Parameters

ximin=-16

ximax=-ximin

Nsteps=2001

nu_0=2

ϵ=-4

ξ_0=1.5

e_0 = -0.954 #determined from previous step





ξ_vector=np.linspace(ximin,ximax,Nsteps)

h=ξ_vector[1]-ξ_vector[0]

nu=-nu_0*(np.exp(-(ξ_vector-ξ_0)**2)+np.exp(-(ξ_vector+ξ_0)**2))



#Find A and W

def findA_W(e_g, e_e):

    A, W = symbols('A W')



    eq1 = Eq(e_0 - A + W - e_g)

    eq2 = Eq(e_0 - A - W + e_e)



    sol_dict = solve((eq1,eq2), (A, W))

    A = sol_dict[A]

    W = sol_dict[W]

    

    return A,W



plt.plot(nu, '-')

plt.show()

print("Potential Wells")



E_0,φ_0  = findOddEnergy(-1.3,-.9)

print("Energy: "+str(E_0))



E_1,φ_1 = findEvenEnergy(E_0,0)

print("Energy: "+str(E_1))



E_2,φ_2 = findOddEnergy(E_1,0)

print("Energy: "+str(E_2))



A,W = findA_W(E_0,E_1)

print("A_0: "+"{:.2E}".format(A), "W_0: "+"{:.2E}".format(W))

# Parameters

ximin=-16

ximax=-ximin

Nsteps=2001

nu_0=4

ϵ=-4

ξ_0=4



ξ_vector=np.linspace(ximin,ximax,Nsteps)

h=ξ_vector[1]-ξ_vector[0]

nu=-nu_0*(np.exp(-(ξ_vector-ξ_0)**2)+np.exp(-(ξ_vector+ξ_0)**2))



plt.plot(nu, '-')

plt.show()

print("Potential Wells")



nu=-nu_0*(np.exp(-(ξ_vector-ξ_0)**2))



E_0,φ_0  = findOddEnergy(-3,-2)

print("Energy: "+str(E_0))



nu=-nu_0*(np.exp(-(ξ_vector+ξ_0)**2))



E_0,φ_0  = findOddEnergy(-3,-2.2)

print("Energy: "+str(E_0))

# Parameters

ximin=-16

ximax=-ximin

Nsteps=2001

nu_0=4

ϵ=-4

ξ_0=1.5

e_0 = -2.375 #determined from previous step



ξ_vector=np.linspace(ximin,ximax,Nsteps)

h=ξ_vector[1]-ξ_vector[0]

nu=-nu_0*(np.exp(-(ξ_vector-ξ_0)**2)+np.exp(-(ξ_vector+ξ_0)**2))



plt.plot(nu, '-')

plt.show()

print("Potential Wells")



E_0,φ_0  = findOddEnergy(-3,-2.50)

print("Energy: "+str(E_0))



E_1,φ_1 = findEvenEnergy(E_0,0)

print("Energy: "+str(E_1))



E_2,φ_2 = findOddEnergy(E_1,0)

print("Energy: "+str(E_2))



A,W = findA_W(E_0,E_1)

print("A_0: "+"{:.2E}".format(A), "W_0: "+"{:.2E}".format(W))



A,W = findA_W(E_0,E_2)

print("A_1: "+"{:.2E}".format(A), "W_1: "+"{:.2E}".format(W))

# Parameters

ximin=-16

ximax=-ximin

Nsteps=2001

nu_0=4

ϵ=-4

ξ_0=1.3



ξ_vector=np.linspace(ximin,ximax,Nsteps)

h=ξ_vector[1]-ξ_vector[0]

nu=-nu_0*(np.exp(-(ξ_vector-ξ_0)**2)+np.exp(-(ξ_vector+ξ_0)**2))



plt.plot(nu, '-')

plt.show()

print("Potential Wells")



E_0,φ_0  = findOddEnergy(-3,-2.55)

print("Energy: "+str(E_0))



E_1,φ_1 = findEvenEnergy(E_0,0)

print("Energy: "+str(E_1))



E_2,φ_2 = findOddEnergy(E_1,0)

print("Energy: "+str(E_2))



A,W = findA_W(E_0,E_1)

print("A_0: "+"{:.2E}".format(A), "W_0: "+"{:.2E}".format(W))



A,W = findA_W(E_0,E_2)

print("A_1: "+"{:.2E}".format(A), "W_1: "+"{:.2E}".format(W))

# Parameters

ximin=-16

ximax=-ximin

Nsteps=2001

nu_0=4

ϵ=-4

ξ_0=1.1



ξ_vector=np.linspace(ximin,ximax,Nsteps)

h=ξ_vector[1]-ξ_vector[0]

nu=-nu_0*(np.exp(-(ξ_vector-ξ_0)**2)+np.exp(-(ξ_vector+ξ_0)**2))



plt.plot(nu, '-')

plt.show()

print("Potential Wells")



E_0,φ_0  = findOddEnergy(-3,-2.55)

print("Energy: "+str(E_0))



E_1,φ_1 = findEvenEnergy(E_0,0)

print("Energy: "+str(E_1))



E_2,φ_2 = findOddEnergy(E_1,0)

print("Energy: "+str(E_2))



A,W = findA_W(E_0,E_1)

print("A_0: "+"{:.2E}".format(A), "W_0: "+"{:.2E}".format(W))



A,W = findA_W(E_0,E_2)

print("A_1: "+"{:.2E}".format(A), "W_1: "+"{:.2E}".format(W))
