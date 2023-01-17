from IPython.display import display, Math,Latex

import numpy as np
pe = np.geomspace(0.01,1000,5) # List of peclet number from 0.1 to 1000 

aperture = 0.8 # Mean size of aperture in mm

D = 0.0016 

nu = 1 # Kinematic Viscocity 

dt = 0.01  # <- based on tau value 
u = (pe * D)/aperture

dx = 0.25 # <- This comes from boundary.getMesh() in palabos 

nuLB = (nu*dt)/(dx*dx) # kinematic viscocity in lattice botlzman units

omega = 1/(3*nuLB+0.5)

tau = 1/omega 



#printing

print("velocity = {}".format(u))

print("dx = {} [mm]".format(dx))



print("nuLB = {} mm^2/s ".format(nuLB))

print("tau = {}".format(tau))
DLB = (D*dt)/(dx*dx) # kinematic viscocity in lattice botlzman units

Domega = 1/(3*DLB+0.5)

Dtau = 1/Domega 

#Prints

print("Diffusion coefficient in LB units  = {}".format(DLB))

print("Diffusion coefficient tau = {}".format(Dtau))
re = (u*aperture)/nu

print("reynolds number in physical units = {}".format(re))

ulb = u *(dt/dx)

alb = aperture/dx



print("Velocity in LB units = ",ulb)

print("Aperture in LB units = ",alb)