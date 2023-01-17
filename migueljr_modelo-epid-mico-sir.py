import matplotlib.pyplot as plt

import scipy.integrate as si

import numpy as np
#ODEs



def SIR_model(y,t,beta,gamma):

    S,I,R = y

    

    dS_dt = -beta*S*I

    dI_dt = beta*S*I - gamma*I

    dR_dt = gamma*I

    

    return ([dS_dt,dI_dt,dR_dt])

    
# S - susceptible

# I - for the number of infectious

# R - for the number of recovered 



So = 0.999

Io = 0.001

Ro = 0.0

beta = 0.2

gamma = 0.08



# Tempo

t = np.linspace(0,180,180)



# Solução

solution = si.odeint(

    SIR_model,

    [So,Io,Ro],

    t,

    args = (beta,gamma),

)

solution = np.array(solution)



# Plot 

plt.figure(figsize = (10,7))

plt.plot(t,solution[:,0],'-ob',label = 'S(t) susceptible')

plt.plot(t,solution[:,1],'-or',label = 'I(t) infectious')

plt.plot(t,solution[:,2],'-og',label = 'R(t) recovered')

plt.legend()

plt.grid()

plt.xlabel('Time')

plt.ylabel('Proportions')

plt.title('Figura 1 - SIR model',fontsize = 15)
#ODEs



def SIR_model(y,t,beta,gamma,N,u):

    S,I,R = y

    

    dS_dt = -beta*S*I + u*(N-S)

    dI_dt = beta*S*I - gamma*I - u*I

    dR_dt = gamma*I - u*R

    

    return ([dS_dt,dI_dt,dR_dt])

    
So = 1.0

Io = 0.001

Ro = 0.0



# Se u = N = 0... voltamos a Figura 1...

u = 0.03# Taxa de mortalidade do vírus (quanto ele mata)

N = 1.0# Taxa de nataidade





beta = 0.2

gamma = 0.08





# Tempo

t = np.linspace(0,180,180)



# Solução

solution = si.odeint(

    SIR_model,

    [So,Io,Ro],

    t,

    args = (beta,gamma,N,u),

)

solution = np.array(solution)



# Plot

plt.figure(figsize = (10,7))

plt.plot(t,solution[:,0],'-ob',label = 'S(t) susceptible')

plt.plot(t,solution[:,1],'-or',label = 'I(t) infectious')

plt.plot(t,solution[:,2],'-og',label = 'R(t) recovered')

plt.legend()

plt.grid()

plt.xlabel('Time')

plt.ylabel('Proportions')

plt.title('Figura 2 - SIR model com taxa de mortalidade e natalidade',fontsize = 15)