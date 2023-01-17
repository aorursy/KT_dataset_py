# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import scipy.integrate as spi



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Beta: infection rate

#Gamma: recovery rate

#Ommega: incubation rate

    

## Define diferential equations for the Model

def odeSIR(y, t, N, beta, gamma):

    S, I, R = y

    dSdt = -beta * S * I / N

    dIdt = beta * S * I / N - gamma * I

    dRdt = gamma * I

    return dSdt, dIdt, dRdt



def odeSEIR(y, t, N, beta, gamma, omega):

    S, E, I, R = y

    dSdt = -beta * S * I / N

    dEdt = beta * S * I / N - omega*E

    dIdt = omega*E - gamma * I

    dRdt = gamma * I

    return dSdt, dEdt, dIdt, dRdt



#Model and simulation for SIR Model

def modelSIR(N, beta, gamma, days):

    I0, R0 = 1, 0

    ## S susceptible

    S0 = N - I0 - R0

    ## beta = ratio of contact between people

    ## gamma = inverse of recovering time 

    # time vector

    t = range(0,days)

    # initial conditions vector

    y0 = S0, I0, R0

    # Eval of the system of differential equations

    ret = spi.odeint(odeSIR, y0, t, args=(N, beta, gamma))

    S, I, R = ret.T

    return S, I, R



#Model and simulation for SEIR Model

def modelSEIR(N, beta, gamma, omega, days):

    """N = total population """

    I0, R0, E0 = 1, 0, 0

    ## S susceptible

    S0 = N - I0 - R0- E0

    ## beta = ratio of contact between people

    ## gamma = inverse of recovering time

    ## omega = incubation rate 

    # time vector

    t = range(0,days)

    # initial conditions vector

    y0 = S0, E0,I0, R0

    # Eval of the system of differential equations

    ret = spi.odeint(odeSEIR, y0, t, args=(N, beta, gamma, omega))

    S, E, I, R = ret.T



    return S, E, I, R



#Fuction for review data

def review(d):

    mean = int(np.mean(d))

    maxx = int(np.max(d))

    max_day = d.argmax()

    print('Daily average = ' + str(mean))

    print('Max rate  = ' + str(maxx))

    print('Day of peek = ' + str(max_day))
S_1, I_1, R_1 = modelSIR(1e5,2.5,1/5,100)  # No containment measures

S_2, I_2, R_2 = modelSIR(1e5,1.5,1/5,100) # with mild containment measures 

S_3, I_3, R_3 = modelSIR(1e5,0.5,1/5,100) # with strong containment measures
t = range(0,len(S_1))

plt.subplots(figsize=(10, 10))

plt.subplot(311)

plt.plot(t, S_1, 'g', label = 'Susceptibles')

plt.plot(t, I_1, 'r', label = 'Infected')

plt.plot(t, R_1, 'b', label = 'Recovered')

plt.ylabel('Count')

plt.title('SIR Model with beta = 2.5')

plt.legend()

plt.subplot(312)

plt.plot(t, S_2, 'g', label = 'Susceptibles')

plt.plot(t, I_2, 'r', label = 'Infected')

plt.plot(t, R_2, 'b', label = 'Recovered')

plt.ylabel('Count')

plt.title('SIR Model with beta = 1.5')

plt.legend()

plt.subplot(313)

plt.plot(t, S_3, 'g', label = 'Susceptibles')

plt.plot(t, I_3, 'r', label = 'Infected')

plt.plot(t, R_3, 'b', label = 'Recovered')

plt.xlabel('Days')

plt.ylabel('Count')

plt.title('SIR Model with beta = 0.5')

plt.legend()
t = range(0,len(I_1))

plt.plot(t, I_1, 'g', label = 'Infected without containment measures')

plt.plot(t, I_2, 'r', label = 'Infected with mild containment measures')

plt.plot(t, I_3, 'b', label = 'Infected with strong containment measures')

plt.xlabel('Days')

plt.ylabel('Count')

plt.legend()

plt.show()
print("---------Infected without containment measures-----------")

review(I_1)

print("---------Infected with mild containment measures-----------")

review(I_2)

print("---------Infected with strong containment measures-----------")

review(I_3)
t = range(0,len(I_1))

plt.plot(t, R_1, 'g', label = 'Recovered without containment measures')

plt.plot(t, R_2, 'r', label = 'Recovered with mild containment measures')

plt.plot(t, R_3, 'b', label = 'Recovered with strong containment measures')

plt.xlabel('Days')

plt.ylabel('Count')

plt.legend()

plt.show()
t = range(0,len(S_1))

plt.plot(t, S_1, 'g', label = 'Susceptibles without containment measures')

plt.plot(t, S_2, 'r', label = 'Susceptibles with mild containment measures')

plt.plot(t, S_3, 'b', label = 'Susceptibles with strong containment measures')

plt.xlabel('Days')

plt.ylabel('Count')

plt.legend()

plt.show()
S1, E1, I1, R1 = modelSEIR(1e5,2.5,1/5,1/7,200)  # without any containment 

S2, E2, I2, R2 = modelSEIR(1e5,1.5,1/5,1/7,200) # with mild containment measures

S3, E3, I3, R3 = modelSEIR(1e5,0.5,1/5,1/7,200) # with strong containment measures
t = range(0,len(S1))

plt.subplots(figsize=(10, 10))

plt.subplot(311)

plt.plot(t, S1, 'g', label = 'Susceptibles')

plt.plot(t, E1, 'c', label = 'Exposed')

plt.plot(t, I1, 'r', label = 'Infected')

plt.plot(t, R1, 'b', label = 'Recovered')

plt.ylabel('Count')

plt.title('SEIR Model with beta = 2.5')

plt.legend()

plt.subplot(312)

plt.plot(t, S2, 'g', label = 'Susceptibles')

plt.plot(t, E2, 'c', label = 'Exposed')

plt.plot(t, I2, 'r', label = 'Infected')

plt.plot(t, R2, 'b', label = 'Recovered')

plt.ylabel('Count')

plt.title('SEIR Model with beta = 1.5')

plt.legend()

plt.subplot(313)

plt.plot(t, S3, 'g', label = 'Susceptibles')

plt.plot(t, E3, 'c', label = 'Exposed')

plt.plot(t, I3, 'r', label = 'Infected')

plt.plot(t, R3, 'b', label = 'Recovered')

plt.xlabel('Days')

plt.ylabel('Count')

plt.title('SEIR Model with beta = 0.5')

plt.legend()


print("---------Infected without any containment -----------")

review(I1)

print("---------Infected with mild containment measures -----------")

review(I2)

print("---------Infected with strong containment measures -----------")

review(I3)
print("---------Exposed without any containment -----------")

review(E1)

print("---------Exposed with mild containment measures -----------")

review(E2)

print("---------Exposed with strong containment measures -----------")

review(E3)
t = range(0,len(I1))

plt.plot(t, I1, 'g', label = 'Infected without any containment')

plt.plot(t, I2, 'r', label = 'Infected with mild containment measures')

plt.plot(t, I3, 'b', label = 'Infected with strong containment measures')

plt.xlabel('Days')

plt.ylabel('Count')

plt.legend()

plt.show()
t = range(0,len(S1))

plt.plot(t, S1, 'g', label = 'Susceptibles without any containment')

plt.plot(t, S2, 'r', label = 'Susceptibles with mild containment measures')

plt.plot(t, S3, 'b', label = 'Susceptibles with strong containment measures')

plt.xlabel('Days')

plt.ylabel('Count')

plt.legend()

plt.show()