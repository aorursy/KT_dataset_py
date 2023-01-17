import matplotlib.pyplot as plt

import math

import numpy as np
n = 1000 # the number of events

t = 3600 # time

l = n/t # lambda 

k= [1, 2, 3, 4, 5]

P1,P2,P3,P4,P5 = [],[],[],[],[]

T = np.arange(0, 15, 0.1)

for time in T:

    

    P = ((l * time)**k[0] * math.exp(-l *time))/(math.factorial(k[0]))

    P1.append(P)

    P = ((l * time)**k[1] * math.exp(-l *time))/(math.factorial(k[1]))

    P2.append(P)

    P = ((l * time)**k[2] * math.exp(-l *time))/(math.factorial(k[2]))

    P3.append(P)

    P = ((l * time)**k[3] * math.exp(-l *time))/(math.factorial(k[3]))

    P4.append(P)

    P = ((l * time)**k[4] * math.exp(-l *time))/(math.factorial(k[4]))

    P5.append(P)

    

    



plt.plot(T,P1, label = "k = 1")

plt.plot(T,P2, label = "k = 2")

plt.plot(T,P3, label = "k = 3")

plt.plot(T,P4, label = "k = 4")

plt.plot(T,P5, label = "k = 5")

plt.ylabel('probability')

plt.xlabel('time, sec')

plt.axis([0, 15, 0, 0.4])

plt.legend()



print("Max P1 = ", max(P1))

print("k = 1: lambda * t = ", l * T[P1.index(max(P1))])

print("Max P2 = ", max(P2))

print("k = 2: lambda * t = ", l * T[P2.index(max(P2))])

print("Max P3 = ", max(P3))

print("k = 3: lambda * t = ", l * T[P3.index(max(P3))])

print("Max P4 = ", max(P4))

print("k = 4: lambda * t = ", l * T[P4.index(max(P4))])

print("Max P5 = ", max(P5))

print("k = 5: lambda * t = ", l * T[P5.index(max(P5))])
mean = 1/l

dev = math.sqrt(1/l) # deviation

p1, p2, F1, F2 = [], [], [], [] # 1 - exponentional, 2 - normal

for time in T:

    p_1 = l * math.exp(-l * time)

    p1.append(p_1)

    

    p_2 = (math.exp(-(time - mean)**2/(2 * dev**2)))/(dev * math.sqrt(2 * math.pi))

    p2.append(p_2)

    

    F = 1 - math.exp(-l * time)

    F1.append(F)

plt.plot(T, p1,label = 'exponential distribution')

plt.plot(T, p2, label = 'normal distribution')

plt.ylabel('probability')

plt.xlabel('time, sec')

plt.legend()