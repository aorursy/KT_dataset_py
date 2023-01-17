#Loosely based on python code available at https://en.wikipedia.org/wiki/Gillespie_algorithm



import random

import matplotlib.pyplot as plt

from math import log

import numpy as np



def model_sir_curve(recovery_rate, population, time, initial_infected, infection_rate_t ):

    

    N = population

    t_max = time

    

    #initial infected

    I = initial_infected 

    #initial recovered

    R = 0 

    #initial susceptable

    S = N - initial_infected 

    values_t = [] #basic python list - consider lower memory/higher performance solution

    values_i = [] #basic python list - consider lower memory/higher performance solution

    iterator = 0

    iterator_max = 4

    

    t=0

    while t < t_max:

        

        # find the infection rate at time t

        for i in infection_rate_t:

            if i[0] <= t and i[1] >= t:

                infection_rate = i[2]

                break

    

        if I==0:

            #no infections left

            break 

        

        #Step 3: Calculate the propensities of the reactions

        infection_constant = infection_rate * S * I / N 

        recovery_constant = (recovery_rate * I )

        constant_total = infection_constant + recovery_constant

    

        #Step 4: Choose event randomly (weighted by propensities)

        if random.uniform(0.0, 1.0) < infection_constant / constant_total :

            #we have an infection S->I

            S = S - 1

            I = I + 1

        else:

            # we have a recovery I-> R

            I = I - 1

            R = R + 1

    

        #Step 5: Choose random time

        dt = -log(random.uniform(0.0, 1.0)) / constant_total # 

        t = t + dt

        

        #dodgy hack to allow it to stay within memory bounds

        #only append every 5th calculated value to the list

        if iterator == iterator_max:

            values_t.append(t)

            values_i.append(I)

            iterator = 0

        else:

            iterator = iterator + 1

            

    return values_t, values_i



#modelling recovery rate as the same across models - 

#randomly selected -> *** needs cited value from somewhere ***

recovery_rate = 0.1



#R0 = 2.5 -> this is the baseline

baseline = "R0 = 2.5"

baseline_infection = np.array([[0,365,.25]])

values_t1, values_i1 = model_sir_curve(recovery_rate, 24600000, 365, 5 ,baseline_infection) 



#R0 = 2.5 to 1.625 after 40 days.

edited = "R0 = 2.5 then R0 = 1.625 after 40 Days"

edited_infection = np.array([[0,40,.25],[41,365,.1625]])

values_t2, values_i2 = model_sir_curve(recovery_rate, 24600000, 365, 5, edited_infection) 



# finally, graph it all.

plt.figure(figsize=(14,10))

plt.xlabel("Time [Days]")

plt.ylabel("Population Infected")

plt.plot(values_t1,values_i1)

plt.plot(values_t2, values_i2)

plt.legend([baseline, edited])

plt.show()