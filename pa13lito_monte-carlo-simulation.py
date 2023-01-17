import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import os

#print(os.listdir("../input"))
def pi_montecarlo(n, n_exp):

    pi_avg = 0

    pi_value_list = []

    for i in range(n_exp):

        value = 0

        x = np.random.uniform(0, 1, n).tolist()

        y = np.random.uniform(0, 1, n).tolist()

        for j in range(n):

            z = np.sqrt(x[j]*x[j] + y[j]*y[j])

            if z<1:

                value += 1

        float_value = float(value)

        pi_value = float_value * 4/n

        pi_value_list.append(pi_value)

        pi_avg += pi_value



    pi = pi_avg/n_exp



    print(pi)

    fig = plt.plot(pi_value_list)

    return (pi, fig)
pi_montecarlo(10000, 2000)