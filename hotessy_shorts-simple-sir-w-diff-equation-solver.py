import numpy as np 

import pandas as pd 



from path import Path

import os

from functools import partial



from ipywidgets import interact, interact_manual, fixed

from IPython.display import display



import cufflinks as cf

import matplotlib.pyplot as plt



from sklearn.linear_model import LinearRegression

from scipy.integrate import odeint, solve_ivp
pd.set_option('display.max_rows', 500)

pd.set_option('use_inf_as_na', True)

cf.set_config_file(offline=True, theme='solar');
def solve_I(β, ɣ, N, I, S):



    dI_dt = β*(I/N)*S - ɣ*I

    return dI_dt
def solve_S(β, N, I, S):



    dS_dt = - β*(I/N)*S

    return dS_dt
def solve_R(ɣ, I):

    dR_dt = ɣ*I

    return dR_dt
def sir_model(β, ɣ, N, t, Y):

    

    I, S, R = Y

    dS_dt = - β*(I/N)*S

    dR_dt = ɣ*I

    dI_dt = - dS_dt - dR_dt

    

    return [dI_dt, dS_dt, dR_dt]
@ interact(β=(0.0, 1., 0.01), ɣ=(0.01, 1., 0.01))

def sir_play(β=0.15, ɣ=0.15):

    

    

    N = 10_000

    I0 = 10

    R0 = 0

    S0 = N - I0 - R0

    

    t = np.linspace(0, 90, 180)

    

#     I = odeint(func=partial(sir_model, β, ɣ, N), y0=[I0, S0, R0], t=t, tfirst=True)

    I = solve_ivp(fun=partial(sir_model, β, ɣ, N), y0=[I0, S0, R0], t_eval=t, t_span=(0, len(t))).y.T



    print(f"Reproduction Rate is {β/ɣ}")

    pd.DataFrame(index=t, data=I, columns=['I', 'S', 'R']).iplot(y=['I', 'R'], secondary_y=['S'])