# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

# colocar na variavel datasetrz os r(ts)para cada dia

def run_SEIR_ODE_model(
        N: 'population size',
        E0: 'init. exposed population',
        I0: 'init. infected population',
        R0: 'init. removed population',
        beta: 'infection probability',
        gamma: 'removal probability', 
        alpha_inv: 'incubation period', 
        t_max: 'numer of days to run'
    ) -> pd.DataFrame:

    S0 = N - I0 - R0 - E0
    alpha = 1/alpha_inv

    # A grid of time points (in days)
    t = range(t_max)

    # The SEIR model differential equations.
    def deriv(y, t, N, beta, gamma, alpha):
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = -dSdt - alpha*E
        dIdt = alpha*E - gamma*I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    # Initial conditions vector
    y0 = S0, E0, I0, R0

    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma, alpha))
    S, E, I, R = ret.T

    return pd.DataFrame({'S': S, 'E': E, 'I': I, 'R': R}, index=t)


if __name__ == '__main__':
    #Re_series = pd.Series(range(700,1)) # aqui tem que ir a sequencia de dias do rt
    
    dffloripa = pd.read_csv("../input/floripa/Re.csv", sep=',')
    dffloripa = dffloripa.sort_values('date', ignore_index=True)
    re_floripa = dffloripa.iloc[:76,13:]
    csv_de_rzeros = re_floripa.iloc[:,1] # iloc 0 para média, 1 para limite superior 2 para inferior
    Re_series = pd.Series(csv_de_rzeros.iloc[:])
    
    #Initial parameters
    
    N = 500_000
    E0, I0, R0 = 35, 35, 0
    Sinicial = N - I0 - R0 - E0
    gamma, alpha_inv = 0.31, 2.9 
    
    for r_column in ['Re_lb', 'Re', 'Re_ub']:
        #solves several ODEs in sequence using r(e) from the Re_series
        e_series = re_floripa[r_column]
        errezero = Re_series.iloc[0]
        beta = gamma*errezero
        t_max = 2
        fitado = pd.DataFrame({'S': [Sinicial], 'E': [E0], 'I': [I0], 'R': [R0]})
        j = 0
        for i in range(len(Re_series)-t_max):
            j = j + (t_max-1)
            results = run_SEIR_ODE_model(N, E0, I0, R0, beta, gamma, alpha_inv, t_max)
            fitado = pd.concat([fitado, results.loc[1:,:]], ignore_index=True)
            E0, I0, R0 = fitado.loc[j,'E'],fitado.loc[j,'I'], fitado.loc[j,'R']
            effectiver = Re_series.iloc[j]
            errezero = effectiver*(N)/(N-(fitado.loc[j,'E'])-(fitado.loc[j,'I'])-(fitado.loc[j,'R']))
            beta = gamma*errezero

        # plot
        #variable
        plt.style.use('ggplot')
        (fitado
         # .div(1_000_000)
         [['E', 'I', 'R']]
         .plot(figsize=(8,6), fontsize=20, logy=False))
        params_title = (
            f'SEIR($\gamma$={gamma}, $\\alpha$={1/alpha_inv}, $N$={N}, '
            f'$E_0$={int(E0)}, $I_0$={int(I0)}, $R_0$={int(R0)})'
        )
        plt.title(f'Numero de Pessoas Atingidas com modelo no "t_final -1" da simulacao:\n r(e)=vide lista, ' + params_title,
                  fontsize=20)
        plt.legend(['Expostas', 'Infectadas', 'Recuperadas'], fontsize=20)
        plt.xlabel('Dias', fontsize=20)
        plt.ylabel('Pessoas', fontsize=20)
        plt.show()