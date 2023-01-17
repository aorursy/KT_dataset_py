# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

!pip install git+https://github.com/lisphilar/covid19-sir#egg=covsirphy
import covsirphy as cs

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

recovered_patients = pd.read_csv("/kaggle/input/covid19-in-turkey/time_series_covid_19_recovered_tr.csv")

deaths = pd.read_csv("/kaggle/input/covid19-in-turkey/time_series_covid_19_deaths_tr.csv")

cases = pd.read_csv("/kaggle/input/covid19-in-turkey/time_series_covid_19_confirmed_tr.csv")

N = 82000000 #constant population

infected = cases.iloc[0,4:]-recovered_patients.iloc[0,4:]- deaths.iloc[0,4:]

confirmed = cases.iloc[0,4:]
              
recovered = recovered_patients.iloc[0,4:]

fatal = deaths.iloc[0,4:]

infection_data = pd.DataFrame([confirmed, infected, fatal, recovered], index = ["Confirmed", "Infected", "Fatal", "Recovered"])

infection_data = infection_data.T
infection_data['Date'] = pd.date_range(start='3/11/2020', periods=len(infection_data), freq='D')
infection_data["Country"] = "Turkey"
infection_data["Province"] = "All"

infection_data.reindex(["Country","Province","Confirmed", "Infected", "Fatal", "Recovered"])
print(infection_data)

cs.line_plot(
    infection_data.set_index("Date")[["Infected", "Fatal", "Recovered"]],
    "Infected, Recovered and Fatal Subgroups of Covid-19 Patients in Turkey",
    h=N, y_integer=True
) #plot of the current situation with infected, fatal and the recovered subgroups

estimator = cs.Estimator(
    clean_df=infection_data, model=cs.SIRF, population=N,country = "Turkey", province = "All", tau=1440)
estimator.run() #estimates parameters for alpha, beta and gamma  

summary = estimator.summary(name="Turkey")

print(summary)

sird_model = {
    "kappa": summary.iloc[0,1], "rho": summary.iloc[0,2], "sigma": summary.iloc[0,3]
}
#using the estimated variables, we plot the s
sird_simulator = cs.ODESimulator(country="Turkey", province="All")
sird_simulator.add(
    model=cs.SIRD, step_n=180, population=N,
    param_dict=sird_model,
    y0_dict={"x": 0.999, "y": 0.001, "z": 0, "w": 0}
)
sird_simulator.run()
sird_simulator.non_dim().tail()

cs.line_plot(
    sird_simulator.non_dim().set_index("t"),
    title="Turkey Covid-19 SIRD modeling",
    ylabel="",
    h=1
)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Total population, N.
N = 82000000
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 1, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta, gamma = 0.2, 1./10 
# A grid of time points (in days)
t = np.linspace(0, 160, 160)

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number (1000s)')
ax.set_ylim(0,9600)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()