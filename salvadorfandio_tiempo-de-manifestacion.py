#from scipy.stats import lognorm

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
from math import erf, log, exp, sqrt



def generate_cdf(mean, var, n_days=20, daily_ticks=10):

    sigma = sqrt(log(var/mean**2 + 1))

    mu = log(mean) - 0.5*sigma**2

    sqrt2sigma_inv = 1.0 / (sqrt(2) * sigma)

    def cdf(x):

        return 0.5 + 0.5*erf(sqrt2sigma_inv*(log(x) - mu)) if x > 0 else 0



    n_ticks = n_days * daily_ticks

    days = list([i/daily_ticks for i in range(n_ticks)])

    acu = list([cdf(d) for d in days])

    return (days, acu)



def generate_cdf_and_df(mean, var, daily_ticks = 10, n_days=20):

    (days, cdf) = generate_cdf(mean, var, daily_ticks, n_days)

    df = []

    c0 = 0

    for c1 in cdf:

        df.append((c1 - c0)*daily_ticks)

        c0 = c1        

    return (days, cdf, df)
(days, _, df_3_7) = generate_cdf_and_df(3, 7)

(_, _, df_5_7) = generate_cdf_and_df(5, 7)

data = pd.DataFrame({'day': days, 'df_5_7': df_5_7, 'df_3_7': df_3_7})

with sns.axes_style("whitegrid"):

    sns.lineplot(data=data, x ='day', y='df_5_7')

    sns.lineplot(data=data, x ='day', y='df_3_7')
(days, _, df_5_5) = generate_cdf_and_df(5, 5)

(_, _, df_5_10) = generate_cdf_and_df(5, 10)

(_, _, df_5_20) = generate_cdf_and_df(5, 20)

data = pd.DataFrame({'day': days, 'df_5_5': df_5_5, 'df_5_10': df_5_10, 'df_5_20': df_5_20})

with sns.axes_style("whitegrid"):

    sns.lineplot(data=data, x ='day', y='df_5_5')

    sns.lineplot(data=data, x ='day', y='df_5_10')

    sns.lineplot(data=data, x ='day', y='df_5_20')
cols = {}

for var in range(5, 12):

    (day, cdf) = generate_cdf(3, var, n_days=25, daily_ticks=1)

    cols['day'] = day

    cols['var%i' % var] = cdf

df = pd.DataFrame(cols)

df
from math import erf, log, exp, sqrt



def simulate(incubation_mean, incubation_var, f1=0.6, f2=0.2, n_days=100, daily_ticks=10):



    n_ticks = n_days * daily_ticks



    sigma = sqrt(log(incubation_var/incubation_mean**2 + 1))

    mu = log(incubation_mean) - 0.5*sigma**2



    sqrt2sigma_inv = 1.0 / (sqrt(2) * sigma)

    def cdf(x):

        if x <= 0:

            return 0

        return 0.5 + 0.5*erf(sqrt2sigma_inv*(log(x) - mu))

    

    infected = list([0 for _ in range(n_ticks)])

    symptomatic = list([0 for _ in range(n_ticks)])

    infected_velocity = list([0 for _ in range(n_ticks)])

    symptomatic_velocity = list([0 for _ in range(n_ticks)])



    last_infected = 1

    last_symptomatic = 1

    for i in range(1, n_ticks):

        f = f1 if 2*i <= n_ticks else f2

        f_tick = (f + 1)**(1/daily_ticks) - 1

        new_infected = f_tick*last_infected

    

        infected[i] = last_infected + new_infected

        infected_velocity[i] = (infected[i]/last_infected)**daily_ticks

        symptomatic[i] += last_symptomatic

        symptomatic_velocity[i] = (symptomatic[i]/last_symptomatic)**daily_ticks



        last_infected = infected[i]

        last_symptomatic = symptomatic[i]

        for j in range(i+1, n_ticks):

            d0 = (j-i-1)/daily_ticks

            d1 = (j-i)/daily_ticks

            cf0 = cdf(d0)

            cf1 = cdf(d1)

            new_symptomatic = (cf1-cf0)*new_infected

            #print("d0: %f, d1: %f, cf0: %f, cf1: %f, df: %f, new_infected: %f, new_symptomatic: %f" %

            #      (d0, d1, cf0, cf1, cf1-cf0, new_infected, new_symptomatic))

            symptomatic[j] += new_symptomatic

            if cf1 == 1:

                break

    

    return({'day': [tick/daily_ticks for tick in range(n_ticks)],

            'symptomatic': symptomatic,

            'symptomatic_velocity': symptomatic_velocity,

            'infected': infected,

            'infected_velocity': infected_velocity})
a = simulate(5, 7, f1=0.4, f2=0.0)

b = simulate(5, 7, f1=0.4, f2=0.2)

c = simulate(3, 7, f1=0.4, f2=0.0)

d = simulate(5, 12, f1=0.4, f2=0.0)

e = simulate(5, 3, f1=0.4, f2=0.0)

r3 = simulate(3, 7, f1=0.4, f2=0.4)

r5 = simulate(5, 7, f1=0.4, f2=0.4)



data_a = pd.DataFrame(a)

data_b = pd.DataFrame(b)

data_c = pd.DataFrame(c)

data_d = pd.DataFrame(d)

data_e = pd.DataFrame(e)

data_r3 = pd.DataFrame(r3)

data_r5 = pd.DataFrame(r5)
with sns.axes_style("whitegrid"):

    fig, ax = plt.subplots(figsize=(16, 15))

    ax.set_yscale('log')

    sns.lineplot(data=data_a, x ='day', y='infected', ax=ax)

    sns.lineplot(data=data_a, x ='day', y='symptomatic', ax=ax)
with sns.axes_style("whitegrid"):

    fig, ax = plt.subplots(figsize=(16, 12))

    ax.set(ylim=(0.9, 1.5))

    sns.lineplot(data=data_a, x ='day', y='infected_velocity', ax=ax)

    sns.lineplot(data=data_b, x ='day', y='infected_velocity', ax=ax)

    sns.lineplot(data=data_a, x ='day', y='symptomatic_velocity', ax=ax)

    sns.lineplot(data=data_b, x ='day', y='symptomatic_velocity', ax=ax)

    sns.lineplot(data=data_c, x ='day', y='symptomatic_velocity', ax=ax)
data_a1 = data_a[(data_a.day > 49) & (data_a.day < 60)]

data_b1 = data_b[(data_b.day > 49) & (data_b.day < 60)]

data_c1 = data_c[(data_c.day > 49) & (data_c.day < 60)]

data_d1 = data_d[(data_d.day > 49) & (data_d.day < 60)]

data_e1 = data_e[(data_e.day > 49) & (data_e.day < 60)]

data_r31 = data_r3[(data_r3.day > 49) & (data_r3.day < 60)]

data_r51 = data_r5[(data_r5.day > 49) & (data_r5.day < 60)]



with sns.axes_style("whitegrid"):

    sns.axes_style("whitegrid")

    fig, ax = plt.subplots(figsize=(16, 15))

    ax.set_yscale('log')

    sns.lineplot(data=data_a1, x ='day', y='infected', ax=ax)

    sns.lineplot(data=data_a1, x ='day', y='symptomatic', ax=ax)

    sns.lineplot(data=data_b1, x ='day', y='symptomatic', ax=ax)

    sns.lineplot(data=data_c1, x ='day', y='symptomatic', ax=ax)

    #sns.lineplot(data=data_d1, x ='day', y='symptomatic', ax=ax)

    #sns.lineplot(data=data_e1, x ='day', y='symptomatic', ax=ax)

    sns.lineplot(data=data_r31, x ='day', y='symptomatic', ax=ax)

    sns.lineplot(data=data_r51, x ='day', y='symptomatic', ax=ax)
with sns.axes_style("whitegrid"):

    sns.axes_style("whitegrid")

    fig, ax = plt.subplots(figsize=(16, 15))

    ax.set_yscale('log')

    sns.lineplot(data=data_a1, x ='day', y='infected', ax=ax)

    sns.lineplot(data=data_a1, x ='day', y='symptomatic', ax=ax)

    sns.lineplot(data=data_d1, x ='day', y='symptomatic', ax=ax)

    sns.lineplot(data=data_e1, x ='day', y='symptomatic', ax=ax)

    sns.lineplot(data=data_r51, x ='day', y='symptomatic', ax=ax)