import numpy as np

from scipy.integrate import odeint

import matplotlib.pyplot as plt
def deriv(y, t, N, beta, gama):

    S, I, R = y

    dSdt = -beta * S * I / N

    dIdt = beta * S * I / N - gama * I

    dRdt = gama * I

    return dSdt, dIdt, dRdt



def sir(N, beta, gama=1/10, I0=1, R0=0, t=90):

    t = np.linspace(0, t, t)

    S0 = N - I0 - R0

    y0 = S0, I0, R0

    ret = odeint(deriv, y0, t, args=(N, beta, gama))

    S, I, R = ret.T

    return {'S': S, 'I': I, 'R': R, 't': t}



def curves(s, title):

    fig, ax = plt.subplots(figsize=(12,8))

    plt.plot(s['t'], s['S'], 'b', alpha=0.5, lw=4, label='Susceptible')

    plt.plot(s['t'], s['I'], 'r', alpha=0.5, lw=4, label='Infected')

    plt.plot(s['t'], s['R'], 'g', alpha=0.5, lw=4, label='Recovered')

    plt.grid(which='major', axis='y')

    plt.ticklabel_format(scilimits=(6,6), axis='y')

    plt.text(0.9,1,s='β: %.3f γ: %.2f' % (beta, gama), transform=ax.transAxes, fontsize=10)

    plt.title(title, fontsize=14, fontweight='bold', color='#333333')

    plt.xlabel('Days', fontsize=12)

    plt.ylabel('Number of cases (million)', fontsize=12)

    legend = plt.legend(loc=5, fontsize=12)

    ax.set_ylim(0)

    [ax.spines[spine].set_visible(False) for spine in ('top', 'right', 'left')]

    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(10))

    plt.show();
gama = 1/5.2

beta = 2.2

curves(sir(209300000 , beta, gama, t=120, I0=100), 'COVID-19 SIR Model in Brazil: No-action')
beta = 0.88

curves(sir(209300000, beta, gama, t=120, I0=100), 'COVID-19 SIR Model in Brazil: Social Distancing')
beta = 0.449

curves(sir(209300000, beta, gama, t=120, I0=100), 'COVID-19 SIR Model in Brazil: Social Distancing + Quarantine')