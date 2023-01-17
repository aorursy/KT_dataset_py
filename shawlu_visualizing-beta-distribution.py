from scipy.stats import beta

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
def simulate_beta(prior_a, prior_b, p, n_trials):

    """

    Run Bayesian updating on prior, given true parameter.

    

    Parameters

    ----------

    prior_a int: prior alpha

    prior_b int: prior beta

    p float: true Bernoulli parameter

    n_trials int: number of iterations to run

    

    Return

    ------

    df DataFrame: parameters for Bernoulli trial p during each iteration

    df_ DataFrame: parameters for Bernoulli trial 1 - p during each iteration

    """

    a = prior_a

    b = prior_b



    df = pd.DataFrame({"a" : [a], "b" : [b]})

    df_ = pd.DataFrame({"a" : [a], "b" : [b]})

    for _ in range(n_trials):

        if np.random.rand() < p:

            a, b = df.iloc[-1].values

            a += 1

            df = df.append({"a" : a, "b" : b}, ignore_index=True)

            

            a, b = df_.iloc[-1].values

            b += 1

            df_ = df_.append({"a" : a, "b" : b}, ignore_index=True)

        else:

            a, b = df.iloc[-1].values

            b += 1

            df = df.append({"a" : a, "b" : b}, ignore_index=True)

            

            a, b = df_.iloc[-1].values

            a += 1

            df_ = df_.append({"a" : a, "b" : b}, ignore_index=True)

        

    return df, df_
n_trials = 1000

prior_a = 100

prior_b = 100



p = 0.2

df, df2 = simulate_beta(prior_a, prior_b, p=p, n_trials=n_trials)
x = np.linspace(0, 1, 1000)

fig, ax = plt.subplots(figsize=(12,4))



# -------------------------------------------

# plot prior

rv_prior = beta(prior_a, prior_b)

ax.plot(x, rv_prior.pdf(x), alpha=0.2)



# -------------------------------------------

# first graph

a, b = df.iloc[-1].values

rv = beta(a, b)

line, = ax.plot(x, rv.pdf(x), alpha=0.8, color='#ff7f0e')



# -------------------------------------------

# second graph

a2, b2 = df2.iloc[-1].values

rv2 = beta(a2, b2)

line2, = ax.plot(x, rv2.pdf(x), alpha=0.8, color='C2')



# -------------------------------------------

# ax.axvline(prior_a / (prior_a + prior_b), alpha=0.2, linestyle="--")

ax.axvline(p, alpha=0.2, linestyle="--", color='#ff7f0e')

ax.axvline(1-p, alpha=0.2, linestyle="--", color='C2')



# -------------------------------------------

# accessory

legend = [" prior p=%.3f, Beta(\u03B1=%i, \u03B2=%i)"%(prior_a / (prior_a + prior_b), prior_a, prior_b), 

          "mean p=%.3f, Beta(\u03B1=%i, \u03B2=%i)"%(a / (a + b), a, b),

          "mean p=%.3f, Beta(\u03B1=%i, \u03B2=%i)"%(a2 / (a2 + b2), a2, b2)]

ax.legend(legend, frameon=False, loc="upper center")



# -------------------------------------------

# minimalism style

ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=True)

for spine in plt.gca().spines.values():

    spine.set_visible(False)



ax.set_ylim(0, 50)



plt.show()