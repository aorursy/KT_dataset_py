import numpy as np

import scipy.stats as stats

import matplotlib.pyplot as plt
# hyperparameters for prior losses generating (some estimation from leaderboard)

L0, SIGMA0 = 6.8, 2



# small noise deviation

EPS = 0.1
def plot_solution(n_cv, n_test, sigma, ax):

    ## hyperparameters

    n_pub, n_priv = 0.15 * n_test, 0.85 * n_test



    ## prior losses

    l0_CV_samples = np.random.normal(loc=L0, scale=SIGMA0, size=int(n_cv))

    l0_pub_samples = np.random.normal(loc=L0, scale=SIGMA0, size=int(n_pub))

    l0_priv_samples = np.random.normal(loc=L0, scale=SIGMA0, size=int(n_priv))



    ## mean losess 

    l0_CV = l0_CV_samples.mean()

    l0_pub = l0_pub_samples.mean()

    l0_priv = l0_priv_samples.mean()

    

    ## distributions

    l_CV_samples = np.random.normal(loc=l0_CV, scale=EPS, size=5)

    l_pub_samples = np.random.normal(loc=l0_pub, scale=sigma, size=5)

    l_priv_samples = np.random.normal(loc=l0_priv, scale=sigma, size=5)

    

    ## ## distributions distributions & samples

    x = np.linspace(L0 - 1*SIGMA0, L0 + 1*SIGMA0, 100)



    ax.plot(x, stats.norm.pdf(x, l0_CV, EPS), label="CV")

    _ = ax.scatter(l_CV_samples, 0.2 * np.random.rand(5), marker='x')



    ax.plot(x, stats.norm.pdf(x, l0_pub, sigma), label="public")

    _ = ax.scatter(l_pub_samples, 0.2 * np.random.rand(5), marker='x')



    ax.plot(x, stats.norm.pdf(x, l0_priv, sigma), label="private")

    _ = ax.scatter(l_priv_samples, 0.2 * np.random.rand(5), marker='x')



    _ = ax.legend()

    

def plot_solutions(n_cv, n_test, sigma):

    # repeat plot_solution

    nrows, ncols = 3, 2

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 12))



    for i in range(nrows):

        for j in range(ncols):

            plot_solution(n_cv=n_cv, n_test=n_test, sigma=sigma, ax=ax[i][j])
plot_solutions(n_cv=1e3, n_test=1e3, sigma=EPS)
plot_solutions(n_cv=1e3, n_test=1e3, sigma=0.3)
plot_solutions(n_cv=1e3, n_test=1e3, sigma=1)
plot_solutions(n_cv=2 * 176, n_test=200, sigma=0.3)