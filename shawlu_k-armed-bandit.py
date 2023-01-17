import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
class Arm(object):

    """

    Each arm's true click through rate is 

    modeled by a beta distribution.

    """

    def __init__(self, idx, a=1, b=1):

        """

        Init with uniform prior.

        """

        self.idx = idx

        self.a = a

        self.b = b

        

    def record_success(self):

        self.a += 1

        

    def record_failure(self):

        self.b += 1

        

    def draw_ctr(self):

        return np.random.beta(self.a, self.b, 1)[0]

    

    def mean(self):

        return self.a / (self.a + self.b)
def monte_carlo_simulation(arms, draw=100):

    """

    Monte Carlo simulation of thetas. Each arm's click through

    rate follows a beta distribution.

    

    Parameters

    ----------

    arms list[Arm]: list of Arm objects.

    draw int: number of draws in Monte Carlo simulation.

    

    Returns

    -------

    mc np.matrix: Monte Carlo matrix of dimension (draw, n_arms).

    p_winner list[float]: probability of each arm being the winner.

    """

    # Monte Carlo sampling

    alphas = [arm.a for arm in arms]

    betas = [arm.b for arm in arms]

    mc = np.matrix(np.random.beta(alphas, betas, size=[draw, len(arms)]))

    

    # count frequency of each arm being winner 

    counts = [0 for _ in arms]

    winner_idxs = np.asarray(mc.argmax(axis=1)).reshape(draw,)

    for idx in winner_idxs:

        counts[idx] += 1

    

    # divide by draw to approximate probability distribution

    p_winner = [count / draw for count in counts]

    return mc, p_winner
def thompson_sampling(arms):

    """

    Stochastic sampling: take one draw for each arm

    divert traffic to best draw.

    

    @param arms list[Arm]: list of Arm objects

    @return idx int: index of winning arm from sample

    """

    sample_p = [arm.draw_ctr() for arm in arms]

    idx = np.argmax(sample_p)

    return idx
def should_terminate(p_winner, est_ctrs, mc, alpha=0.05):

    """

    Decide whether experiument should terminate. When value remaining in

    experiment is less than 1% of the winning arm's click through rate.

    

    Parameters

    ----------

    p_winner list[float]: probability of each arm being the winner.

    est_ctrs list[float]: estimated click through rates.

    mc np.matrix: Monte Carlo matrix of dimension (draw, n_arms).

    alpha: controlling for type I error

    

    @returns bool: True if experiment should terminate.

    """

    winner_idx = np.argmax(p_winner)

    values_remaining = (mc.max(axis=1) - mc[:, winner_idx]) / mc[:, winner_idx]

    pctile = np.percentile(values_remaining, q=100 * (1 - alpha))

    return pctile < 0.01 * est_ctrs[winner_idx]
def k_arm_bandit(ctrs, alpha=0.05, burn_in=1000, max_iter=100000, draw=100, silent=False):

    """

    Perform stochastic k-arm bandit test. Experiment is terminated when

    value remained in experiment drops below certain threshold.

    

    Parameters

    ----------

    ctrs list[float]: true click through rates for each arms.

    alpha float: terminate experiment when the (1 - alpha)th percentile

        of the remaining value is less than 1% of the winner's click through rate.

    burn_in int: minimum number of iterations.

    max_iter int: maxinum number of iterations.

    draw int: number of rows in Monte Carlo simulation.

    silent bool: print status at the end of experiment.

    

    Returns

    -------

    idx int: winner's index.

    est_ctrs list[float]: estimated click through rates.

    history_p list[list[float]]: storing est_ctrs and p_winner.

    traffic list[int]: number of traffic in each arm.

    """

    n_arms = len(ctrs)

    arms = [Arm(idx=i) for i in range(n_arms)]

    history_p = [[] for _ in range(n_arms)]

    

    for i in range(max_iter):

        idx = thompson_sampling(arms)

        arm, ctr = arms[idx], ctrs[idx]



        # update arm's beta parameters

        if np.random.rand() < ctr:

            arm.record_success()

        else:

            arm.record_failure()



        # record current estimates of each arm being winner

        mc, p_winner = monte_carlo_simulation(arms, draw)

        for j, p in enumerate(p_winner):

            history_p[j].append(p)

            

        # record current estimates of each arm's ctr

        est_ctrs = [arm.mean() for arm in arms]

        

        # terminate when value remaining is negligible

        if i >= burn_in and should_terminate(p_winner, est_ctrs, mc, alpha):

            if not silent: print("Terminated at iteration %i"%(i + 1))

            break



    traffic = [arm.a + arm.b - 2 for arm in arms]

    return idx, est_ctrs, history_p, traffic
def plot_history(ctrs, est_ctrs, df_history, title, rolling=10, fname=None, transparent=False):

    """

    Plot evolution of conversion rates estimates or winner probability for each arm.

    

    Parameters

    ----------

    ctr, est_ctrs list[float]: true ctrs and estiamted ctrs.

    df_history list[list[float]]: a nested list of each arm's history.

    rolling int: rolling window length.

    fname str: enter file name if need to store, including '.png'.

    transparent bool: make background transparent.

    """

    true_winner_idx = np.argmax(ctrs)

    winner_idx = np.argmax(est_ctrs)

    

    cols = ["arm_%i_ctr=%.2f"%(i + 1, ctr) for i, ctr in enumerate(ctrs)]

    data = {col : hist for col, hist in zip(cols, df_history)}

    df_history_ma = pd.DataFrame(data).rolling(rolling).mean()

    

    plt.figure(figsize=(12, 4))

    for i, col in enumerate(cols):

        if i == true_winner_idx :

            plt.plot(df_history_ma[col], lw=2, color='b')

        elif i == winner_idx:

            plt.plot(df_history_ma[col], lw=2, color='r')

        else:

            plt.plot(df_history_ma[col], alpha=0.5)



    legend = ["true ctr = %.3f, est ctr = %.3f"%(true, est) for true, est in zip(ctrs, est_ctrs)]

    plt.legend(legend, frameon=False, loc='upper center', ncol=3)

    plt.title(title)

    plt.ylim(0, 1)

    

    plt.show()
seed = 11

np.random.seed(seed)



ctrs = [0.04, 0.048, 0.03, 0.037, 0.044]

true_winner_idx = np.argmax(ctrs)

print("true_winner_idx:", true_winner_idx, ctrs)



(winner_idx, est_ctrs, history_p, traffic) = k_arm_bandit(ctrs, alpha=0.05, burn_in=1400)
plot_history(ctrs, est_ctrs, history_p, 

             title="K-armed Bandit Algorithm (terminated in %i iterations)"%sum(traffic), rolling=100)