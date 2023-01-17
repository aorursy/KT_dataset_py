import numpy as np

from scipy.stats import beta, t, norm

from scipy.special import btdtri

import matplotlib.pyplot as plt
p = 0.5

n = 10



success = np.random.binomial(p=p, n=n)

failure = n - success

print("success = %i, failure = %i"%(success, failure))
prior_a = 1

prior_b = 1



a = prior_a + success

b = prior_b + failure

rv = beta(a, b)



b_up = btdtri(a, b, 0.975)

b_lo = btdtri(a, b, 0.025)



print("95%% credible interval: [%.3f, %.3f]"%(b_lo, b_up))
p_hat = success / n

se = np.sqrt(p_hat * (1 - p_hat) / n)



f_up = p_hat + 1.96 * se

f_lo = p_hat - 1.96 * se



print("95%% confidence interval: [%.3f, %.3f]"%(f_lo, f_up))
fig = plt.figure(figsize=(14, 4))

grid = plt.GridSpec(1, 2, hspace=0.2, wspace=0.2)



ax1 = fig.add_subplot(grid[:, :1])

ax2 = fig.add_subplot(grid[:, 1:])



# bayesian credible interval

x = np.linspace(0, 1, 1000)

ax1.plot(x, rv.pdf(x), color='blue')



# plot prior if necessary

rv_prior = beta(prior_a, prior_b)

ax1.plot(x, rv_prior.pdf(x), alpha=0.2)



# bayesian credible interval

right_line = ax1.axvline(b_up, lw=2, color='blue')

left_line = ax1.axvline(b_lo, lw=2, color='blue')

fill = ax1.axvspan(b_lo, b_up, alpha=0.2, color='blue')



ax1.set_xlabel("95%% credible interval: [%.3f, %.3f]"%(b_lo, b_up))

ax1.legend(["Beta(\u03B1=%i, \u03B2=%i)"%(a, b), "flat prior"], frameon=False)



ax1.spines["top"].set_visible(False)

ax1.spines["right"].set_visible(False)

ax1.spines["left"].set_visible(False)

ax1.spines["bottom"].set_visible(False)

    

# frequentist confidence interval

ax2.plot(x, norm.pdf(x, loc=p_hat, scale=se), color='r')

right_line = ax2.axvline(f_up, lw=2, color='r')

left_line = ax2.axvline(f_lo, lw=2, color='r')

fill = ax2.axvspan(f_lo, f_up, alpha=0.2, color='r')



ax2.set_xlabel("95%% confidence interval: [%.3f, %.3f]"%(f_lo, f_up))

ax2.legend(["N (\u03BC=%.2f, \u03C3=%.2f)"%(p_hat, se)], frameon=False)



ax2.spines["top"].set_visible(False)

ax2.spines["right"].set_visible(False)

ax2.spines["left"].set_visible(False)

ax2.spines["bottom"].set_visible(False)