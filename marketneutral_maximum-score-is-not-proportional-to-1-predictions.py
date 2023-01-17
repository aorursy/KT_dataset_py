import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
np.random.seed(seed=100)
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize'] = (10, 7)
n_days = 20
r_mean = np.array([0.025, 0.075])
epsilon = np.array([0.05, 0.05])
w_1 = np.repeat(1/r_mean[0], n_days)
w_2 = np.repeat(1/r_mean[1], n_days)

print(w_1)
print(w_2)
r_1 = np.random.normal(loc=r_mean[0], scale=epsilon[0], size=n_days)
r_2 = np.random.normal(loc=r_mean[1], scale=epsilon[1], size=n_days)

plt.plot(r_1);
plt.plot(r_2);
plt.title('Single Realization of Generating Process');
def score(w1, w2, r1, r2):
    x = w1*r1 + w2*r2
    return np.mean(x)/np.std(x)
score(w_1, w_2, r_1, r_2)
# the one-period expected two asset information ratio given weights, predictions, and noise
# I make it negative because we are going to optimize to find the maximum, but scipy will only find the minimum,
#  so we find the minimum of the negative to get the maximum
def information_ratio_2(w, y_hat, epsilon):
    r = w[0]*y_hat[0] + w[1]*y_hat[1]
    s = np.sqrt(w[0]*w[0]*epsilon[0]*epsilon[0] + w[1]*w[1]*epsilon[1]*epsilon[1])
    return -r/s
bounds = ((-1,1), (-1,1))
res = minimize(
    information_ratio_2,
    np.array([1.0, 1.0]),
    args=(r_mean, epsilon),
    bounds=bounds, 
    method='SLSQP'
)
print(res.x)
w_optimal_1 = np.repeat(res.x[0], n_days)
w_optimal_2 = np.repeat(res.x[1], n_days)

print(w_optimal_1)
print(w_optimal_2)
score(w_optimal_1, w_optimal_2, r_1, r_2)
one_over_r = []
alt = []
ex_post = []
n_sims = 10000
    
# set weights for the 1/r policy
w_1 = np.repeat(1/r_mean[0], n_days)
w_2 = np.repeat(1/r_mean[1], n_days)

# set weights for the alternative policy
bounds = ((-1,1), (-1,1))
res = minimize(
    information_ratio_2,
    np.array([1.0, 1.0]),
    args=(r_mean, epsilon),
    bounds=bounds, 
    method='SLSQP'
)
w_optimal_1 = np.repeat(res.x[0], n_days)
w_optimal_2 = np.repeat(res.x[1], n_days)

for i in range(n_sims):
    r_1 = np.random.normal(loc=r_mean[0], scale=epsilon[0], size=n_days)
    r_2 = np.random.normal(loc=r_mean[1], scale=epsilon[1], size=n_days)

    # run the 1/r weights
    trial_score = score(w_1, w_2, r_1, r_2)
    one_over_r.append(trial_score)
    
    # run the one-period optimal weights
    trial_score = score(w_optimal_1, w_optimal_2, r_1, r_2)
    alt.append(trial_score)
    
    # run the "leak scenario"; we know the actual realizations
    w_1_expost = 1/r_1
    w_2_expost = 1/r_2
    trial_score = score(w_1_expost, w_2_expost, r_1, r_2)
    ex_post.append(trial_score)
    
plt.hist(one_over_r, alpha=0.5);
plt.hist(alt, alpha=0.5);
plt.legend(['1/r', 'alternative']);
plt.title('Comparison to the 1/r policy and Optimal Policy for %d Simulations' % n_sims);
plt.xlabel('Score');
np.mean(ex_post)  # :-)
