import matplotlib.pyplot as plt

import numpy as np

from scipy.stats import norm
class LogVarianceNorm:

    def calculate_gradient(self, t, m, log_var):

        var =  np.exp(log_var)

        gradients = [

            (m - t) / var,

            1 - ((t - m) ** 2) / var

        ]



        fisher_matrix = [

            [1 / var, 0],

            [0, 1]

        ]



        return np.array(gradients), np.array(fisher_matrix)

    

    def score(self, t, m, log_var):

        var = np.exp(log_var)

        return - norm.logpdf(t, loc=m, scale=var ** .5)

def true_function(X):

    return np.sin(3 * X)



def true_noice_scale(X):

    return abs(np.sin(X))



n_samples = 200

np.random.seed(71)

X = np.random.uniform(-3, 1, n_samples)

y = true_function(X) + np.random.normal(scale=true_noice_scale(X), size=n_samples)
xx = np.linspace(-3.3, 1.2, 301)

plt.scatter(X, y)

plt.plot(xx, true_function(xx))
from sklearn.tree import DecisionTreeRegressor



def get_weak_learner():

    return DecisionTreeRegressor(max_depth=3, min_samples_leaf=3)


n_samples = 200

np.random.seed(71)

X = np.random.uniform(-3, 1, n_samples)

y = true_function(X) + np.random.normal(scale=abs(X), size=n_samples)



log_var_norm = LogVarianceNorm()



# hyper parameters

learning_rate = 2e-2

n_iterations = 1000



estimators = []

step_sizes = []



X = X.reshape(-1, 1)

N = len(X)



# initialize parameters by whole dataset mean and variance

m, log_var = np.zeros(shape=(N,)), np.zeros(shape=(N,))

m += np.mean(y)

log_var += np.log(np.var(y) ** .5)



for iteration in range(n_iterations):



    # 1. compute natural gradient

    grads = []

    for i in range(N):

        score, fisher = log_var_norm.calculate_gradient(y[i], m[i], log_var[i])

        grad = np.linalg.solve(fisher, score)

        grads.append(grad)

    grads = np.array(grads)



    # 2. fit weak learner

    # mean estimator

    clf_mean = get_weak_learner()

    clf_mean.fit(X, y=grads[:, 0])



    # log_var estimator

    clf_var = get_weak_learner()

    clf_var.fit(X, y=grads[:, 1])



    directions = np.zeros(shape=(N, 2))

    directions[:, 0] = clf_mean.predict(X)

    directions[:, 1] = clf_var.predict(X) 

    estimators.append(

        [clf_mean, clf_var]

    )



    # 3. linear search

    scores = []

    stepsize_choices = np.linspace(.5, 2, 21)

    for stepsize in stepsize_choices:

        d = directions * stepsize

        score_i = log_var_norm.score(y, m - d[:, 0], log_var - d[:, 1]).mean()

        scores.append(score_i)



    best_idx = np.argmin(scores)

    rho_i = stepsize_choices[best_idx]

    

    stepsize_i = learning_rate * rho_i

    step_sizes.append(stepsize_i)



    # 4. update parameters

    grad_parameters = directions * stepsize_i

    m -= grad_parameters[:, 0]

    log_var -= grad_parameters[:, 1]

    

    if iteration % 50 == 0:

        print(f'[iter: {iteration}]\tscore: {scores[best_idx]:.4f}\tstepsize: {rho_i:.3f}')
xx = np.linspace(-4, 2, 501) # predict input X



xx_mean = np.zeros_like(xx) + np.mean(y)

xx_log_var = np.zeros_like(xx) + np.log(np.var(y) ** .5)



# Prediction results of all weak learners are weighted and added together

for i in range(len(estimators)):

    xx_mean -= step_sizes[i] * estimators[i][0].predict(xx.reshape(-1, 1))

    xx_log_var -= step_sizes[i] * estimators[i][1].predict(xx.reshape(-1, 1))



xx_var = np.exp(xx_log_var)
fig, axes = plt.subplots(figsize=(10, 8), nrows=2, sharex=True)

axes[0].scatter(X, y, label='Training Points')

axes[0].plot(xx, xx_mean, c='C1', label='Predict (mean)')

axes[0].fill_between(xx, xx_mean - xx_var ** .5, xx_mean + xx_var ** .5, alpha=.3, color='C1', label='One Sigma')

axes[0].legend()



axes[1].plot(xx, xx_var ** .5, label='Predict (std)')

axes[1].plot(xx, true_noice_scale(xx), label='Groud Truth')

axes[1].set_ylabel('model uncertainty')

axes[1].legend()

fig.tight_layout()