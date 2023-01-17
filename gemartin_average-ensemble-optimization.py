import numpy as np
from sklearn.datasets import load_boston

boston = load_boston()
X = boston.data
y = boston.target
features = boston.feature_names
print(features)
X = X[:,5].reshape(-1, 1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(X_train, y_train, facecolor=None, edgecolor='royalblue', alpha=.3)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel('Average number of rooms')
ax.set_ylabel('Median value of homes ($1000)')

plt.show()
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

model_1 = LinearRegression()
model_2 = DecisionTreeRegressor(max_depth=3, random_state=0)
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

kf = KFold(n_splits=5, random_state=0)

nrows_trn = X_train.shape[0]

mod1_oof_trn = np.empty(nrows_trn)
mod2_oof_trn = np.empty(nrows_trn)

mod1_scores = np.empty(5)
mod2_scores = np.empty(5)

for k, (trn_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_trn, X_val = X_train[trn_idx], X_train[val_idx]
    y_trn, y_val = y_train[trn_idx], y_train[val_idx]
    
    model_1.fit(X_trn, y_trn)
    mod1_oof_trn[val_idx] = model_1.predict(X_val)
    mod1_scores[k] = mean_squared_error(y_val, mod1_oof_trn[val_idx])
    
    model_2.fit(X_trn, y_trn)
    mod2_oof_trn[val_idx] = model_2.predict(X_val)
    mod2_scores[k] = mean_squared_error(y_val, mod2_oof_trn[val_idx])
    
model_1.fit(X_train, y_train)
model_2.fit(X_train, y_train)

mod1_predictions = model_1.predict(X_test)
mod2_predictions = model_2.predict(X_test)
import matplotlib.gridspec as gridspec

print('Model 1 CV score: {:.4f} ({:.4f})'.format(mod1_scores.mean(),
                                                 mod1_scores.std()))
print('Model 2 CV score: {:.4f} ({:.4f})'.format(mod2_scores.mean(),
                                                 mod2_scores.std()))

fig = plt.figure(figsize=(10, 10))
G = gridspec.GridSpec(2, 2,
                     height_ratios=[1, 2])

ax1 = plt.subplot(G[0, :])
ax1.boxplot([mod1_scores, mod2_scores])
ax1.set_title('Models CV scores and regression lines', fontsize=16)
ax1.spines['bottom'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xticklabels(['Model 1', 'Model 2'], fontsize=12)
ax1.tick_params(bottom=False)

ax2 = plt.subplot(G[1, 0])

X_plot = np.arange(X_train.min(), X_train.max()).reshape(-1, 1)
y_plot = model_1.predict(X_plot)

ax2.scatter(X_train, y_train, facecolor=None, edgecolor='royalblue', alpha=.3)
ax2.plot(X_plot, y_plot, 'orange')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xlabel('Average Number of rooms')
ax2.set_ylabel('Median value of homes')

ax3 = plt.subplot(G[1, 1], sharey=ax2)

X_plot = np.arange(X_train.min(), X_train.max()).reshape(-1, 1)
y_plot = model_2.predict(X_plot)

ax3.scatter(X_train, y_train, facecolor=None, edgecolor='royalblue', alpha=.3)
ax3.plot(X_plot, y_plot, 'orange')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_xlabel('Average number of rooms')

plt.tight_layout()
plt.show()
train_predictions = np.concatenate([mod1_oof_trn[:, None],
                                    mod2_oof_trn[:, None]], axis=1)

# Preview the first five rows
train_predictions[:5]
def objective(weights):
    """ Calculate the score of a weighted average of predictions
    
    Parameters
    ----------
    weights: array
        the weights applied to the average of the base predictions
        
    Returns
    -------
    float
        The mean_squared_error score of the ensemble
    """
    y_ens = np.average(train_predictions, axis=1, weights=weights)
    return mean_squared_error(y_train, y_ens)
from scipy.optimize import minimize

# I define initial weights from which the algorithm will try searching a minima
# I usually set the initial weigths to be the same for each columns, but they
# can be set randomly
w0 = np.empty(train_predictions.shape[1])
w0.fill(1 / train_predictions.shape[1])

# I define bounds, i.e. lower and upper values of weights.
# I want the weights to be between 0 and 1.
bounds = [(0,1)] * train_predictions.shape[1]

# I set some constraints. Here, I want the sum of the weights to be equal to 1
cons = [{'type': 'eq',
         'fun': lambda w: w.sum() - 1}]

# Then, I try to find the weights that will minimize my objective function.
# There are several solvers (methods) to choose from. I use SLSQP because
# it can handle constraints.
res = minimize(objective,
               w0,
               method='SLSQP',
               bounds=bounds,
               options={'disp':True, 'maxiter':10000},
               constraints=cons)

best_weights = res.x

print('\nOptimized weights:')
print('Model 1: {:.4f}'.format(best_weights[0]))
print('Model 2: {:.4f}'.format(best_weights[1]))
# look at the results on the test set
# individual scores
print('Model 1 test score = {:.4f}'.format(mean_squared_error(y_test, mod1_predictions)))
print('Model 2 test score = {:.4f}'.format(mean_squared_error(y_test, mod2_predictions)))

# unoptimized ensemble
test_predictions = np.concatenate([mod1_predictions[:, None],
                                   mod2_predictions[:, None]], axis=1)
unoptimized_ensemble = np.average(test_predictions, axis=1, weights=w0)
print('Unoptimized ensemble test score: {:.4f}'.format(mean_squared_error(y_test,
                                                                          unoptimized_ensemble)))

# optimized ensemble
optimized_ensemble = np.average(test_predictions, axis=1, weights=best_weights)
print('Optimized ensemble test score: {:.4f}'.format(mean_squared_error(y_test,
                                                                        optimized_ensemble)))

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.scatter(X_test, y_test, color=None, edgecolor='b', alpha=.4)

X_plot = np.arange(X_test.min(), X_test.max() + 1).reshape(-1, 1)
line_1 = model_1.predict(X_plot)
line_2 = model_2.predict(X_plot)
blend = np.concatenate([line_1[:, None],
                        line_2[:, None]], axis=1)
line_ens = np.average(blend, axis=1, weights=w0)
line_opt = np.average(blend, axis=1, weights=best_weights)

ax.plot(X_plot, model_1.predict(X_plot), c='b', alpha=.7, label='model 1')
ax.plot(X_plot, model_2.predict(X_plot), c='r', alpha=.7, label='model 2')
ax.plot(X_plot, line_ens, c='orange', alpha=.7, label='unoptimized ensemble')
ax.plot(X_plot, line_opt, c='g', alpha=.7, label='optimized ensemble')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend()
ax.set_title('Comparison of regression lines of all models', fontsize=16)
ax.set_xlabel('Average number of rooms')
ax.set_ylabel('Median value of homes')
plt.show()