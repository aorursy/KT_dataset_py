import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split, GridSearchCV

# Boston housing price estimation dataset
Xf, yf = load_boston(True)

def experiment(fit_all):        
    # select randomly 20% of 506 samples ~ 100 samples
    _, X, _, y = train_test_split(Xf, yf, test_size=0.2)
    
    # 50 samples train / test, 50 samples final evaluation
    X, X_new, y, y_new = train_test_split(X, y, test_size=0.5)
    sc = StandardScaler()
    
    if fit_all:
        # fit on whole dataset
        X = sc.fit_transform(X, y)
    
    # Usual train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    if not fit_all:
        # a more proper fit: only on train part
        sc.fit(X_train)
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test)
    
    model = GridSearchCV(
        estimator=LinearSVR(),
        param_grid={
            'C': [10 ** i for i in [-3, -2, -1, 0]]
        }
    )
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    
    score_new = model.score(sc.transform(X_new), y_new)
    return abs(score - score_new)
from joblib import Parallel, delayed
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

# How many times to repeat experiment, in order to average out randomness
N_reps = 10000

for fit_on_test in [False, True]:        
    # run experiments in parallel
    errors = Parallel(n_jobs=-1, verbose=1)(delayed(experiment)(fit_on_test) for _ in range(N_reps))
    
    # estimate confidence bounds
    conf = bs.bootstrap(np.array(errors), stat_func=bs_stats.mean)
    
    # communicate results
    print("Average error of test estimate,",'fit on all' if fit_on_test else 'fit train only', ":")
    print(conf)
# Reproduce the results with script above
sizes = [50, 100, 200, 400]  # sizes of dataset, controlled with test_size=0.1
fit_on_train = [
    # mean, lower_bound, upper_bound
    [1.090, 0.992, 1.184],
    [0.269, 0.262, 0.275],
    [0.171, 0.168, 0.174],
    [0.117, 0.115, 0.119]
]
fit_on_all = [
    [3.318, 3.134, 3.480],
    [0.343, 0.333, 0.352],
    [0.173, 0.170, 0.176],
    [0.118, 0.116, 0.120]
]