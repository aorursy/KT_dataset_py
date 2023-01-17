import sys
sys.path.append('../input')
from flight_revenue_simulator import simulate_revenue, score_me

def pricing_function(days_left, tickets_left, demand_level):
    """Sample pricing function"""
    price = demand_level - 10
    return price
simulate_revenue(days_left=7, tickets_left=50, pricing_function=pricing_function, verbose=True)
score_me(pricing_function)
def pricing_function(days_left, tickets_left, demand_level):
    """Sample pricing function"""
    price = demand_level - tickets_left / days_left
    return price
score_me(pricing_function)
import sys, io
import matplotlib.pyplot as plt
import numpy as np

log_hyperparam = 1
k_hyperparam = 0.01
def pricing_function(days_left, tickets_left, demand_level):
    """Sample pricing function"""
    log_term = log_hyperparam * (1 / (1 + np.exp(-k_hyperparam * (demand_level-150))) - 0.5)
    #print(logit)
    price = demand_level - tickets_left / days_left - log_term
    return price

def get_scores(hyperparam_steps):
    scores = []
    for i in hyperparam_steps:
        #remember default stdout
        old_stdout = sys.stdout
        #variable to store simulator's output
        out = io.StringIO()
        #redirect stdoutput
        sys.stdout = out

        global log_hyperparam
        log_hyperparam = i
        score_me(pricing_function)
        
        #put back stdoutput
        sys.stdout = old_stdout

        score = out.getvalue().split()[-1]
        score = float(score.replace('$', ''))

        scores.append(score)
        
    return scores
hyperparam_steps = np.linspace(0, 100, num=100)
test_scores = get_scores(hyperparam_steps)
#I have to find function's maximum manually because maximise doesn't work, probably because of tricks with redirecting of stdout
combined_scores = np.array((hyperparam_steps, test_scores))
argmax = combined_scores[1].argmax()
max_hyperparam = combined_scores[0,argmax]
max_revenue = combined_scores[1,argmax]
print(f"Maximum average revenue is: {max_revenue}, when sigmoid's hyperparameter is: {max_hyperparam}")
plt.plot(combined_scores[0], combined_scores[1])
hyperparam_steps = np.linspace(max_hyperparam-5, max_hyperparam+5, num=100)
test_scores = get_scores(hyperparam_steps)
combined_scores = np.array((hyperparam_steps, test_scores))
argmax = combined_scores[1].argmax()
max_hyperparam = combined_scores[0,argmax]
max_revenue = combined_scores[1,argmax]
print(f"Maximum average revenue is: {max_revenue}, when sigmoid's hyperparameter is: {max_hyperparam}")