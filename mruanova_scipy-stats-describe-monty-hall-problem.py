# https://en.wikipedia.org/wiki/Monty_Hall_problem
# also
# https://pbpython.com/monte-carlo.html
import numpy as np
from scipy import stats
results = np.array([])
trials = 5000
for i in range(trials):
    # put the car behing a random door
    car = np.random.randint(1,4,1)
    # have the contestant pick a random door
    pick = np.random.randint(1,4,1)
    # open one of the doors that does not have the car and is not the contestants
    open_door = np.array([1,2,3])
    open_door = np.setdiff1d(open_door,[car,pick])
    open_door = np.random.choice(open_door,1)
    # the other door that remained closed
    other_closed_door = np.array([1,2,3])
    other_closed_door = np.setdiff1d(other_closed_door,[pick, open_door])
    # now you have two values, is the car in your door 'pick'
    # or is it in the remaining closed door and you should switch not 'pick' or 'open_door'?
    # we will say that you stay with your pick so 1 is if you win by staying and 0 if you lose
    if other_closed_door == car:
        results = np.append(results,0)
    else:
        results = np.append(results,1)
stats.describe(results)
