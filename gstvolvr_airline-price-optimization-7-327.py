import sys

sys.path.append('../input')

from flight_revenue_simulator import simulate_revenue, score_me
import numpy as np

def f(n, k): return np.math.factorial(n+k) // (np.math.factorial(n) * np.math.factorial(k))



def combos(n):

    """

    find all the combinations for 2 days

    """

    combos = 0

    items = set(range(n+1))

    for i in range(n+1):

        if (n - i) in items:

            combos += 1

    return combos



for i in range(5, 20):

    assert combos(i) == f(i, 1)
%%time

def pricing_function(days_left, tickets_left, demand_level):

    return demand_level - 10

score_me(pricing_function, sims_per_scenario=1000)
def pricing_function(days_left, tickets_left, demand_level):

    n_tickets = np.random.randint(0, tickets_left)

    return demand_level - n_tickets  

score_me(pricing_function, sims_per_scenario=1000)
%%time

def pricing_function(days_left, tickets_left, demand_level):

    return demand_level - tickets_left // days_left

score_me(pricing_function, sims_per_scenario=1000)
%%time

def pricing_function(days_left, tickets_left, demand_level):

    if days_left == 1:

        return demand_level - tickets_left

    if demand_level >= 150:

        n_tickets =  tickets_left // days_left

    else:

        n_tickets = 0

    return demand_level - n_tickets



score_me(pricing_function, sims_per_scenario=1000)
%%time 

import scipy as sp

import scipy.stats    

import scipy.special



def pmf(n, k, p):

    """

    pmf of binomial distribution

    """

    return sp.special.comb(n, k) * (p ** k) * (1 - p) ** (n - k)



def cdf(n, k, p):

    """

    cdf of biomial distribution

    """

    return sum([pmf(n, i, p) for i in range(0, k+1)])



def chance_of_higher_demand_tomorrow(demand_level):

    """

    inverse cdf of uniform distritbuion

    """

    mn, mx = 100, 200

    return 1 - (demand_level - mn) / (mx - mn)



def pricing_function(days_left, tickets_left, demand_level):

    expected_demand = 150

    p = chance_of_higher_demand_tomorrow(demand_level)    

    chance_of_higher_demand = 1 - cdf(days_left, 1, p) # chance of having at least one success over {days_left} with prob {p}

    

    if days_left == 1:

        n_tickets = tickets_left

    elif chance_of_higher_demand <= 0.95:

        n_tickets = tickets_left // days_left + ((1 - chance_of_higher_demand) / 3) * tickets_left



    else:

        n_tickets = 0

    return demand_level - n_tickets



score_me(pricing_function, sims_per_scenario=1000)