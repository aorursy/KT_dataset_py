import sys
sys.path.append('../input')
from flight_revenue_simulator import simulate_revenue, score_me

import numpy as np
import math
import numpy as np
import math

# BOUNDS
MAX_DAYS_LEFT = 100
# let's assume there are a maximum of 100 tickets left at the beginning of the simulation
TICKETS_MAX = 100
DEMAND_LEVEL_MAX=199
DEMAND_LEVEL_MIN=100
DEMAND_LEVEL_RANGE=DEMAND_LEVEL_MAX-DEMAND_LEVEL_MIN+1

# list of 2 dimensional arrays -one for each value of days_left- containing
# optimal prices by tickets_left and demand_level
optimal_price_by_days_left = []
# list of 2 dimensional arrays -one for each value of days_left- containing
# optimal expected revenues by tickets_left and demand_level
optimal_expected_revenue_by_days_left = []
# 1 dimensional intermediate array containing expected optimal revenue at (days_left-1) for each value of tickets_left.
expected_optimal_revenue_n_1_by_tickets_left = []

def compute_optimal_price_and_revenue_0(tickets_left, demand_level):
    '''
    Compute optimal price and expected revenue at 1 day left for given tickets_left and demand_level
    :param tickets_left:
    :param demand_level:
    :return: optimal price and expected revenue at 1 day left
    '''
    if tickets_left >= demand_level / 2:
        return int(demand_level / 2.), math.ceil(demand_level / 2.) * math.floor(demand_level / 2.)
    else:
        return demand_level - tickets_left, (demand_level - tickets_left) * tickets_left


def compute_optimal_price_and_revenue(tickets_left, demand_level, days_left):
    '''
    Compute optimal price and expected revenue at a given days_left for given tickets_left and demand_level
    :param tickets_left:
    :param demand_level:
    :param days_left:
    :return: optimal price and expected revenue at days_left
    '''
    if days_left == 1:
        return compute_optimal_price_and_revenue_0(tickets_left, demand_level)

    min_p = max(0, demand_level - tickets_left)
    optimal_expected_revenue_n = -1
    optimal_price_n = -1
    for p in range(min_p, demand_level + 1):
        tickets_left_n_1 = tickets_left - (demand_level - p)
        r_day = (demand_level - p) * p
        expected_revenue_n = r_day + expected_optimal_revenue_n_1_by_tickets_left[tickets_left_n_1]
        if expected_revenue_n > optimal_expected_revenue_n:
            optimal_expected_revenue_n = expected_revenue_n
            optimal_price_n = p
    return optimal_price_n, optimal_expected_revenue_n


def build_price_and_revenue_tables(compute_optimal_price_and_revenue):
    '''
    Build optimal price and expected revenue tables given compute procedure
    :param compute_optimal_price_and_revenue: optimal price and expected revenue compute procedure
    :return: optimal price and expected revenue tables
    '''
    optimal_price = np.zeros((TICKETS_MAX + 1, DEMAND_LEVEL_RANGE), dtype=int)
    optimal_expected_revenue = np.zeros((TICKETS_MAX + 1, DEMAND_LEVEL_RANGE), dtype=float)
    for t in range(TICKETS_MAX + 1):
        for l in range(DEMAND_LEVEL_RANGE):
            optimal_price[t][l], optimal_expected_revenue[t][l] = compute_optimal_price_and_revenue(t, l + DEMAND_LEVEL_MIN)
    return optimal_price, optimal_expected_revenue


def compute_and_store_price_and_revenue_tables(days_left):
    '''
    Compute and store price and revenue tables at days_left
    :param days_left:
    '''
    print("Start computing tables for days left " + str(days_left))
    f = lambda tickets_left, demand_level: compute_optimal_price_and_revenue(tickets_left, demand_level, days_left)
    optimal_price, optimal_expected_revenue = build_price_and_revenue_tables(f)
    optimal_price_by_days_left.append(optimal_price)
    optimal_expected_revenue_by_days_left.append(optimal_expected_revenue)
    global expected_optimal_revenue_n_1_by_tickets_left
    expected_optimal_revenue_n_1_by_tickets_left = np.average(optimal_expected_revenue, axis=1)
    print("End computing tables for days left " + str(days_left))

for d in range(0, MAX_DAYS_LEFT):
    days_left = d + 1
    compute_and_store_price_and_revenue_tables(days_left)

# save tables
np.save("optimal_expected_revenue_by_days_left_v1.0.1", optimal_expected_revenue_by_days_left)
np.save("optimal_price_by_days_left_v1.0.1", optimal_price_by_days_left)
# load tables
optimal_expected_revenue_by_days_left = np.load("optimal_expected_revenue_by_days_left_v1.0.1.npy")
optimal_price_by_days_left = np.load("optimal_price_by_days_left_v1.0.1.npy")
def get_optimal_price_and_revenue_from_tables(tickets_left, demand_level, days_left):
    '''
    Get optimal price and expected revenue values from tables 
    :param tickets_left:
    :param demand_level:
    :param days_left:
    :return: optimal price and expected revenue values from tables
    '''
    if demand_level < DEMAND_LEVEL_MIN or demand_level > DEMAND_LEVEL_MAX:
        raise Exception("demand level " + str(demand_level) + "  not in the range ["+str(DEMAND_LEVEL_MIN)+", "+str(DEMAND_LEVEL_MAX)+"]")
    if tickets_left > TICKETS_MAX:
        raise Exception("tickets left " + str(tickets_left) + " are beyond assumed maximum of " + str(TICKETS_MAX))
    optimal_price = optimal_price_by_days_left[days_left - 1][tickets_left][demand_level - DEMAND_LEVEL_MIN]
    optimal_expected_revenue = optimal_expected_revenue_by_days_left[days_left - 1][tickets_left][demand_level - DEMAND_LEVEL_MIN]
    return optimal_price, optimal_expected_revenue
def basic_pricing_function(days_left, tickets_left, demand_level):
    """Sample pricing function"""
    price = demand_level - 10
    return price

def pricing_function(days_left, tickets_left, demand_level):
    # PS : here price as returned by the get_optimal_price_and_revenue_from_tables method is an integer
    # in the return statement, a small correction is brought to price in order to optimize the quantity_demanded computed in method _tickets_sold
    # and which depends on rounding : quantity_demanded = floor(max(0, demand_level - p))
    price, rev = get_optimal_price_and_revenue_from_tables(int(tickets_left), min(199, int(round(demand_level))), days_left)
    return price - (int(round(demand_level)) - demand_level)
simulate_revenue(days_left=5, tickets_left=50, pricing_function=basic_pricing_function, verbose=True)
score_me(pricing_function, 1000)
score_me(pricing_function, 200)