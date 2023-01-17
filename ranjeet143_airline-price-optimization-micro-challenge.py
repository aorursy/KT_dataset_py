import sys

sys.path.append('../input')

from flight_revenue_simulator import simulate_revenue, score_me

def pricing_function(days_left, tickets_left, demand_level):

    

    if demand_level > 182:

        return demand_level - 10

    else:

        return demand_level - tickets_left/days_left
simulate_revenue(days_left=7, tickets_left=50, pricing_function=pricing_function, verbose=True)
score_me(pricing_function)