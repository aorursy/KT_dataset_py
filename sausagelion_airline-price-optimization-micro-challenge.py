import sys

sys.path.append('../input')

from flight_revenue_simulator import simulate_revenue, score_me

def pricing_function(days_left, tickets_left, demand_level):

    """Sample pricing function"""

    if days_left == 1:

        price = demand_level - tickets_left

    else:

        #we will sell tickets only if demand in in top demand_prop %

        #we find optimum demand prop based on derivarive

        days_prop = (tickets_left / 50 / days_left) ** 0.5

        allowed_demand = int(200 - days_prop*100)

        if demand_level >= allowed_demand:

            tickets_to_sell = int(tickets_left / days_left / days_prop)

            price = demand_level - tickets_to_sell

        else:

            #we are not in top demand prop, thus sell no tickets

            price = demand_level

    return price



score_me(pricing_function)
simulate_revenue(days_left=7, tickets_left=50, pricing_function=pricing_function, verbose=True)
score_me(pricing_function)