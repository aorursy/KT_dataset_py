import sys

sys.path.append('../input')

from flight_revenue_simulator import simulate_revenue, score_me

from math import exp





def pricing_function(days_left, tickets_left, demand_level):

    """Sample pricing function"""

    if days_left == 1:

        return demand_level - min(tickets_left, demand_level / 2)

    ticket_per_day = tickets_left / days_left

    price = demand_level - get_ticket_count_to_sell(demand_level, ticket_per_day)

    return price





def get_ticket_count_to_sell(demand_level, ticket_per_day):

    demand_coeff = demand_level / 150

    return ticket_per_day * demand_coeff * get_gradient_coeff(demand_coeff)





def get_gradient_coeff(demand_coeff):

    arg = demand_coeff

    return (exp(arg - 1) - exp(-arg + 1)) * 10 / (exp(arg - 1) + exp(-arg + 1)) + 1



simulate_revenue(days_left=7, tickets_left=50, pricing_function=pricing_function, verbose=True)
score_me(pricing_function)